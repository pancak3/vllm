# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import atexit
import os
import sys
from threading import Lock
from typing import Optional, cast

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from vllm.logger import init_logger

logger = init_logger(__name__)

_MICROSECONDS_PER_SECOND = 1_000_000


def to_microseconds(timestamp_seconds: float) -> int:
    return int(timestamp_seconds * _MICROSECONDS_PER_SECOND)


class PostgresMetricsLogger:
    __slots__ = (
        "_conninfo",
        "_schema",
        "_table",
        "_insert_statement",
        "_conn",
        "_write_lock",
        "_closed",
        "_pending_tasks",
        "_tasks_lock",
    )

    def __init__(self, conninfo: str, schema: str, table: str) -> None:
        assert psycopg is not None and sql is not None
        self._conninfo = conninfo
        self._schema = schema
        self._table = table
        self._write_lock = Lock()
        self._tasks_lock = Lock()
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._conn = None
        self._insert_statement = sql.SQL(
            """
            INSERT INTO {schema}.{table} (
                "RequestID",
                "StartGenerationAt",
                "RespondFirstTokenAt",
                "RespondLastTokenAt",
                "AggregatedQueryHits"
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT ("RequestID") DO UPDATE SET
                "StartGenerationAt" = EXCLUDED."StartGenerationAt",
                "RespondFirstTokenAt" = EXCLUDED."RespondFirstTokenAt",
                "RespondLastTokenAt" = EXCLUDED."RespondLastTokenAt",
                "AggregatedQueryHits" = EXCLUDED."AggregatedQueryHits"
            """
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table),
        )

        try:
            self._conn = psycopg.connect(self._conninfo)
            self._conn.autocommit = False
            with self._conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            self._conn.commit()
        except Exception as exc:  # pragma: no cover - connection issues
            print(
                f"[PostgresMetricsLogger] Failed to connect to PostgreSQL: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            sys.exit(1)
        else:
            print(
                (
                    "[PostgresMetricsLogger] Connected to "
                    f"{self._schema}.{self._table}"
                ),
                flush=True,
            )

        atexit.register(self.close)

    @classmethod
    def from_env(cls) -> Optional[PostgresMetricsLogger]:
        if psycopg is None or sql is None or make_conninfo is None:
            logger.warning(
                "psycopg is not installed; skipping PostgreSQL metrics logging"
            )
            return None

        db_url = os.getenv("INFERENCE_DB_URL")
        username = os.getenv("INFERENCE_DB_USERNAME")
        password = os.getenv("INFERENCE_DB_PASSWORD")
        dbname = os.getenv("INFERENCE_DB_NAME")
        port = os.getenv("INFERENCE_DB_PORT")
        schema = os.getenv("INFERENCE_DB_SCHEMA")
        table = os.getenv("INFERENCE_DB_TABLE")

        if not all((db_url, username, password, dbname, schema, table)):
            logger.info(
                "PostgreSQL metrics logging disabled: required environment "
                "variables are missing"
            )
            return None

        assert db_url and username and password and dbname and schema and table

        conninfo_kwargs = {
            "user": username,
            "password": password,
            "dbname": dbname,
        }
        if port:
            conninfo_kwargs["port"] = port

        if "://" in db_url or "=" in db_url:
            conninfo = make_conninfo(
                db_url,
                **conninfo_kwargs,
            )
        else:
            conninfo = make_conninfo(
                host=db_url,
                **conninfo_kwargs,
            )

        return cls(conninfo, cast(str, schema), cast(str, table))

    def log(
        self,
        request_id: str,
        start_generation_at_us: int,
        respond_first_token_at_us: int,
        respond_last_token_at_us: int,
        prefix_hits: int,
    ) -> None:
        if psycopg is None or self._closed:
            return

        with self._write_lock:
            if self._closed or self._conn is None:
                return
            try:
                with self._conn.cursor() as cursor:
                    cursor.execute(
                        self._insert_statement,
                        (
                            request_id,
                            start_generation_at_us,
                            respond_first_token_at_us,
                            respond_last_token_at_us,
                            prefix_hits,
                        ),
                    )
                self._conn.commit()
            except Exception:
                if self._conn is not None:
                    try:
                        self._conn.rollback()
                    except Exception:
                        logger.exception(
                            "Failed to rollback PostgreSQL metrics transaction"
                        )
                raise

    async def log_async(
        self,
        request_id: str,
        start_generation_at_us: int,
        respond_first_token_at_us: int,
        respond_last_token_at_us: int,
        prefix_hits: int,
    ) -> None:
        if psycopg is None or self._closed:
            return

        loop = asyncio.get_running_loop()

        task = loop.create_task(
            self._log_async_runner(
                request_id,
                start_generation_at_us,
                respond_first_token_at_us,
                respond_last_token_at_us,
                prefix_hits,
            )
        )
        task.add_done_callback(self._log_task_done)
        with self._tasks_lock:
            if self._closed:
                task.cancel()
            else:
                self._pending_tasks.add(task)

    async def _log_async_runner(
        self,
        request_id: str,
        start_generation_at_us: int,
        respond_first_token_at_us: int,
        respond_last_token_at_us: int,
        prefix_hits: int,
    ) -> None:
        try:
            await asyncio.to_thread(
                self.log,
                request_id,
                start_generation_at_us,
                respond_first_token_at_us,
                respond_last_token_at_us,
                prefix_hits,
            )
        except Exception:
            logger.exception("Failed to log inference metrics to PostgreSQL")

    def _log_task_done(self, task: asyncio.Task[None]) -> None:
        with self._tasks_lock:
            self._pending_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception("Failed to log inference metrics to PostgreSQL", exc_info=exc)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        with self._tasks_lock:
            pending = list(self._pending_tasks)
            self._pending_tasks.clear()
        for task in pending:
            task.cancel()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.exception("Error closing PostgreSQL metrics connection")
        global _METRICS_LOGGER, _METRICS_LOGGER_INITIALIZED
        if _METRICS_LOGGER is self:
            _METRICS_LOGGER = None
            _METRICS_LOGGER_INITIALIZED = False


_METRICS_LOGGER_LOCK = Lock()
_METRICS_LOGGER_INITIALIZED = False
_METRICS_LOGGER: Optional[PostgresMetricsLogger] = None


def get_inference_metrics_logger() -> Optional[PostgresMetricsLogger]:
    global _METRICS_LOGGER_INITIALIZED, _METRICS_LOGGER

    if _METRICS_LOGGER_INITIALIZED:
        return _METRICS_LOGGER

    with _METRICS_LOGGER_LOCK:
        if not _METRICS_LOGGER_INITIALIZED:
            _METRICS_LOGGER = PostgresMetricsLogger.from_env()
            _METRICS_LOGGER_INITIALIZED = True

    return _METRICS_LOGGER
