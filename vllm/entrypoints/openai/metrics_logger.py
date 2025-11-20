# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import time
from collections import deque
from threading import Condition, Event, Lock, Thread
from typing import Optional, cast

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from vllm.logger import init_logger

logger = init_logger(__name__)

_MICROSECONDS_PER_SECOND = 1_000_000
_DEFAULT_BATCH_SIZE = 500
_DEFAULT_FLUSH_INTERVAL_SECONDS = 5.0
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_DEFAULT_RETRY_MAX_DELAY_SECONDS = 60.0
_MIN_FLUSH_INTERVAL_SECONDS = 0.1
_MIN_RETRY_DELAY_SECONDS = 0.1


def to_microseconds(timestamp_seconds: float) -> int:
    return int(timestamp_seconds * _MICROSECONDS_PER_SECOND)


MetricsRow = tuple[str, int, int, int, int, int, int]


def _read_positive_int(env_var: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < minimum:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning(
            "Invalid value for %s=%s; using default %d", env_var, value, default
        )
        return default


def _read_positive_float(env_var: str, default: float, minimum: float) -> float:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        parsed = float(value)
        if parsed < minimum:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning(
            "Invalid value for %s=%s; using default %.2f",
            env_var,
            value,
            default,
        )
        return default


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
        "_batch_size",
        "_flush_interval",
        "_retry_base_delay",
        "_retry_max_delay",
        "_buffer",
        "_buffer_condition",
        "_worker_thread",
        "_shutdown_event",
        "_last_flush_time",
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
        self._batch_size = _read_positive_int(
            "INFERENCE_DB_BATCH_SIZE", _DEFAULT_BATCH_SIZE
        )
        self._flush_interval = _read_positive_float(
            "INFERENCE_DB_FLUSH_INTERVAL_SECONDS",
            _DEFAULT_FLUSH_INTERVAL_SECONDS,
            _MIN_FLUSH_INTERVAL_SECONDS,
        )
        self._retry_base_delay = _read_positive_float(
            "INFERENCE_DB_RETRY_BASE_DELAY_SECONDS",
            _DEFAULT_RETRY_BASE_DELAY_SECONDS,
            _MIN_RETRY_DELAY_SECONDS,
        )
        self._retry_max_delay = max(
            self._retry_base_delay,
            _read_positive_float(
                "INFERENCE_DB_RETRY_MAX_DELAY_SECONDS",
                _DEFAULT_RETRY_MAX_DELAY_SECONDS,
                self._retry_base_delay,
            ),
        )
        self._buffer: deque[MetricsRow] = deque()
        self._buffer_condition = Condition(self._write_lock)
        self._shutdown_event = Event()
        self._last_flush_time = time.monotonic()
        self._insert_statement = sql.SQL(
            """
            INSERT INTO {schema}.{table} (
                "RequestID",
                "StartGenerationAt",
                "RespondFirstTokenAt",
                "RespondLastTokenAt",
                "AggregatedQueryHits",
                "GeneratedTokens",
                "PromptTokens"
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ("RequestID") DO UPDATE SET
                "StartGenerationAt" = EXCLUDED."StartGenerationAt",
                "RespondFirstTokenAt" = EXCLUDED."RespondFirstTokenAt",
                "RespondLastTokenAt" = EXCLUDED."RespondLastTokenAt",
                "AggregatedQueryHits" = EXCLUDED."AggregatedQueryHits",
                "GeneratedTokens" = EXCLUDED."GeneratedTokens",
                "PromptTokens" = EXCLUDED."PromptTokens"
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

        self._worker_thread = Thread(
            target=self._flush_worker,
            name="PostgresMetricsLoggerWorker",
            daemon=True,
        )
        self._worker_thread.start()
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
        n_generated_tokens: int,
        num_prompt_tokens: int,
    ) -> None:
        if psycopg is None or self._closed:
            return

        row: MetricsRow = (
            request_id,
            start_generation_at_us,
            respond_first_token_at_us,
            respond_last_token_at_us,
            prefix_hits,
            n_generated_tokens,
            num_prompt_tokens,
        )

        with self._buffer_condition:
            if self._closed:
                return
            self._buffer.append(row)
            self._buffer_condition.notify()

    async def log_async(
        self,
        request_id: str,
        start_generation_at_us: int,
        respond_first_token_at_us: int,
        respond_last_token_at_us: int,
        prefix_hits: int,
        n_generated_tokens: int,
        num_prompt_tokens: int,
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
                n_generated_tokens,
                num_prompt_tokens,
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
        n_generated_tokens: int,
        num_prompt_tokens: int,
    ) -> None:
        try:
            self.log(
                request_id,
                start_generation_at_us,
                respond_first_token_at_us,
                respond_last_token_at_us,
                prefix_hits,
                n_generated_tokens,
                num_prompt_tokens,
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
        self._shutdown_event.set()
        with self._buffer_condition:
            self._buffer_condition.notify_all()
        with self._tasks_lock:
            pending = list(self._pending_tasks)
            self._pending_tasks.clear()
        for task in pending:
            task.cancel()
        if hasattr(self, "_worker_thread") and self._worker_thread is not None:
            self._worker_thread.join()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.exception("Error closing PostgreSQL metrics connection")
        global _METRICS_LOGGER, _METRICS_LOGGER_INITIALIZED
        if _METRICS_LOGGER is self:
            _METRICS_LOGGER = None
            _METRICS_LOGGER_INITIALIZED = False

    def _ensure_connection(self):
        if self._conn is not None and not self._conn.closed:
            return self._conn
        self._conn = psycopg.connect(self._conninfo)
        self._conn.autocommit = False
        return self._conn

    def _handle_connection_failure(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.rollback()
        except Exception:
            logger.exception(
                "Failed to rollback PostgreSQL metrics transaction during retry"
            )
        try:
            self._conn.close()
        except Exception:
            logger.exception(
                "Error closing PostgreSQL metrics connection after failure"
            )
        self._conn = None

    def _flush_worker(self) -> None:
        while True:
            with self._buffer_condition:
                while True:
                    if self._shutdown_event.is_set() and not self._buffer:
                        return
                    buffer_len = len(self._buffer)
                    if buffer_len >= self._batch_size:
                        break
                    if buffer_len > 0:
                        elapsed = time.monotonic() - self._last_flush_time
                        if elapsed >= self._flush_interval:
                            break
                        timeout = max(self._flush_interval - elapsed, _MIN_FLUSH_INTERVAL_SECONDS)
                        self._buffer_condition.wait(timeout)
                    else:
                        self._buffer_condition.wait()
                flush_count = min(len(self._buffer), self._batch_size)
                batch = [self._buffer.popleft() for _ in range(flush_count)]
            self._flush_batch_with_retry(batch)

    def _flush_batch_with_retry(self, batch: list[MetricsRow]) -> None:
        if not batch:
            return
        attempt = 0
        delay = self._retry_base_delay
        while True:
            attempt += 1
            try:
                conn = self._ensure_connection()
                with conn.cursor() as cursor:
                    cursor.executemany(self._insert_statement, batch)
                conn.commit()
                self._last_flush_time = time.monotonic()
                return
            except Exception:
                logger.exception(
                    "Failed to flush %d PostgreSQL metrics rows (attempt %d); retrying in %.1fs",
                    len(batch),
                    attempt,
                    delay,
                )
                self._handle_connection_failure()
                time.sleep(delay)
                delay = min(delay * 2, self._retry_max_delay)


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
