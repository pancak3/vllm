# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
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
    __slots__ = ("_conninfo", "_schema", "_table", "_insert_statement")

    def __init__(self, conninfo: str, schema: str, table: str) -> None:
        assert psycopg is not None and sql is not None
        self._conninfo = conninfo
        self._schema = schema
        self._table = table
        self._insert_statement = sql.SQL(
            """
            INSERT INTO {schema}.{table} (
                "RequestID",
                "StartGenerationAt",
                "RespondFirstTokenAt",
                "RespondLastTokenAt"
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT ("RequestID") DO UPDATE SET
                "StartGenerationAt" = EXCLUDED."StartGenerationAt",
                "RespondFirstTokenAt" = EXCLUDED."RespondFirstTokenAt",
                "RespondLastTokenAt" = EXCLUDED."RespondLastTokenAt"
            """
        ).format(
            schema=sql.Identifier(schema),
            table=sql.Identifier(table),
        )

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
    ) -> None:
        if psycopg is None:
            return

        try:
            with psycopg.connect(self._conninfo) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        self._insert_statement,
                        (
                            request_id,
                            start_generation_at_us,
                            respond_first_token_at_us,
                            respond_last_token_at_us,
                        ),
                    )
                conn.commit()
        except Exception:
            logger.exception("Failed to log inference metrics to PostgreSQL")


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
