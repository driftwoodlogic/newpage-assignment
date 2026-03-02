from __future__ import annotations

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings


pool = ConnectionPool(conninfo=settings.database_url, max_size=10)


@contextmanager
def get_conn():
    with pool.connection() as conn:
        register_vector(conn)
        yield conn


def execute(sql: str, params: tuple | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def fetch_all(sql: str, params: tuple | None = None) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()
