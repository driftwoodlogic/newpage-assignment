#!/usr/bin/env python3
"""Ingest people + chunks into Postgres with OpenAI embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg
from openai import OpenAI
from pgvector.psycopg import Vector, register_vector

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config import settings


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunked(seq: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def connect():
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def insert_documents(conn: psycopg.Connection, people_path: Path) -> int:
    rows = []
    for rec in read_jsonl(people_path):
        source_id = str(rec.get("identifier", rec.get("name")))
        rows.append((source_id, rec.get("name"), rec.get("url"), json.dumps(rec)))

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (source_id, name, url, raw) VALUES (%s, %s, %s, %s::jsonb) "
            "ON CONFLICT (source_id) DO NOTHING",
            rows,
        )
    conn.commit()
    return len(rows)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def insert_chunks(conn: psycopg.Connection, chunks_path: Path, batch_size: int) -> int:
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    chunks = list(read_jsonl(chunks_path))
    total = len(chunks)
    inserted = 0

    for batch in chunked(chunks, batch_size):
        texts = [c.get("text", "") for c in batch]
        embeddings = embed_texts(client, texts)

        rows = []
        for c, emb in zip(batch, embeddings):
            rows.append(
                (
                    c.get("chunk_id"),
                    c.get("source_id"),
                    c.get("name"),
                    c.get("url"),
                    c.get("section"),
                    c.get("type"),
                    c.get("field_name"),
                    c.get("fact_value"),
                    c.get("text"),
                    Vector(emb),
                    json.dumps({"source": "wikipedia-people"}),
                )
            )

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (chunk_id, source_id, name, url, section, type, field_name, fact_value, text, embedding, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (chunk_id) DO NOTHING",
                rows,
            )
        conn.commit()

        inserted += len(rows)
        print(f"Inserted {inserted}/{total} chunks")
        time.sleep(0.2)

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest people + chunks into Postgres")
    parser.add_argument("--people", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for embeddings")

    conn = connect()
    try:
        doc_count = insert_documents(conn, Path(args.people))
        print(f"Inserted documents: {doc_count}")

        chunk_count = insert_chunks(conn, Path(args.chunks), args.batch_size)
        print(f"Inserted chunks: {chunk_count}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
