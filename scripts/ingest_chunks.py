#!/usr/bin/env python3
"""Ingest UK aseptic cleanroom PDF corpus chunks into Postgres with OpenAI embeddings."""

from __future__ import annotations

import argparse
import json
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
from app.pipeline_defaults import DEFAULT_INGEST_BATCH_SIZE


STANDARD_CHUNK_KEYS = {
    "chunk_id",
    "source_id",
    "name",
    "url",
    "section",
    "type",
    "text",
    "field_name",
    "fact_value",
}


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


def insert_documents(conn: psycopg.Connection, docs_path: Path) -> int:
    rows = []
    for rec in read_jsonl(docs_path):
        raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else rec
        rows.append(
            (
                str(rec.get("source_id")),
                rec.get("name"),
                rec.get("url"),
                json.dumps(raw, ensure_ascii=False),
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (source_id, name, url, raw) VALUES (%s, %s, %s, %s::jsonb) "
            "ON CONFLICT (source_id) DO UPDATE SET name = EXCLUDED.name, url = EXCLUDED.url, raw = EXCLUDED.raw",
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
        texts = [str(c.get("text", "")) for c in batch]
        embeddings = embed_texts(client, texts)

        rows = []
        for c, emb in zip(batch, embeddings):
            metadata = {k: v for k, v in c.items() if k not in STANDARD_CHUNK_KEYS}
            rows.append(
                (
                    c.get("chunk_id"),
                    c.get("source_id"),
                    c.get("name"),
                    c.get("url"),
                    c.get("section"),
                    c.get("type") or "pdf_chunk",
                    c.get("field_name"),
                    c.get("fact_value"),
                    c.get("text"),
                    Vector(emb),
                    json.dumps(metadata, ensure_ascii=False),
                )
            )

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (chunk_id, source_id, name, url, section, type, field_name, fact_value, text, embedding, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "source_id = EXCLUDED.source_id, name = EXCLUDED.name, url = EXCLUDED.url, section = EXCLUDED.section, "
                "type = EXCLUDED.type, field_name = EXCLUDED.field_name, fact_value = EXCLUDED.fact_value, "
                "text = EXCLUDED.text, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                rows,
            )
        conn.commit()

        inserted += len(rows)
        print(f"Embedded + upserted {inserted}/{total} chunks")
        time.sleep(0.2)

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest UK aseptic cleanroom PDF docs + chunks into Postgres")
    parser.add_argument("--documents", default="data/processed/documents.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INGEST_BATCH_SIZE)
    args = parser.parse_args()

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required for embeddings")

    conn = connect()
    try:
        doc_count = insert_documents(conn, Path(args.documents))
        print(f"Upserted documents: {doc_count}")

        chunk_count = insert_chunks(conn, Path(args.chunks), args.batch_size)
        print(f"Upserted chunks: {chunk_count}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
