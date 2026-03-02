from __future__ import annotations

import json
import time
import uuid
from typing import Any

from openai import OpenAI
from pgvector.psycopg import Vector

from app.config import settings
from app.db import fetch_all, get_conn
from app.observability import get_tracer


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def estimate_cost(prompt_tokens: int | None, completion_tokens: int | None) -> float:
    if prompt_tokens is None or completion_tokens is None:
        return 0.0
    cost_in = (prompt_tokens / 1_000_000) * settings.cost_input_per_1m
    cost_out = (completion_tokens / 1_000_000) * settings.cost_output_per_1m
    return round(cost_in + cost_out, 6)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def retrieve_chunks(query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    sql = (
        "SELECT chunk_id, source_id, name, url, section, type, text, "
        "1 - (embedding <=> %s) AS score "
        "FROM chunks "
        "ORDER BY embedding <=> %s "
        "LIMIT %s"
    )
    vec = Vector(query_embedding)
    return fetch_all(sql, (vec, vec, top_k))


def rerank_chunks(client: OpenAI, question: str, chunks: list[dict[str, Any]], top_r: int) -> list[dict[str, Any]]:
    if not settings.rerank_enabled or top_r <= 0 or not chunks:
        return chunks[:top_r] if top_r else chunks

    # Trim chunk text to keep prompts bounded.
    items = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) > 500:
            text = text[:499] + "…"
        items.append({"chunk_id": c.get("chunk_id"), "text": text})

    prompt = {
        "role": "system",
        "content": (
            "You are a relevance ranker. Score each chunk for how useful it is to answer the question. "
            "Return JSON in the form {\"ranking\": [{\"chunk_id\": \"...\", \"score\": 0-100}, ...]}."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({"question": question, "chunks": items}, ensure_ascii=False),
    }

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[prompt, user],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        ranked = data.get("ranking") or data.get("ranks") or data.get("scores")
    except json.JSONDecodeError:
        ranked = None

    if not ranked:
        return chunks[:top_r]

    score_map = {item["chunk_id"]: item.get("score", 0) for item in ranked if "chunk_id" in item}
    chunks_sorted = sorted(chunks, key=lambda c: score_map.get(c.get("chunk_id"), 0), reverse=True)
    return chunks_sorted[:top_r]


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    context_lines = []
    for i, c in enumerate(contexts, start=1):
        context_lines.append(f"[{i}] {c.get('text','')}")
    context_text = "\n".join(context_lines)

    system = (
        "You are a grounded assistant. Use only the provided context to answer. "
        "If the answer is not in the context, say you don't know. "
        "Cite sources using [n] after statements."
    )
    user = f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(client: OpenAI, question: str, contexts: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    messages = build_prompt(question, contexts)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content or ""
    usage = response.usage
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return answer, usage_dict


def answer_query(question: str, top_k: int | None = None, rerank_k: int | None = None) -> dict[str, Any]:
    tracer = get_tracer("rag")
    client = get_openai_client()

    query_id = str(uuid.uuid4())
    top_k = top_k or settings.top_k
    rerank_k = rerank_k or settings.rerank_k

    timings: dict[str, float] = {}

    start_total = time.perf_counter()
    with tracer.start_as_current_span("embed") as span:
        t0 = time.perf_counter()
        query_embedding = embed_texts(client, [question])[0]
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000
        span.set_attribute("embedding.model", settings.embedding_model)

    with tracer.start_as_current_span("retrieve") as span:
        t0 = time.perf_counter()
        retrieved = retrieve_chunks(query_embedding, top_k)
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        span.set_attribute("retrieval.top_k", top_k)
        span.set_attribute("retrieval.count", len(retrieved))

    with tracer.start_as_current_span("rerank") as span:
        t0 = time.perf_counter()
        rerank_k = min(rerank_k, len(retrieved)) if rerank_k else 0
        reranked = rerank_chunks(client, question, retrieved, rerank_k)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
        span.set_attribute("rerank.k", rerank_k)
        span.set_attribute("rerank.count", len(reranked))

    with tracer.start_as_current_span("generate") as span:
        t0 = time.perf_counter()
        answer, usage = generate_answer(client, question, reranked)
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000
        span.set_attribute("llm.model", settings.llm_model)
        span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens"))
        span.set_attribute("llm.completion_tokens", usage.get("completion_tokens"))

    timings["total_ms"] = (time.perf_counter() - start_total) * 1000

    cost = estimate_cost(usage.get("prompt_tokens"), usage.get("completion_tokens"))

    response = {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "contexts": [
            {
                "chunk_id": c.get("chunk_id"),
                "source_id": c.get("source_id"),
                "name": c.get("name"),
                "url": c.get("url"),
                "section": c.get("section"),
                "score": c.get("score"),
                "text": c.get("text"),
            }
            for c in reranked
        ],
        "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved],
        "reranked_chunk_ids": [c.get("chunk_id") for c in reranked],
        "latency_ms": {k: round(v, 2) for k, v in timings.items()},
        "usage": usage,
        "cost_usd": cost,
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "top_k": top_k,
        "rerank_k": rerank_k,
    }

    log_query(response)
    return response


def log_query(payload: dict[str, Any]) -> None:
    sql = (
        "INSERT INTO query_logs "
        "(query_id, query, answer, model, embed_model, top_k, rerank_k, "
        "latencies_ms, tokens_in, tokens_out, cost_usd, retrieved_chunk_ids, reranked_chunk_ids) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)"
    )

    usage = payload.get("usage", {})
    params = (
        payload.get("query_id"),
        payload.get("question"),
        payload.get("answer"),
        payload.get("model"),
        payload.get("embedding_model"),
        payload.get("top_k"),
        payload.get("rerank_k"),
        json.dumps(payload.get("latency_ms"), ensure_ascii=False),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        payload.get("cost_usd"),
        payload.get("retrieved_chunk_ids"),
        payload.get("reranked_chunk_ids"),
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
