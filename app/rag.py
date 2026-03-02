from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from openai import OpenAI
from pgvector.psycopg import Vector

from app.config import settings
from app.db import fetch_all, get_conn
from app.observability import get_tracer

_ATTR_INPUT_VALUE = "input.value"
_ATTR_OUTPUT_VALUE = "output.value"
_ATTR_INPUT_MIME = "input.mime_type"
_ATTR_OUTPUT_MIME = "output.mime_type"
_ATTR_SPAN_KIND = "openinference.span.kind"
_ATTR_LLM_MODEL_NAME = "llm.model_name"

_MAX_ATTR_CHARS = 4000


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def model_supports_explicit_temperature(model_name: str) -> bool:
    # Some newer nano models only accept the default temperature.
    return not model_name.startswith("gpt-5-nano")


def estimate_cost(prompt_tokens: int | None, completion_tokens: int | None) -> float:
    if prompt_tokens is None or completion_tokens is None:
        return 0.0
    cost_in = (prompt_tokens / 1_000_000) * settings.cost_input_per_1m
    cost_out = (completion_tokens / 1_000_000) * settings.cost_output_per_1m
    return round(cost_in + cost_out, 6)


def _truncate_attr_text(value: str, limit: int = _MAX_ATTR_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _json_attr(payload: Any) -> str:
    try:
        raw = json.dumps(payload, ensure_ascii=False)
    except Exception:
        raw = str(payload)
    return _truncate_attr_text(raw)


def _set_span_io(span: Any, *, input_payload: Any | None = None, output_payload: Any | None = None) -> None:
    if input_payload is not None:
        span.set_attribute(_ATTR_INPUT_VALUE, _json_attr(input_payload))
        span.set_attribute(_ATTR_INPUT_MIME, "application/json")
    if output_payload is not None:
        span.set_attribute(_ATTR_OUTPUT_VALUE, _json_attr(output_payload))
        span.set_attribute(_ATTR_OUTPUT_MIME, "application/json")


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "by",
    "with",
    "what",
    "which",
    "does",
    "do",
    "is",
    "are",
    "be",
    "about",
    "according",
    "guidance",
    "say",
}


def expand_query_for_regulatory_docs(question: str) -> str:
    q = question.strip()
    q_low = q.lower()
    expansions: list[str] = []
    acronym_map = {
        "ccs": "contamination control strategy",
        "gmp": "good manufacturing practice",
        "hepa": "high efficiency particulate air filter",
        "mhra": "medicines and healthcare products regulatory agency",
        "nhs": "national health service uk",
        "pqr": "product quality review",
        "qa": "quality assurance",
        "qc": "quality control",
        "ms licence": "specials manufacturer licence",
    }
    for key, value in acronym_map.items():
        if key in q_low:
            expansions.append(value)
    if "cleanroom" in q_low or "aseptic" in q_low:
        expansions.extend(["environmental monitoring", "contamination control", "sterile medicinal products"])
    if "trend" in q_low or "trending" in q_low:
        expansions.append("trend review environmental monitoring deviations complaints")

    if not expansions:
        return q
    return q + "\nRelated regulatory terms: " + "; ".join(dict.fromkeys(expansions))


def retrieve_chunks(query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    sql = (
        "SELECT chunk_id, source_id, name, url, section, type, text, metadata, "
        "1 - (embedding <=> %s) AS score "
        "FROM chunks "
        "ORDER BY embedding <=> %s "
        "LIMIT %s"
    )
    vec = Vector(query_embedding)
    return fetch_all(sql, (vec, vec, top_k))


def _query_terms(question: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9'-]+", question.lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}


def hybrid_rescore_chunks(question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    terms = _query_terms(question)
    if not chunks or not terms:
        return chunks[:top_k]

    rescored: list[dict[str, Any]] = []
    for c in chunks:
        section = str(c.get("section") or "").lower()
        text = str(c.get("text") or "").lower()
        hit_terms = 0
        section_hits = 0
        for t in terms:
            in_text = t in text
            in_section = t in section
            if in_text or in_section:
                hit_terms += 1
            if in_section:
                section_hits += 1

        lexical = hit_terms / max(1, len(terms))
        section_boost = min(0.25, 0.08 * section_hits)
        vector_score = float(c.get("score") or 0.0)
        combined = (0.78 * vector_score) + (0.22 * lexical) + section_boost

        c2 = dict(c)
        c2["vector_score"] = round(vector_score, 4)
        c2["lexical_score"] = round(lexical, 4)
        c2["score"] = round(combined, 4)
        rescored.append(c2)

    rescored.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return rescored[:top_k]


def rerank_chunks(client: OpenAI, question: str, chunks: list[dict[str, Any]], top_r: int) -> list[dict[str, Any]]:
    if not settings.rerank_enabled or top_r <= 0 or not chunks:
        return chunks[:top_r] if top_r else chunks

    # Trim chunk text to keep prompts bounded.
    items = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) > 500:
            text = text[:499] + "…"
        items.append(
            {
                "chunk_id": c.get("chunk_id"),
                "section": c.get("section"),
                "text": text,
            }
        )

    prompt = {
        "role": "system",
        "content": (
            "You are a relevance ranker for UK aseptic cleanroom and sterile manufacturing guidance. "
            "Score each chunk for how useful it is to answer the question accurately from the provided documents. "
            "Return JSON in the form {\"ranking\": [{\"chunk_id\": \"...\", \"score\": 0-100}, ...]}."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({"question": question, "chunks": items}, ensure_ascii=False),
    }

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [prompt, user],
        "response_format": {"type": "json_object"},
    }
    if model_supports_explicit_temperature(settings.llm_model):
        kwargs["temperature"] = 0
    response = client.chat.completions.create(**kwargs)

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
        md = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        page_start = md.get("page_start")
        page_end = md.get("page_end")
        page_label = ""
        if page_start is not None:
            page_label = f", p.{page_start}" if page_start == page_end else f", pp.{page_start}-{page_end}"
        label = f"{c.get('name','Source')} | {c.get('section','Section')}{page_label}"
        context_lines.append(f"[{i}] {label}\n{c.get('text','')}")
    context_text = "\n".join(context_lines)

    system = (
        "You are a grounded assistant for UK aseptic pharmacy and cleanroom regulatory guidance. "
        "Use only the provided context to answer. "
        "If the answer is not in the context, say you don't know. "
        "Cite sources using [n] after statements. Prefer precise wording for requirements and expectations."
    )
    user = f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(client: OpenAI, question: str, contexts: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    messages = build_prompt(question, contexts)
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
    }
    if model_supports_explicit_temperature(settings.llm_model):
        kwargs["temperature"] = 0
    response = client.chat.completions.create(**kwargs)
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

    with tracer.start_as_current_span("query") as root_span:
        root_span.set_attribute(_ATTR_SPAN_KIND, "CHAIN")
        _set_span_io(
            root_span,
            input_payload={"question": question, "top_k": top_k, "rerank_k": rerank_k},
        )
        root_span.set_attribute("query.id", query_id)

        start_total = time.perf_counter()
        with tracer.start_as_current_span("embed") as span:
            t0 = time.perf_counter()
            expanded_query = expand_query_for_regulatory_docs(question)
            span.set_attribute(_ATTR_SPAN_KIND, "EMBEDDING")
            _set_span_io(span, input_payload={"question": question}, output_payload={"expanded_query": expanded_query})
            query_embedding = embed_texts(client, [expanded_query])[0]
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000
            span.set_attribute("embedding.model", settings.embedding_model)
            span.set_attribute("embedding.query_expanded", expanded_query != question)
            span.set_attribute("embedding.vector_dim", len(query_embedding))

        with tracer.start_as_current_span("retrieve") as span:
            t0 = time.perf_counter()
            candidate_k = min(max(top_k * 3, top_k), 60)
            span.set_attribute(_ATTR_SPAN_KIND, "RETRIEVER")
            _set_span_io(
                span,
                input_payload={"question": question, "top_k": top_k, "candidate_k": candidate_k},
            )
            retrieved_raw = retrieve_chunks(query_embedding, candidate_k)
            retrieved = hybrid_rescore_chunks(question, retrieved_raw, top_k)
            timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000
            span.set_attribute("retrieval.top_k", top_k)
            span.set_attribute("retrieval.candidate_k", candidate_k)
            span.set_attribute("retrieval.count", len(retrieved))
            _set_span_io(
                span,
                output_payload={
                    "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved],
                    "sections": [c.get("section") for c in retrieved[:5]],
                },
            )

        with tracer.start_as_current_span("rerank") as span:
            t0 = time.perf_counter()
            rerank_k = min(rerank_k, len(retrieved)) if rerank_k else 0
            span.set_attribute(_ATTR_SPAN_KIND, "RERANKER")
            _set_span_io(
                span,
                input_payload={
                    "question": question,
                    "rerank_k": rerank_k,
                    "candidate_chunk_ids": [c.get("chunk_id") for c in retrieved],
                },
            )
            reranked = rerank_chunks(client, question, retrieved, rerank_k)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            span.set_attribute("rerank.k", rerank_k)
            span.set_attribute("rerank.count", len(reranked))
            _set_span_io(span, output_payload={"reranked_chunk_ids": [c.get("chunk_id") for c in reranked]})

        with tracer.start_as_current_span("generate") as span:
            t0 = time.perf_counter()
            span.set_attribute(_ATTR_SPAN_KIND, "LLM")
            _set_span_io(
                span,
                input_payload={
                    "question": question,
                    "contexts": [
                        {
                            "chunk_id": c.get("chunk_id"),
                            "section": c.get("section"),
                            "text": str(c.get("text") or "")[:500],
                        }
                        for c in reranked
                    ],
                },
            )
            answer, usage = generate_answer(client, question, reranked)
            timings["generate_ms"] = (time.perf_counter() - t0) * 1000
            span.set_attribute("llm.model", settings.llm_model)
            span.set_attribute(_ATTR_LLM_MODEL_NAME, settings.llm_model)
            span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens"))
            span.set_attribute("llm.completion_tokens", usage.get("completion_tokens"))
            _set_span_io(
                span,
                output_payload={
                    "answer": answer,
                    "usage": usage,
                },
            )

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
                    "vector_score": c.get("vector_score"),
                    "lexical_score": c.get("lexical_score"),
                    "metadata": c.get("metadata"),
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

        _set_span_io(
            root_span,
            output_payload={
                "query_id": query_id,
                "answer": answer,
                "retrieved_chunk_ids": response["retrieved_chunk_ids"],
                "reranked_chunk_ids": response["reranked_chunk_ids"],
                "usage": usage,
                "latency_ms": response["latency_ms"],
            },
        )

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
