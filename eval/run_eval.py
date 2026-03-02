#!/usr/bin/env python3
"""Run retrieval/API eval against the local RAG API and write a summary JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import httpx
from tqdm import tqdm

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalise_contexts(resp: dict[str, Any]) -> list[str]:
    raw = resp.get("contexts") or []
    if not isinstance(raw, list):
        return []

    out: list[str] = []
    for c in raw:
        if isinstance(c, str) and c.strip():
            out.append(c)
        elif isinstance(c, dict):
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                out.append(t)
    return out


async def call_api(
    client: httpx.AsyncClient,
    url: str,
    question: str,
    *,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = await client.post(url, json={"query": question})
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("API response JSON is not an object")
            return data

        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            last_exc = exc
            if attempt >= retries:
                break

            retry_after = None
            if isinstance(exc, httpx.HTTPStatusError):
                ra = exc.response.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        retry_after = None

            sleep_for = retry_after or (backoff * (2**attempt) + random.uniform(0, backoff))
            await asyncio.sleep(sleep_for)

    if last_exc:
        raise last_exc
    raise RuntimeError("API call failed without exception")


def compute_retrieval_metrics(
    gold_ids: list[str] | None,
    retrieved_ids: list[str] | None,
) -> dict[str, float]:
    gold = set(gold_ids or [])
    retrieved = set(retrieved_ids or [])

    if not retrieved:
        return {"precision": 0.0, "recall": 0.0}

    hit = len(gold & retrieved)
    precision = hit / len(retrieved)
    recall = hit / len(gold) if gold else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float("nan") if not vals else sum(vals) / len(vals)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


async def _process_item(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    item: dict[str, Any],
    args: argparse.Namespace,
    pbar: tqdm,
) -> dict[str, Any] | None:
    q = item.get("question")
    if not isinstance(q, str) or not q.strip():
        pbar.update(1)
        return None

    async with sem:
        resp = await call_api(client, args.api, q, retries=args.retries, backoff=args.backoff)

    retrieved_ids = resp.get("retrieved_chunk_ids") or []
    reranked_ids = resp.get("reranked_chunk_ids") or []
    contexts = _normalise_contexts(resp)
    # recall: did the retrieval stage surface the gold chunk at all?
    # recall_reranked: did the gold chunk survive into the context sent to the LLM?
    recall_metrics = compute_retrieval_metrics(item.get("gold_chunk_ids"), retrieved_ids)
    reranked_metrics = compute_retrieval_metrics(item.get("gold_chunk_ids"), reranked_ids)
    latency = resp.get("latency_ms") if isinstance(resp.get("latency_ms"), dict) else {}
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}

    pbar.update(1)
    return {
        "question_id": item.get("id"),
        "question": q,
        "gold_chunk_ids": item.get("gold_chunk_ids") or [],
        "retrieved_chunk_ids": retrieved_ids,
        "reranked_chunk_ids": reranked_ids,
        "recall_reranked": reranked_metrics["recall"],
        "recall": recall_metrics["recall"],
        "answer": resp.get("answer"),
        "contexts": contexts,
        "gold_context": item.get("gold_context"),
        "gold_answer": item.get("gold_answer"),
        "latency_ms": latency,
        "cost_usd": resp.get("cost_usd"),
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
    }


async def run_eval_pass(
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
    limits = httpx.Limits(
        max_connections=args.concurrency + 5,
        max_keepalive_connections=args.concurrency,
    )

    with tqdm(total=len(questions), desc="Evaluating", unit="q") as pbar:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [_process_item(sem, client, item, args, pbar) for item in questions]
            raw = await asyncio.gather(*tasks)

    return [r for r in raw if r is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval/API eval against the RAG API")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--api", default="http://localhost:8000/query")
    parser.add_argument("--out")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    args = parser.parse_args()

    questions = read_jsonl(Path(args.questions))
    if not questions:
        raise SystemExit("No questions loaded")

    results = asyncio.run(run_eval_pass(questions, args))

    recall_avg = safe_mean([r["recall"] for r in results])
    recall_reranked_avg = safe_mean([r["recall_reranked"] for r in results])

    total_latencies = [float((r.get("latency_ms") or {}).get("total_ms")) for r in results if (r.get("latency_ms") or {}).get("total_ms") is not None]
    cost_vals = [float(r["cost_usd"]) for r in results if isinstance(r.get("cost_usd"), (int, float))]

    summary = {
        "run_id": str(uuid.uuid4()),
        "count": len(results),
        "recall_avg": round(recall_avg, 4),
        "recall_reranked_avg": round(recall_reranked_avg, 4),
    }
    if total_latencies:
        summary["latency_p95_ms"] = round(percentile(total_latencies, 0.95), 2)
        summary["latency_avg_ms"] = round(safe_mean(total_latencies), 2)
    if cost_vals:
        summary["cost_avg_usd"] = round(safe_mean(cost_vals), 6)

    summary_json = json.dumps(summary, indent=2)
    print(summary_json)

    if args.out:
        Path(args.out).write_text(summary_json + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
