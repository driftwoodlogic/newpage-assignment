#!/usr/bin/env python3
"""Run eval against the API and store metrics."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import httpx
import psycopg
from tqdm import tqdm
from openai import OpenAI

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config import settings

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall
    from ragas.metrics import faithfulness, context_precision, context_recall
    from ragas.llms import llm_factory
except Exception as exc:
    raise RuntimeError("RAGAS + datasets required. Install deps with: uv sync") from exc


# ----------------------------- IO ---------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ----------------------------- API --------------------------------

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


def call_api(
    client: httpx.Client,
    url: str,
    question: str,
    *,
    retries: int,
    backoff: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = client.post(url, json={"query": question})
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
            time.sleep(sleep_for)

    if last_exc:
        raise last_exc
    raise RuntimeError("API call failed without exception")


# --------------------------- Metrics -------------------------------

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


# ---------------------------- RAGAS -------------------------------

def _agg(x):
    if isinstance(x, list):
        return safe_mean(x)
    return float(x)

def build_ragas_dataset(rows: list[dict[str, Any]]) -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    for r in rows:
        q = str(r.get("question") or "").strip()
        a = str(r.get("answer") or "").strip()
        ctx = r.get("contexts") or []
        ref = r.get("ground_truth")

        if not q or not a:
            continue

        ctx2 = [c for c in ctx if isinstance(c, str) and c.strip()]

        refs: list[str] = []
        if isinstance(ref, str) and ref.strip():
            refs = [ref]
        elif isinstance(ref, list):
            refs = [x for x in ref if isinstance(x, str) and x.strip()]

        questions.append(q)
        answers.append(a)
        contexts.append(ctx2)
        ground_truths.append(refs)

    refs_flat = [r[0] if r else "" for r in ground_truths]
    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
            "reference": refs_flat,
        }
    )


def run_ragas(ragas_rows: list[dict[str, Any]]) -> dict[str, float]:
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=getattr(settings, "openai_base_url", None),
    )

    llm = llm_factory(settings.llm_model, client=client)

    metrics = [
        faithfulness,
        context_precision,
        context_recall,
    ]

    ds = build_ragas_dataset(ragas_rows)

    result = evaluate(ds, metrics=metrics, llm=llm)

    return {
        "faithfulness": round(_agg(result["faithfulness"]), 4),
        "context_precision": round(_agg(result["context_precision"]), 4),
        "context_recall": round(_agg(result["context_recall"]), 4),
    }


# ------------------------- Eval Pass -------------------------------

def run_eval_pass(
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    results: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []

    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    with httpx.Client(timeout=timeout, limits=limits) as client:
        for item in tqdm(questions, desc="Evaluating", unit="q"):
            q = item.get("question")
            if not isinstance(q, str) or not q.strip():
                continue

            resp = call_api(client, args.api, q, retries=args.retries, backoff=args.backoff)

            retrieved_ids = resp.get("retrieved_chunk_ids") or []
            reranked_ids = resp.get("reranked_chunk_ids") or []
            contexts = _normalise_contexts(resp)

            retrieval = compute_retrieval_metrics(item.get("gold_chunk_ids"), retrieved_ids)

            result = {
                "question_id": item.get("id"),
                "question": q,
                "gold_chunk_ids": item.get("gold_chunk_ids") or [],
                "retrieved_chunk_ids": retrieved_ids,
                "reranked_chunk_ids": reranked_ids,
                "precision": retrieval["precision"],
                "recall": retrieval["recall"],
                "answer": resp.get("answer"),
                "contexts": contexts,
                "gold_context": item.get("gold_context"),
                "gold_answer": item.get("gold_answer"),
            }

            results.append(result)

            ragas_rows.append(
                {
                    "question": q,
                    "answer": resp.get("answer"),
                    "contexts": contexts,
                    "ground_truth": item.get("gold_answer"),
                    # "reference": item.get("gold_answer"),
                }
            )

            if args.sleep:
                time.sleep(args.sleep)

    return results, ragas_rows


# ----------------------------- Main --------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval against the RAG API")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--api", default="http://localhost:8000/query")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--backoff", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--test-n", type=int, default=5, help="Preflight test set size")
    args = parser.parse_args()

    questions = read_jsonl(Path(args.questions))
    if not questions:
        raise SystemExit("No questions loaded")

    # ------------------ Preflight Test Run ------------------

    if args.test_n > 0:
        print(f"\nRunning full pipeline test on first {args.test_n} samples...\n")

        test_results, test_ragas_rows = run_eval_pass(
            questions[: args.test_n], args
        )

        _ = run_ragas(test_ragas_rows)

        print("\nPreflight test succeeded — running full dataset.\n")

    # ------------------ Full Run ------------------

    results, ragas_rows = run_eval_pass(questions, args)
    ragas_metrics = run_ragas(ragas_rows)

    precision_avg = safe_mean(r["precision"] for r in results)
    recall_avg = safe_mean(r["recall"] for r in results)

    summary = {
        "run_id": str(uuid.uuid4()),
        "count": len(results),
        "precision_avg": round(precision_avg, 4),
        "recall_avg": round(recall_avg, 4),
        **ragas_metrics,
    }

    summary_json = json.dumps(summary, indent=2)
    print(summary_json)

    if args.out:
        Path(args.out).write_text(summary_json + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
