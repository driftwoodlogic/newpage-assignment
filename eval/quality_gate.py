#!/usr/bin/env python3
"""Fail if eval summary metrics fall below thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary JSON from run_eval.py")
    parser.add_argument("--min-recall", type=float, default=0.70,
                        help="Min fraction of questions where gold chunk was retrieved (default: 0.70)")
    parser.add_argument("--min-recall-reranked", type=float, default=0.60,
                        help="Min fraction of questions where gold chunk survived reranking into LLM context (default: 0.60)")
    parser.add_argument("--max-p95-latency", type=float, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    failures = []
    if summary.get("recall_avg", 1.0) < args.min_recall:
        failures.append(f"recall_avg {summary.get('recall_avg')} < {args.min_recall}")

    if summary.get("recall_reranked_avg", 1.0) < args.min_recall_reranked:
        failures.append(f"recall_reranked_avg {summary.get('recall_reranked_avg')} < {args.min_recall_reranked}")

    p95 = summary.get("latency_p95_ms")
    if args.max_p95_latency is not None and isinstance(p95, (int, float)) and p95 > args.max_p95_latency:
        failures.append(f"latency_p95_ms {p95} > {args.max_p95_latency}")

    cost_avg = summary.get("cost_avg_usd")
    if args.max_cost is not None and isinstance(cost_avg, (int, float)) and cost_avg > args.max_cost:
        failures.append(f"cost_avg_usd {cost_avg} > {args.max_cost}")

    if failures:
        print("Quality gate failed:\n- " + "\n- ".join(failures))
        return 2

    print("Quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
