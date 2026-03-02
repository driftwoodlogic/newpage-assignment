#!/usr/bin/env python3
"""Fail if eval summary metrics fall below thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary JSON from run_eval.py")
    parser.add_argument("--min-grounded", type=float, default=0.75)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--max-p95-latency", type=float, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    failures = []
    grounded = summary.get("faithfulness") or summary.get("groundedness")
    if grounded is not None and grounded < args.min_grounded:
        failures.append(f"groundedness {grounded} < {args.min_grounded}")

    if summary.get("recall_avg", 1.0) < args.min_recall:
        failures.append(f"recall_avg {summary.get('recall_avg')} < {args.min_recall}")

    if summary.get("precision_avg", 1.0) < args.min_precision:
        failures.append(f"precision_avg {summary.get('precision_avg')} < {args.min_precision}")

    if failures:
        print("Quality gate failed:\n- " + "\n- ".join(failures))
        return 2

    print("Quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
