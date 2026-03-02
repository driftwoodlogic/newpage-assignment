#!/usr/bin/env python3
"""Convert questions_100.jsonl to Promptfoo test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.questions))
    tests = []
    for r in rows:
        tests.append(
            {
                "vars": {
                    "question": r.get("question"),
                    "gold_chunk_ids": r.get("gold_chunk_ids"),
                    "gold_context": r.get("gold_context"),
                }
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tests, indent=2), encoding="utf-8")
    print(f"Wrote {len(tests)} tests -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
