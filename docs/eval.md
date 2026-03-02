# Eval Dataset

File:
- `questions_100.jsonl`: 100 synthetic questions with gold passages.
- `promptfoo.yaml`: Promptfoo config calling the local API.
- `promptfoo_tests.json`: Promptfoo test cases generated from questions.
- `run_eval.py`: API-driven eval runner (retrieval + optional RAGAS).
- `quality_gate.py`: Fails if metrics drop below thresholds.

Schema:
- `id`: question id
- `question`: natural language question
- `gold_answer`: target answer text
- `gold_chunk_ids`: list of chunk ids expected to support the answer
- `gold_context`: reference passage (for groundedness)
- `source_id`: original person identifier
- `source_url`: Wikipedia URL
- `type`: `infobox` | `narrative`
