# Runbook

This runbook captures the local workflow for the current implementation.

## Prereqs
- uv (Python runner)
- Docker (for local Postgres + pgvector)
- `.env` with `OPENAI_API_KEY` (and optional settings)

## Local Services
- Postgres + pgvector (local)
- Phoenix (local)

## Docker compose
```bash
docker compose up --build
```

Makefile shortcut:
```bash
make up
```

## Data Prep
1) Select 2,000 people records from `data/people_*.ndjson`.
2) Chunk into:
   - Abstract + infobox fields (short, precise facts)
   - Narrative sections (biography, awards, etc.)
3) Insert into `documents` and `chunks` tables with embeddings.

### Build dataset (uv)
```bash
uv run python scripts/build_dataset.py \
  --input data/people_0.ndjson data/people_1.ndjson data/people_2.ndjson data/people_3.ndjson \
  --sample-size 2000 --eval-questions 100 \
  --out-people data/processed/people_2k.ndjson \
  --out-chunks data/processed/chunks_2k.jsonl \
  --out-eval eval/questions_100.jsonl
```

## Setup (uv)
```bash
cp .env.example .env
# add OPENAI_API_KEY to .env

uv sync
```

## Database init
```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_eval
uv run bash scripts/init_db.sh
```
Note: `sql/schema.sql` sets the embedding dimension to 1536 for `text-embedding-3-small`.  
If you switch embedding models, update the dimension and rebuild the table.

If you are using Docker and don’t have `psql` locally:
```bash
cat sql/schema.sql | docker compose exec -T db psql -U postgres -d rag_eval
```

## Ingest data
```bash
export OPENAI_API_KEY=your_key_here
uv run python scripts/ingest_chunks.py \
  --people data/processed/people_2k.ndjson \
  --chunks data/processed/chunks_2k.jsonl
```

## App (FastAPI)
1) Start the API server.
2) Call `/query` with a test question.
3) Verify response includes:
   - Answer
   - Citations
   - Latency breakdown

```bash
uv run uvicorn app.main:app --reload --port 8000
```

## Observability (Phoenix)
1) Start Phoenix.
2) Confirm traces show embed/retrieve/generate spans.
3) Inspect a slow or low‑quality trace.

## Evaluation (Promptfoo + RAGAS)
1) Generate 100 questions (60% infobox, 40% narrative).
2) Run Promptfoo eval.
3) Store results in Postgres.
4) Verify RAGAS metrics are captured.

```bash
uv run python scripts/build_promptfoo_tests.py \
  --questions eval/questions_100.jsonl \
  --out eval/promptfoo_tests.json
```

Promptfoo execution (from repo root):
```bash
promptfoo eval -c eval/promptfoo.yaml
```

```bash
uv run python eval/run_eval.py --questions eval/questions_100.jsonl --out eval/summary.json
```

## Quality Gate
1) Define thresholds for groundedness, recall@k, p95 latency, cost/query.
2) Run the eval suite.
3) Verify the run fails if metrics drop below targets.

```bash
uv run python eval/quality_gate.py --summary eval/summary.json
```

## Demo Flow
1) Show baseline metrics.
2) Introduce a regression (prompt or retriever change).
3) Show quality gate failure + Phoenix trace diagnosis.
