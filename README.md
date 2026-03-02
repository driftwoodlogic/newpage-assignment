# rag-eval-observability

Entry points:
- `docs/runbook.md` (end-to-end steps)
- `docs/architecture.md` (design + rationale)
- `docs/eval.md` (eval suite files)
- `docs/processed_data.md` (dataset outputs)

## Dataset build
```bash
uv run python scripts/build_dataset.py \
  --input data/people_0.ndjson data/people_1.ndjson data/people_2.ndjson data/people_3.ndjson \
  --sample-size 2000 --eval-questions 100 \
  --out-people data/processed/people_2k.ndjson \
  --out-chunks data/processed/chunks_2k.jsonl \
  --out-eval eval/questions_100.jsonl
```

## Run Dockers:
````bash
make up
````

## Quick start (local Postgres)
```bash
uv sync
set -a; source .env; set +a

# If `psql` is not installed locally, use the DB container instead:
cat sql/schema.sql | docker compose exec -T db psql -U postgres -d rag_eval

uv run python scripts/ingest_chunks.py \
  --people data/processed/people_2k.ndjson \
  --chunks data/processed/chunks_2k.jsonl
```

## Eval run
```bash
uv run python eval/run_eval.py --questions eval/questions_100.jsonl --out eval/summary.json
uv run python eval/quality_gate.py --summary eval/summary.json
```

## Example
Raw fields:
````json
{
  "identifier": 34903697,
  "name": "Carlos Francisco",
  "url": "https://en.wikipedia.org/wiki/Carlos_Francisco",
  "abstract": "Carlos Domingo Francisco Serrano is a Cuban international footballer who plays for Santiago de Cuba, as a left back.",
  "infobox": [
    {
      "has_parts": [
        {
          "has_parts": [
            { "type": "field", "name": "Full name", "value": "Carlos Domingo Francisco Serrano" }
          ]
        }
      ]
    }
  ],
  "article_sections": [
    {
      "name": "Club career",
      "has_parts": [
        { "type": "paragraph", "value": "Francisco plays his club football for hometown side Santiago de Cuba." }
      ]
    }
  ]
}
````
Chunks:
````json
{
  "chunk_id": "34903697:abstract",
  "source_id": "34903697",
  "name": "Carlos Francisco",
  "url": "https://en.wikipedia.org/wiki/Carlos_Francisco",
  "section": "Abstract",
  "type": "abstract",
  "text": "Carlos Domingo Francisco Serrano is a Cuban international footballer who plays for Santiago de Cuba, as a left back."
}
{
  "chunk_id": "34903697:infobox:0",
  "source_id": "34903697",
  "name": "Carlos Francisco",
  "url": "https://en.wikipedia.org/wiki/Carlos_Francisco",
  "section": "Infobox:Full name",
  "type": "infobox",
  "field_name": "Full name",
  "fact_value": "Carlos Domingo Francisco Serrano",
  "text": "Full name: Carlos Domingo Francisco Serrano"
}
{
  "chunk_id": "34903697:section:0",
  "source_id": "34903697",
  "name": "Carlos Francisco",
  "url": "https://en.wikipedia.org/wiki/Carlos_Francisco",
  "section": "Club career",
  "type": "section",
  "text": "Francisco plays his club football for hometown side Santiago de Cuba."
}
````
