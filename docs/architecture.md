# RAG Eval + Observability Architecture (Phoenix + Promptfoo)

## Goals
- Demonstrate a complete RAG evaluation and observability pipeline.
- Make it easy to understand, run locally, and show tangible value.
- Cover retrieval quality, groundedness, latency breakdown, and cost per query.
- Use the English Wikipedia People dataset for a people-centric OSINT/PAI-style demo.

## Chosen Stack
- **API**: FastAPI
- **Vector store**: Postgres + pgvector (local)
- **Metrics store**: Postgres (local)
- **Tracing/observability**: Phoenix (OpenTelemetry)
- **Eval harness**: Promptfoo
- **RAG eval metrics**: RAGAS (groundedness, context precision/recall)

## High-Level Architecture

### Runtime plane (serving traffic)
1. **FastAPI** receives a query.
2. **Embedding** step produces a vector.
3. **Vector search** in Postgres (pgvector) retrieves top-k chunks.
4. Optional **rerank** step (if enabled).
5. **LLM generation** with retrieved context.
6. **Tracing** via OpenTelemetry to Phoenix for latency breakdowns.
7. **Metrics logging** to Postgres (latency, cost, model, prompt, version).

### Evaluation plane (quality + regressions)
1. **Promptfoo** runs smoke/regression checks against the API.
2. **API-driven eval runner** computes retrieval metrics and optional RAGAS metrics.
3. **Results** stored in Postgres.
4. **Quality gate** fails CI if metrics drop below thresholds.
5. **Phoenix** used to inspect traces and failure cases.

## Data Flow Details

### Runtime request flow
- Input: user query
- Output: final answer + citations
- Telemetry:
  - Embed latency
  - Retrieve latency
  - Rerank latency (if present)
  - Generate latency
  - Token usage + estimated cost
  - Version tags (prompt, model, retrieval config)

### Evaluation flow
- Input: synthetic question set + gold passages
- Output:
  - precision@k / recall@k
  - groundedness / citation coverage
  - latency and cost summary
- Storage: evaluation run metadata + per-test scores

## Demo Dataset Scope
- Source: English Wikipedia People dataset (local NDJSON).
- Index size: 2,000 people.
- Eval set size: 100 questions.
- Coverage:
  - Infobox fields (clean facts: birth date/place, occupation, affiliation).
  - Narrative sections (biography/awards/etc for harder retrieval).

## Storage Model (Postgres)

### Vector store
- `documents`: raw docs + metadata
- `chunks`: chunk text + metadata + embeddings (pgvector)

### Metrics store
- `query_logs`: query_id, answer, model, latencies, token usage, cost, retrieved ids
- `eval_runs`: run_id, model, embed_model, retrieval settings, notes
- `eval_scores`: run_id, question_id, metrics (jsonb)

## Observability
- Phoenix captures traces + spans for:
  - embed
  - retrieve
  - rerank
  - generate
- Phoenix UI used to debug slow or low-quality runs.
- Postgres stores request-level metrics in `query_logs`.

## Quality Gate
- Promptfoo runs on CI or pre-commit hook.
- Fail build if:
  - groundedness < threshold
  - recall@k < threshold
  - latency p95 > target
  - cost/query > target

## Why These Choices
- **Phoenix**: fast local setup, OpenTelemetry based, good for trace inspection.
- **Promptfoo**: simple CI integration and regression testing.
- **Postgres**: single local dependency for vectors + metrics.
- **RAGAS**: standard RAG metrics with a clear mapping to groundedness.

---

# Implementation Plan

## Phase 1: Skeleton + Conventions
Deliverables:
- `app/` FastAPI app with `/query` endpoint.
- `eval/` directory for datasets + Promptfoo config.
- `scripts/` directory for data prep and eval helpers.

## Phase 2: Postgres + pgvector Schema
Deliverables:
- DDL for `documents`, `chunks`, embeddings, and metadata.
- DDL for `query_logs`, `eval_runs`, `eval_scores`.
- Local Docker config (or instructions) to boot Postgres with pgvector.

## Phase 3: Dataset Prep (2k people)
Deliverables:
- Ingestion script to select 2,000 people from `data/people_*.ndjson`.
- Chunking plan:
  - Abstract + infobox fields as short chunks.
  - Narrative sections as longer chunks.
- Minimal metadata fields: `name`, `url`, `section`, `source_id`.

## Phase 4: RAG Pipeline
Deliverables:
- Embed + retrieve + generate flow in FastAPI.
- Optional rerank hook (feature flag).
- Response includes citations + latency breakdown.

## Phase 5: Observability with Phoenix
Deliverables:
- OpenTelemetry spans for embed, retrieve, rerank, generate.
- Phoenix exporter configuration (OTLP HTTP).
- Trace attributes: model, prompt version, top-k, chunk ids.

## Phase 6: Evaluation Suite (100 questions)
Deliverables:
- Synthetic question generator:
  - 60% infobox-derived questions.
  - 40% narrative-section questions.
- Gold passage mapping (chunk ids).
- Promptfoo config for API smoke evals.
- API-driven eval runner that stores results.
- RAGAS metrics (optional): groundedness + context precision/recall.

## Phase 7: Quality Gate
Deliverables:
- Thresholds in `eval/quality_gate.py`:
  - groundedness >= target
  - recall@k >= target
  - latency p95 <= target
  - cost/query <= target
- CI target to fail on regressions.

## Phase 8: Demo Story
Deliverables:
- Baseline report (metrics dashboard screenshot).
- Regression example (change prompt or retriever).
- Quality gate fail example + Phoenix trace for diagnosis.
