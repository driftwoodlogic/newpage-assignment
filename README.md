# rag-eval-observability

## Repo Function
This repo builds and serves a local RAG system (using Postgres/pgvector and FastAPI) over UK aseptic cleanroom regulatory PDFs, and evaluates the pipeline outputs using:
 - Promptfoo for lightweight smoke/regression checking
 - A custom script (run_eval.py) to run the synthetic question set against the live API, producing a summary of:
   - Retrieval precision/recall metrics
   - Latency/cost metrics
 - Phoenix to visualise LLM traces

## Dataset used
2 cleanroom regulatory files:
 - EU GMP Annex 1 - core sterile manufacturing guidance
 - MHRA Guidance for Specials (unlicensed medicines) Manufacturers

## Run It  
Set your OpenAI API key in the .env
````bash
uv sync
make up # start postgres/pg vector (db and vector db) and phoenix (tracing)
set -a; source .env; set +a
cat sql/schema.sql | docker compose exec -T db psql -U postgres -d rag_eval # app;y db schema

uv run python scripts/build_dataset.py \
    --input-dir data/pdf \
    --out-docs data/processed/documents.jsonl \
    --out-chunks data/processed/chunks.jsonl \
    --out-eval eval/questions_100.jsonl

uv run python scripts/ingest_chunks.py \
    --documents data/processed/documents.jsonl \
    --chunks data/processed/chunks.jsonl \
    --batch-size 64

uv run uvicorn app.main:app --reload --port 8000

curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What are the requirements for HEPA filter integrity testing in Grade A environments?"}' | jq

uv run python scripts/build_promptfoo_tests.py \
    --questions eval/questions_100.jsonl \
    --out eval/promptfoo_tests.json

make promptfoo-eval
make promptfoo-view

uv run python eval/run_eval.py \
    --questions eval/questions_100.jsonl \
    --out eval/summary.json

uv run python eval/quality_gate.py --summary eval/summary.json
````
View traces @ http://localhost:6006

# Technical 

## Generating Synthetic Questions
 - Generate questions from top scored candidate sentences extracted from each chunk
 - Scored on the basis of regulatory relevance
 - Questions and gold answers generated using LLM using: document name, chunk, page metadata

## Query Expansion
Acronyms -> related regulatory terms when they are detected. Examples:
- `CCS` -> `contamination control strategy`
- `GMP` -> `good manufacturing practice`
- `HEPA` -> `high efficiency particulate air filter`

## Hybrid Rescoring
 - pgvector returns chunk candidates with a vector similarity score
 - creates combined score using vector score, word overlap, section/heading overlap
 - sort retrieved chunk candidates by combined score

Why:
- Pure vector search may miss exact term/heading matches

## Rerank
 - Retrieved chunks sent to LLM for ranking in terms of relevance to question

# Deployment
## Putting this into production
 - Cloudfront + S3 for source docs, JSONl outputs, web client
 - ECS Fargate for FastAPI container
 - RDS PostgreSQL with pg vector as db and vector db
 - ECR for container images
 - Cloudwatch for logs + metrics, can use AWS Distro for OpenTelemetry (used by Phoenix)
 - Secrets Manager for keys, creds
 - ALP for HTTPS API routing
 - Step functions or AWS Batch for ingestion jobs
 - CICD using quality gates (github actions)

Above is a good foundation for scalability, would want to add several features:
 - Caching e.g. elasticache to cache embeddings and frequent results
 - Auto scaling basedon CPU/memory. RDS replicas S pgvector queries are read-heavy - depends on number of users
 - Ensure its deployed in a VPC with private subnets where appropriate. VPC endpoints for S3/Secrets manager to avoid internet traversal
 - API protection(waf)
 - Observability - add in cloudwatch alarms

... although given regulatory area for this, customers may want to run this on prem so they can input their own data freely
Or use models hosted on bedrock is probably the easiest

## RAG/LLM approach and decisions
 - **Embedding**: Used a small embedding model given the small dataset size and low complexity - `text-embedding-3-small`; chunks stored in pgvector - a performant vector store which allows us to use a single PostGres instance 
 - **Chunking**: paragraph-aware chunking with overlap to ensure sections aren't missed. Section headings and page numbers are prepended to each chunk to improve the retrieval signal
 - **Query expansion**: domain acronyms (CCS, GMP, HEPA, etc.) expanded to full regulatory terms before embedding as the LLM with otherwise not pick these up as a signal
 - **Hybrid rescoring**: vector score combined with word term overlap and a boost if headings match - using pure-vector would often miss extraction of regulatory terminology
 - **LLM rerank**: top-K candidates re-scored by the LLM for relevance before generation, doing this lowers the amount of context we need to give to the LLM, improving accuracy
 - **Generation**: Used a very small model: `gpt-5-nano` for generation with temperature 0 to attempt to get repeatable results. In production this could use a larger model to improve results given a larger range of docs would be used. Otherwise, good results have been shown by this model so a local model using Ollama would make sense to be privacy preserving
 - **System Prompt**: Use a grounded system prompt telling the model to answer only from context, cite by `[n]`, say "don't know" if not found
 - **Synthetic eval set**: 100 questions generated from the range of chunks used as the ground-truth set for retrieval precision/recall and Promptfoo regression checks

## Key Technical Decisions
 - Overall aimed for observability and eval-driven development as key factors for this given the requirement for high accuracy when deployed. This also allows us to see where the pipeline is failing or underperforming and fix that specific problem before deployment. Observability is key here and would be even more key if adding in agentic steps
 - Single postgres instance for everything simplifies the infra and deployment significantly
 - The LLM client is configurable meaning we can swap it for other providers easily, or use AWS Bedrock or local models for privacy
 - Single centralised config ensures that all tunable hyperparameters are available
 - Two tier evaluation means that non-technical users can use promptfoo to understand and broadly understand and detect regressions, and the development team can use the detailed eval to drill into specific failure cases
 - Pydantic is used to be type safe everywhere

# Engineering standards
## Followed:
 - Observability
 - Config management
 - Type hints
 - Eval-driven development
 - Dependency management
 
## Skipped:
 - Tests
 - CICD
 - Linting
 - Logging

# AI development process
 - Heavy use of claude/codex for development, generating architectures, generating solutions to individual problems
 - Lots of HITL to correct the above and ensure it fits the use case
 - README done by hand

# Given more time
 - Actually build the web frontend rather than just the observability pipeline
 - I'd investigate agentic development for this use case - the biggest unknown
 - Putting this into production would obviously be looking at a lot more docs so tighten up evaluation based on our use case (what we're trying to achieve) and content of our docs
## Future Work
 - Agents with retrieval tools > Classic 1 shot RAG flow. Agents can:
   - Plan the task, retrieve repeatedly, reformulate queries, inspect sources, decide if evidence is sufficient, optionally use tools (search, DB, API)
 - Would spend a lot more time on eval question set related to problem we're trying to solve
 - Expand to a range of different documents to be useful to customers
 - Spin up and host web frontend customers can use
 - Given regulatory standpoint, customers may want to input their own data - move to deploying on customer site with local models 
   - Customer data never leaves customer site
   - Can deploy as an agent (think claude) on their data in the far flung future - purely for data inspection not modification
   - Ensure provenance of all data returned is verifiable
