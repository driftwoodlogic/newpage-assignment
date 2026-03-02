CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  source_id TEXT PRIMARY KEY,
  name TEXT,
  url TEXT,
  raw JSONB
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  source_id TEXT REFERENCES documents(source_id),
  name TEXT,
  url TEXT,
  section TEXT,
  type TEXT,
  field_name TEXT,
  fact_value TEXT,
  text TEXT,
  embedding VECTOR(1536),
  metadata JSONB
);

CREATE INDEX IF NOT EXISTS chunks_source_id_idx ON chunks(source_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS query_logs (
  query_id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  query TEXT,
  answer TEXT,
  model TEXT,
  embed_model TEXT,
  top_k INTEGER,
  rerank_k INTEGER,
  latencies_ms JSONB,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cost_usd NUMERIC,
  retrieved_chunk_ids TEXT[],
  reranked_chunk_ids TEXT[]
);

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  git_sha TEXT,
  model TEXT,
  embed_model TEXT,
  top_k INTEGER,
  rerank_k INTEGER,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS eval_scores (
  run_id UUID REFERENCES eval_runs(run_id),
  question_id TEXT,
  metrics JSONB,
  PRIMARY KEY (run_id, question_id)
);
