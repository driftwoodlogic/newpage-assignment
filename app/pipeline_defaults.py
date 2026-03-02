from __future__ import annotations

# Centralized project defaults for runtime and dataset/eval generation.

# Runtime / models
DEFAULT_LLM_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536

# Retrieval runtime defaults
DEFAULT_TOP_K = 12
DEFAULT_RERANK_K = 6
DEFAULT_RERANK_ENABLED = True

# Dataset build defaults
DEFAULT_PDF_INPUT_DIR = "data/pdf"
DEFAULT_EVAL_QUESTIONS = 100
DEFAULT_DATASET_SEED = 42
DEFAULT_CHUNK_TARGET_CHARS = 900
DEFAULT_CHUNK_MAX_CHARS = 1400
DEFAULT_CHUNK_OVERLAP_PARAGRAPHS = 1

# PDF parsing / chunking heuristics (tunable)
HEADING_MAX_CHARS = 140
HEADING_MAX_WORDS = 14
PARAGRAPH_MIN_CHARS = 40
TABLELIKE_LINE_BLOCK_MAX_CHARS = 220

# Synthetic eval generation heuristics (tunable)
EVAL_SENTENCE_MIN_CHARS = 60
EVAL_SENTENCE_MAX_CHARS = 340
EVAL_TOP_SENTENCES_PER_CHUNK = 2
EVAL_REPEAT_CHUNK_SKIP_PROB = 0.7

# Ingestion defaults
DEFAULT_INGEST_BATCH_SIZE = 64
