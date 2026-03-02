from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.pipeline_defaults import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RERANK_ENABLED,
    DEFAULT_RERANK_K,
    DEFAULT_TOP_K,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_eval"

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    llm_model: str = DEFAULT_LLM_MODEL

    # Retrieval
    top_k: int = DEFAULT_TOP_K
    rerank_k: int = DEFAULT_RERANK_K
    rerank_enabled: bool = DEFAULT_RERANK_ENABLED

    # Cost estimation (USD per 1M tokens)
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0

    # Phoenix / OpenTelemetry
    phoenix_otlp_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project: str = "rag-eval-observability"
    otel_service_name: str = "rag-eval-api"


settings = Settings()
