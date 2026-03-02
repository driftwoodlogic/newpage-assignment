from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_eval"

    # OpenAI
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o"

    # Retrieval
    top_k: int = 20
    rerank_k: int = 5
    rerank_enabled: bool = True

    # Cost estimation (USD per 1M tokens)
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0

    # Phoenix / OpenTelemetry
    phoenix_otlp_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project: str = "rag-eval-observability"
    otel_service_name: str = "rag-eval-api"


settings = Settings()
