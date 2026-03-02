from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.observability import init_tracer
from app.rag import answer_query

app = FastAPI(title="RAG Eval + Observability API")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int | None = None
    rerank_k: int | None = None


@app.on_event("startup")
def on_startup() -> None:
    init_tracer()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return answer_query(req.query, top_k=req.top_k, rerank_k=req.rerank_k)
