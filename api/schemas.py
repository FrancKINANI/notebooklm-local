"""
Pydantic request/response schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# ── Request schemas ──────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """RAG query request."""

    question: str = Field(..., min_length=1, description="The user's question")
    model: str = Field(
        default="llama3.1",
        description="Model key: 'llama3.1' or 'lfm2.5'",
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of chunks to retrieve"
    )


class IngestRequest(BaseModel):
    """Document ingestion request."""

    source_path: str = Field(..., description="Path to file or directory to ingest")
    skip_index: bool = Field(default=False, description="Skip vector indexing step")


class FeedbackRequest(BaseModel):
    """User feedback for a specific response."""

    question: str
    answer: str
    is_positive: bool
    model_key: str


class SessionEvalRequest(BaseModel):
    """Request to finalize session and run evaluation."""

    model_key: str = "llama3.1"
    feedbacks: List[FeedbackRequest] = []


# ── Response schemas ─────────────────────────────────────────────────────────


class SourceInfo(BaseModel):
    """Information about a source chunk used in generation."""

    filename: str
    chunk_index: int = -1
    distance: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    model: str
    model_key: str
    sources: List[SourceInfo] = []
    num_sources: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0


class IngestResponse(BaseModel):
    """Ingestion response."""

    status: str = "success"
    message: str = ""
    chunks_count: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    ollama_available: bool = False
    vectorstore_count: int = 0
