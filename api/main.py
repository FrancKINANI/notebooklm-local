"""
FastAPI application — LocalNotebook RAG API.

Endpoints:
  POST /query     — Ask a question against indexed documents
  POST /ingest    — Ingest and index new documents
  GET  /health    — Health check (Ollama + VectorStore status)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SessionEvalRequest,
    SourceInfo,
)
from src.generation.llm import OllamaLLM
from src.pipeline.rag import RAGPipeline, build_index, ingest_documents
from src.evaluation.ragas_eval import evaluate_pipeline, log_session_to_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LocalNotebook RAG API",
    description="Privacy-first RAG system with local LLMs (Ollama)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized pipelines (one per model)
_pipelines: dict[str, RAGPipeline] = {}


def _get_pipeline(model_key: str) -> RAGPipeline:
    """Get or create a RAG pipeline for the given model."""
    if model_key not in _pipelines:
        try:
            _pipelines[model_key] = RAGPipeline(model_key=model_key)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize pipeline for '{model_key}': {e}",
            )
    return _pipelines[model_key]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question against indexed documents."""
    pipeline = _get_pipeline(request.model)

    try:
        result = pipeline.ask(query=request.question, top_k=request.top_k)
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        model=result["model"],
        model_key=result["model_key"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        num_sources=result["num_sources"],
        latency_ms=result["latency_ms"],
        tokens_per_second=result["tokens_per_second"],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest and index documents from a given path."""
    source = Path(request.source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {source}")

    try:
        ingest_documents(
            source_dir=str(source),
            output_dir="data/processed",
        )

        if not request.skip_index:
            build_index(chunks_path="data/processed/chunks.json")

            # Trigger background evaluation
            background_tasks.add_task(
                evaluate_pipeline,
                model_key="llama3:8b",  # Default to llama3:8b for background eval
                log_to_mlflow=True,
            )

        return IngestResponse(
            status="success",
            message=f"Ingested documents from {source}. Evaluation started in background.",
        )
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session_eval")
async def session_eval(request: SessionEvalRequest, background_tasks: BackgroundTasks):
    """Process session feedback and trigger RAGAS evaluation."""
    try:

        def _run_and_log():
            # 1. Run RAGAS
            metrics = evaluate_pipeline(
                model_key=request.model_key,
                log_to_mlflow=False,  # We'll log a custom session run instead
            )
            # 2. Log combined session to MLflow
            log_session_to_mlflow(
                model_key=request.model_key,
                ragas_metrics=metrics,
                feedbacks=[f.dict() for f in request.feedbacks],
            )

        background_tasks.add_task(_run_and_log)
        return {
            "status": "success",
            "message": "Session evaluation triggered in background.",
        }
    except Exception as e:
        logger.error("Session eval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    llm = OllamaLLM()
    ollama_ok = llm.is_available()

    vs_count = 0
    try:
        pipeline = _get_pipeline("llama3:8b")
        vs_count = pipeline.vectorstore.count
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_available=ollama_ok,
        vectorstore_count=vs_count,
    )
