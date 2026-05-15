"""
End-to-end RAG pipeline orchestration.

Wires together ingestion, embedding, retrieval, reranking, and generation
into a single callable pipeline with config-driven parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.embeddings.vectorstore import VectorStore
from src.generation.llm import OllamaLLM
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_directory
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/config.yaml")
MODELS_PATH = Path("configs/models.yaml")


def _load_config() -> Dict[str, Any]:
    """Load the global config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_models_config() -> Dict[str, Any]:
    """Load models.yaml."""
    with open(MODELS_PATH) as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """
    Full RAG pipeline: retrieve → (rerank) → generate.

    Parameters
    ----------
    model_key : str
        Key in models.yaml ("lfm2.5" or "llama3.1").
    config : dict, optional
        Override config (loaded from config.yaml if None).
    """

    def __init__(
        self,
        model_key: str = "llama3.1",
        config: Dict[str, Any] | None = None,
    ):
        self.config = config or _load_config()
        models_cfg = _load_models_config()

        # Resolve model
        if model_key not in models_cfg["models"]:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Available: {list(models_cfg['models'].keys())}"
            )
        model_info = models_cfg["models"][model_key]
        self.model_key = model_key

        # Initialize components
        vs_cfg = self.config["vectorstore"]
        emb_cfg = self.config["embeddings"]
        ret_cfg = self.config["retrieval"]
        gen_cfg = self.config["generation"]

        self.vectorstore = VectorStore(
            persist_directory=vs_cfg["persist_directory"],
            collection_name=vs_cfg["collection_name"],
            embedding_model=emb_cfg["model"],
            batch_size=emb_cfg.get("batch_size", 32),
        )

        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            top_k=ret_cfg["top_k"],
        )

        self.reranker = None
        if ret_cfg.get("reranking", False):
            self.reranker = CrossEncoderReranker(
                model_name=ret_cfg.get(
                    "reranker_model",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ),
                top_k=ret_cfg["top_k"],
            )

        self.llm = OllamaLLM(
            model=model_info["name"],
            temperature=gen_cfg.get("temperature", 0.1),
            max_tokens=gen_cfg.get("max_tokens", 512),
            top_k=gen_cfg.get("top_k", 50),
            repeat_penalty=gen_cfg.get("repeat_penalty", 1.05),
        )

        logger.info(
            "RAG pipeline initialized: model=%s, reranking=%s",
            model_info["name"],
            bool(self.reranker),
        )

    def ask(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        """
        Full RAG: retrieve → rerank → generate.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int, optional
            Override retrieval top-k.

        Returns
        -------
        dict
            Keys: answer, sources, model, latency_ms, tokens_per_second.
        """
        # 1. Retrieve
        candidates = self.retriever.retrieve(query, top_k=top_k)

        # 2. Rerank (optional)
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates)

        # 3. Build context
        context = "\n\n---\n\n".join(
            f"[Source: {c['metadata'].get('filename', 'unknown')}]\n{c['text']}"
            for c in candidates
        )

        # 4. Generate
        result = self.llm.generate(query, context)

        # 5. Attach sources
        result["sources"] = [
            {
                "filename": c["metadata"].get("filename", "unknown"),
                "chunk_index": c["metadata"].get("chunk_index", -1),
                "distance": c.get("distance"),
                "rerank_score": c.get("rerank_score"),
            }
            for c in candidates
        ]
        result["model_key"] = self.model_key
        result["num_sources"] = len(candidates)

        return result


# ── Standalone functions for DVC pipeline stages ──


def ingest_documents(
    source_dir: str = "data/raw",
    output_dir: str = "data/processed",
) -> None:
    """
    DVC stage: ingest and chunk documents.

    Reads all files from source_dir, chunks them, and saves
    the result as JSON to output_dir.
    """
    config = _load_config()
    ing_cfg = config["ingestion"]

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    docs = load_directory(source_path)
    if not docs:
        logger.warning("No documents found in %s", source_path)
        return

    chunks = chunk_documents(
        docs,
        chunk_size=ing_cfg["chunk_size"],
        chunk_overlap=ing_cfg["chunk_overlap"],
        strategy=ing_cfg.get("chunking_strategy", "recursive"),
    )

    # Save chunks as JSON
    output_file = output_path / "chunks.json"
    serialized = [
        {
            "page_content": c.page_content,
            "metadata": {
                k: v
                for k, v in c.metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
        }
        for c in chunks
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d chunks to %s", len(chunks), output_file)


def build_index(
    chunks_path: str = "data/processed/chunks.json",
) -> None:
    """
    DVC stage: build the vector index from processed chunks.

    Reads the chunked documents and indexes them into ChromaDB.
    """
    from langchain_core.documents import Document

    config = _load_config()
    vs_cfg = config["vectorstore"]
    emb_cfg = config["embeddings"]

    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        logger.error("Chunks file not found: %s", chunks_file)
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in raw_chunks
    ]

    store = VectorStore(
        persist_directory=vs_cfg["persist_directory"],
        collection_name=vs_cfg["collection_name"],
        embedding_model=emb_cfg["model"],
        batch_size=emb_cfg.get("batch_size", 32),
    )

    count = store.add_documents(documents)
    logger.info("Indexed %d chunks into vector store", count)
