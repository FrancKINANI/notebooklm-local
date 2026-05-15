"""
ChromaDB vector store interface.

Provides a unified interface for indexing and querying document embeddings
with persistent storage and metadata filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_core.documents import Document

from src.embeddings.encoder import SentenceTransformerEncoder

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent ChromaDB vector store.

    Parameters
    ----------
    persist_directory : str
        Path to the ChromaDB persistence directory.
    collection_name : str
        Name of the ChromaDB collection.
    embedding_model : str
        HuggingFace model ID for embeddings.
    batch_size : int
        Encoding batch size.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "localnotebook",
        embedding_model: str = "intfloat/multilingual-e5-small",
        batch_size: int = 32,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize encoder
        self.encoder = SentenceTransformerEncoder(
            model_name=embedding_model,
            batch_size=batch_size,
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.encoder,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "VectorStore initialized: collection='%s', docs=%d",
            collection_name,
            self.collection.count(),
        )

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Index a list of LangChain documents into ChromaDB.

        Parameters
        ----------
        documents : list[Document]
            Chunked documents to index.
        batch_size : int
            Number of documents to upsert per batch.

        Returns
        -------
        int
            Number of documents indexed.
        """
        if not documents:
            logger.warning("No documents to index.")
            return 0

        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            # Create a deterministic ID from source + chunk index
            source = doc.metadata.get("filename", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", i)
            doc_id = f"{source}::chunk_{chunk_idx}"

            ids.append(doc_id)
            texts.append(doc.page_content)

            # ChromaDB metadata must be str/int/float/bool
            clean_meta = {
                k: v
                for k, v in doc.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            metadatas.append(clean_meta)

        # Batch upsert
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            logger.debug("Upserted batch %d–%d / %d", start, end, total)

        logger.info("Indexed %d chunks into '%s'", total, self.collection_name)
        return total

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Number of results to return.
        where : dict, optional
            ChromaDB metadata filter.

        Returns
        -------
        list[dict]
            Results with keys: id, text, metadata, distance.
        """
        query_embedding = self.encoder.encode_query(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        logger.debug("Query returned %d results (top_k=%d)", len(output), top_k)
        return output

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info("Deleted collection '%s'", self.collection_name)

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
