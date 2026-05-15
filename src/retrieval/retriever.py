"""
Document retriever.

Performs similarity search against the vector store and returns
ranked candidate documents for downstream generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.embeddings.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Semantic retriever backed by a VectorStore.

    Parameters
    ----------
    vectorstore : VectorStore
        The initialized vector store to search against.
    top_k : int
        Default number of results to retrieve.
    """

    def __init__(self, vectorstore: VectorStore, top_k: int = 5):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a query.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int, optional
            Override the default number of results.
        where : dict, optional
            Metadata filter passed to ChromaDB.

        Returns
        -------
        list[dict]
            Ranked results with keys: id, text, metadata, distance, rank.
        """
        k = top_k or self.top_k

        results = self.vectorstore.query(query=query, top_k=k, where=where)

        # Add rank
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        logger.info(
            "Retrieved %d chunks for query: '%s...'",
            len(results),
            query[:60],
        )
        return results

    def retrieve_texts(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Convenience method: return only the text content of retrieved chunks.

        Parameters
        ----------
        query : str
            The user's question.
        top_k : int, optional
            Number of results.

        Returns
        -------
        list[str]
            Text content of matching chunks.
        """
        results = self.retrieve(query, top_k=top_k)
        return [r["text"] for r in results]
