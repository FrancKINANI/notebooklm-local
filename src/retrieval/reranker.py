"""
Cross-encoder reranker.

Uses a cross-encoder model to re-score retrieved candidates,
improving precision by jointly encoding (query, passage) pairs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker using a HuggingFace cross-encoder model.

    Parameters
    ----------
    model_name : str
        Cross-encoder model ID from HuggingFace.
    top_k : int
        Number of results to keep after reranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 3,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents using the cross-encoder.

        Parameters
        ----------
        query : str
            The user's question.
        candidates : list[dict]
            Retrieved candidates (must have a "text" key).
        top_k : int, optional
            Override the default number of results to keep.

        Returns
        -------
        list[dict]
            Reranked candidates sorted by cross-encoder score (descending).
        """
        if not candidates:
            return []

        k = min(top_k or self.top_k, len(candidates))

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        reranked = reranked[:k]

        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank

        logger.info(
            "Reranked %d -> %d candidates for: '%s...'",
            len(candidates),
            len(reranked),
            query[:60],
        )
        return reranked
