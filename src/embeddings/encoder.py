"""
Sentence-transformer embedding encoder.

Wraps a HuggingFace sentence-transformers model to produce embeddings
compatible with ChromaDB's custom embedding function interface.
"""

from __future__ import annotations

import logging
from typing import List

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)


class SentenceTransformerEncoder(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using sentence-transformers.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g. "intfloat/multilingual-e5-small").
    batch_size : int
        Number of texts to encode per batch.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "Model loaded — dimension: %d",
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        """
        Encode a batch of documents into embeddings.

        For E5 models, we prepend "passage: " or "query: " prefix
        to improve retrieval quality.
        """
        # E5 models expect prefixed inputs
        is_e5 = "e5" in self.model_name.lower()
        texts = [f"passage: {t}" for t in input] if is_e5 else list(input)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """
        Encode a single query string.

        Uses "query: " prefix for E5 models to match their training format.
        """
        is_e5 = "e5" in self.model_name.lower()
        text = f"query: {query}" if is_e5 else query

        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()
