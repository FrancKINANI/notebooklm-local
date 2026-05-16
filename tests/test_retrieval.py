"""
Tests for the retrieval module (retriever + reranker).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.retriever import Retriever


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vectorstore():
    """Create a mock VectorStore."""
    vs = MagicMock()
    vs.query.return_value = [
        {
            "id": "doc1::chunk_0",
            "text": "Python is a programming language.",
            "metadata": {"filename": "doc1.txt", "chunk_index": 0},
            "distance": 0.15,
        },
        {
            "id": "doc1::chunk_1",
            "text": "Machine learning uses algorithms.",
            "metadata": {"filename": "doc1.txt", "chunk_index": 1},
            "distance": 0.25,
        },
        {
            "id": "doc2::chunk_0",
            "text": "Deep learning is a subset of ML.",
            "metadata": {"filename": "doc2.txt", "chunk_index": 0},
            "distance": 0.35,
        },
    ]
    return vs


# ── Retriever tests ──────────────────────────────────────────────────────────


class TestRetriever:
    def test_retrieve_returns_ranked(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore, top_k=5)
        results = retriever.retrieve("What is Python?")

        assert len(results) == 3
        assert results[0]["rank"] == 1
        assert results[2]["rank"] == 3
        mock_vectorstore.query.assert_called_once()

    def test_retrieve_custom_top_k(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore, top_k=5)
        retriever.retrieve("test", top_k=2)
        mock_vectorstore.query.assert_called_once_with(
            query="test", top_k=2, where=None
        )

    def test_retrieve_texts(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore, top_k=5)
        texts = retriever.retrieve_texts("test")

        assert len(texts) == 3
        assert "Python is a programming language." in texts[0]

    def test_retrieve_with_filter(self, mock_vectorstore):
        retriever = Retriever(mock_vectorstore, top_k=5)
        retriever.retrieve("test", where={"filename": "doc1.txt"})
        mock_vectorstore.query.assert_called_once_with(
            query="test", top_k=5, where={"filename": "doc1.txt"}
        )
