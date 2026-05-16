"""
Tests for the RAG pipeline integration.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestRAGPipeline:
    """Integration tests for RAGPipeline using mocks."""

    @patch("src.pipeline.rag._load_models_config")
    @patch("src.pipeline.rag._load_config")
    @patch("src.pipeline.rag.VectorStore")
    @patch("src.pipeline.rag.OllamaLLM")
    def test_ask_returns_answer(
        self, mock_llm_cls, mock_vs_cls, mock_config, mock_models
    ):
        """Test that ask() returns a properly structured response."""
        mock_config.return_value = {
            "vectorstore": {
                "persist_directory": "./test_chroma",
                "collection_name": "test",
            },
            "embeddings": {"model": "test-model", "batch_size": 8},
            "retrieval": {"top_k": 3, "reranking": False},
            "generation": {
                "temperature": 0.1,
                "max_tokens": 100,
                "top_k": 50,
                "repeat_penalty": 1.0,
            },
        }
        mock_models.return_value = {
            "models": {
                "llama3.1": {
                    "name": "llama3.1:8b",
                    "provider": "ollama",
                }
            }
        }

        # Mock vectorstore query
        mock_vs = mock_vs_cls.return_value
        mock_vs.query.return_value = [
            {
                "id": "test::chunk_0",
                "text": "Test context content.",
                "metadata": {"filename": "test.txt", "chunk_index": 0},
                "distance": 0.1,
            }
        ]

        # Mock LLM generate
        mock_llm = mock_llm_cls.return_value
        mock_llm.generate.return_value = {
            "answer": "This is the answer.",
            "model": "llama3.1:8b",
            "latency_ms": 150.0,
            "tokens_per_second": 25.0,
            "eval_count": 10,
        }

        from src.pipeline.rag import RAGPipeline

        pipeline = RAGPipeline(model_key="llama3.1")
        result = pipeline.ask("What is this?")

        assert "answer" in result
        assert result["answer"] == "This is the answer."
        assert result["model_key"] == "llama3.1"
        assert "sources" in result

    @patch("src.pipeline.rag._load_models_config")
    @patch("src.pipeline.rag._load_config")
    def test_invalid_model_raises(self, mock_config, mock_models):
        """Test that an invalid model key raises ValueError."""
        mock_config.return_value = {
            "vectorstore": {"persist_directory": "./test", "collection_name": "t"},
            "embeddings": {"model": "m"},
            "retrieval": {"top_k": 3},
            "generation": {},
        }
        mock_models.return_value = {"models": {"llama3.1": {"name": "x"}}}

        from src.pipeline.rag import RAGPipeline

        with pytest.raises(ValueError, match="Unknown model"):
            RAGPipeline(model_key="nonexistent")
