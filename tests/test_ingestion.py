"""
Tests for the ingestion module (loader + chunker).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_document, load_directory


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """Create a sample .txt file."""
    f = tmp_path / "sample.txt"
    f.write_text(
        "This is paragraph one about artificial intelligence.\n\n"
        "This is paragraph two about machine learning and deep learning.\n\n"
        "This is paragraph three about natural language processing.",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """Create a sample .md file."""
    f = tmp_path / "readme.md"
    f.write_text(
        "# Title\n\nSome intro text.\n\n## Section 1\n\nContent for section one.\n\n"
        "## Section 2\n\nContent for section two.",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_dir(tmp_path: Path, sample_txt: Path, sample_md: Path) -> Path:
    """Directory with multiple files."""
    return tmp_path


# ── Loader tests ─────────────────────────────────────────────────────────────


class TestLoader:
    def test_load_txt(self, sample_txt: Path):
        docs = load_document(sample_txt)
        assert len(docs) >= 1
        assert "artificial intelligence" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "text"
        assert docs[0].metadata["filename"] == "sample.txt"

    def test_load_md(self, sample_md: Path):
        docs = load_document(sample_md)
        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "markdown"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/file.txt")

    def test_load_unsupported_raises(self, tmp_path: Path):
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)

    def test_load_directory(self, sample_dir: Path):
        docs = load_directory(sample_dir)
        assert len(docs) >= 2  # At least from .txt and .md
        filenames = {d.metadata["filename"] for d in docs}
        assert "sample.txt" in filenames
        assert "readme.md" in filenames


# ── Chunker tests ────────────────────────────────────────────────────────────


class TestChunker:
    def test_recursive_chunking(self, sample_txt: Path):
        docs = load_document(sample_txt)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 100 + 20  # Allow overlap margin
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "recursive"

    def test_semantic_chunking(self, sample_txt: Path):
        docs = load_document(sample_txt)
        chunks = chunk_documents(
            docs, chunk_size=100, chunk_overlap=20, strategy="semantic"
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata["chunk_strategy"] == "semantic"

    def test_metadata_preserved(self, sample_txt: Path):
        docs = load_document(sample_txt)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "filename" in chunk.metadata
            assert chunk.metadata["filename"] == "sample.txt"

    def test_empty_docs(self):
        chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
        assert chunks == []
