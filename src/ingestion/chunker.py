"""
Document chunking strategies.

Supports recursive character splitting and semantic chunking.
All strategies produce LangChain Document objects with preserved metadata.
"""

from __future__ import annotations

import logging
from typing import List, Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    strategy: Literal["recursive", "semantic"] = "recursive",
) -> List[Document]:
    """
    Split documents into smaller chunks.

    Parameters
    ----------
    documents : list[Document]
        Source documents to chunk.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    strategy : str
        Chunking strategy: "recursive" (default) or "semantic".

    Returns
    -------
    list[Document]
        Chunked documents with original metadata preserved + chunk index.
    """
    if strategy == "semantic":
        return _semantic_chunk(documents, chunk_size, chunk_overlap)
    return _recursive_chunk(documents, chunk_size, chunk_overlap)


def _recursive_chunk(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Recursive character-based splitting.

    Uses a hierarchy of separators (paragraphs → sentences → words → chars)
    to split text at natural boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_strategy"] = "recursive"

    logger.info(
        "Recursive chunking: %d docs → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _semantic_chunk(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Semantic chunking using sentence-level grouping.

    Falls back to recursive chunking but uses sentence-aware separators
    and a tighter overlap for semantic coherence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size // 4),
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_strategy"] = "semantic"

    logger.info(
        "Semantic chunking: %d docs → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
