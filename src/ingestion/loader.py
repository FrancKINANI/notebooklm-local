"""
Multi-format document loader.

Supports PDF, TXT, Markdown, and HTML files.
Uses LangChain document loaders for consistent Document output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Supported extensions → loader factory ────────────────────────────────────

_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
}


def _load_pdf(path: Path) -> List[Document]:
    """Load a PDF file using PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def _load_text(path: Path) -> List[Document]:
    """Load a plain-text file."""
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8").load()


def _load_markdown(path: Path) -> List[Document]:
    """Load a Markdown file (treated as text with metadata)."""
    from langchain_community.document_loaders import TextLoader

    docs = TextLoader(str(path), encoding="utf-8").load()
    for doc in docs:
        doc.metadata["format"] = "markdown"
    return docs


def _load_html(path: Path) -> List[Document]:
    """Load an HTML file using BSHTMLLoader."""
    from langchain_community.document_loaders import BSHTMLLoader

    return BSHTMLLoader(str(path), open_encoding="utf-8").load()


_LOADER_FN = {
    "pdf": _load_pdf,
    "text": _load_text,
    "markdown": _load_markdown,
    "html": _load_html,
}


# ── Public API ───────────────────────────────────────────────────────────────


def load_document(path: str | Path) -> List[Document]:
    """
    Load a single document and return a list of LangChain Documents.

    Parameters
    ----------
    path : str | Path
        Path to the source file.

    Returns
    -------
    list[Document]
        Parsed documents with text content and metadata.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    ext = path.suffix.lower()
    file_type = _EXTENSION_MAP.get(ext)
    if file_type is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {list(_EXTENSION_MAP.keys())}"
        )

    logger.info("Loading %s file: %s", file_type, path.name)
    docs = _LOADER_FN[file_type](path)

    # Enrich metadata
    for doc in docs:
        doc.metadata.update(
            {
                "source": str(path),
                "filename": path.name,
                "file_type": file_type,
            }
        )

    logger.info("Loaded %d page(s) from %s", len(docs), path.name)
    return docs


def load_directory(directory: str | Path, glob: str = "**/*") -> List[Document]:
    """
    Recursively load all supported documents from a directory.

    Parameters
    ----------
    directory : str | Path
        Root directory to scan.
    glob : str
        Glob pattern (default: all files recursively).

    Returns
    -------
    list[Document]
        All documents found and loaded.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_docs: List[Document] = []
    supported = set(_EXTENSION_MAP.keys())

    for file_path in sorted(directory.glob(glob)):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            try:
                docs = load_document(file_path)
                all_docs.extend(docs)
            except Exception as exc:
                logger.warning("Skipping %s: %s", file_path.name, exc)

    logger.info(
        "Loaded %d document(s) from %s (%d pages total)",
        len(set(d.metadata.get("filename") for d in all_docs)),
        directory,
        len(all_docs),
    )
    return all_docs
