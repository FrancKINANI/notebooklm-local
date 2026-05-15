#!/usr/bin/env python3
"""
CLI script: Ingest documents into the RAG pipeline.

Usage:
    python scripts/ingest.py --source data/raw/
    python scripts/ingest.py --source data/raw/document.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.rag import ingest_documents, build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest and index documents for LocalNotebook RAG"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw",
        help="Source file or directory (default: data/raw/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed chunks (default: data/processed/)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip the vector index building step",
    )
    args = parser.parse_args()

    logger.info("=== Ingestion Pipeline ===")

    # Step 1: Ingest and chunk
    logger.info("Step 1/2: Ingesting documents from %s", args.source)
    ingest_documents(source_dir=args.source, output_dir=args.output)

    # Step 2: Build vector index
    if not args.skip_index:
        chunks_path = str(Path(args.output) / "chunks.json")
        logger.info("Step 2/2: Building vector index from %s", chunks_path)
        build_index(chunks_path=chunks_path)
    else:
        logger.info("Step 2/2: Skipped (--skip-index)")

    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()
