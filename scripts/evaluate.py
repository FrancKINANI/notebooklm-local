#!/usr/bin/env python3
"""
CLI script: Run RAGAS evaluation on the RAG pipeline.

Usage:
    python scripts/evaluate.py --model llama3
    python scripts/evaluate.py --model lfm2.5 --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ragas_eval import evaluate_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with RAGAS metrics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        choices=["llama3", "lfm2.5"],
        help="Model to evaluate (default: llama3)",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default="data/eval",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/ragas_results.json",
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    args = parser.parse_args()

    logger.info("=== RAGAS Evaluation ===")
    logger.info("Model: %s", args.model)

    metrics = evaluate_pipeline(
        model_key=args.model,
        eval_path=args.eval_path,
        output_path=args.output,
        log_to_mlflow=not args.no_mlflow,
    )

    logger.info("=== Results ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
