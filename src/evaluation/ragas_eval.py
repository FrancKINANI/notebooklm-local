"""
RAGAS evaluation runner.

Evaluates RAG pipeline quality using RAGAS metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Results are logged to MLflow and saved as JSON metrics.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import mlflow

logger = logging.getLogger(__name__)

METRICS_DIR = Path("metrics")


def load_eval_dataset(eval_path: str = "data/eval") -> List[Dict[str, str]]:
    """
    Load the evaluation dataset (question/answer pairs).

    Expected format: JSON file with list of objects:
    [{"question": "...", "ground_truth": "...", "contexts": ["..."]}]

    Parameters
    ----------
    eval_path : str
        Path to the evaluation directory or JSON file.

    Returns
    -------
    list[dict]
        Evaluation samples.
    """
    path = Path(eval_path)

    if path.is_file() and path.suffix == ".json":
        json_file = path
    elif path.is_dir():
        candidates = list(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No JSON eval files in {path}")
        json_file = candidates[0]
    else:
        raise FileNotFoundError(f"Eval path not found: {path}")

    with open(json_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info("Loaded %d eval samples from %s", len(dataset), json_file)
    return dataset


def evaluate_pipeline(
    model_key: str = "llama3",
    eval_path: str = "data/eval",
    output_path: str = "metrics/ragas_results.json",
    log_to_mlflow: bool = True,
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on the RAG pipeline.

    Parameters
    ----------
    model_key : str
        Model to evaluate ("lfm2.5" or "llama3").
    eval_path : str
        Path to evaluation dataset.
    output_path : str
        Path to save metrics JSON.
    log_to_mlflow : bool
        Whether to log results to MLflow.

    Returns
    -------
    dict
        RAGAS metric scores.
    """
    from src.pipeline.rag import RAGPipeline, _load_config

    config = _load_config()
    pipeline = RAGPipeline(model_key=model_key)
    eval_data = load_eval_dataset(eval_path)

    # Collect RAG outputs for RAGAS
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    latencies = []

    for sample in eval_data:
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")

        result = pipeline.ask(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(
            [s.get("text", "") for s in result.get("sources", [])]
            if isinstance(result.get("sources", [{}])[0], dict)
            and "text" in result.get("sources", [{}])[0]
            else [question]
        )
        ground_truths.append(ground_truth)
        latencies.append(result.get("latency_ms", 0))

    # Compute RAGAS metrics
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        eval_dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths,
            }
        )

        ragas_result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        metrics = dict(ragas_result)
    except ImportError:
        logger.warning("RAGAS not available, computing basic metrics only")
        metrics = {}
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        metrics = {}

    # Add latency metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    metrics["avg_latency_ms"] = round(avg_latency, 2)
    metrics["model"] = model_key
    metrics["num_samples"] = len(eval_data)

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = Path(output_path)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", out_file)

    # Log to MLflow
    if log_to_mlflow:
        _log_to_mlflow(metrics, config, model_key)

    return metrics


def log_session_to_mlflow(
    model_key: str,
    ragas_metrics: Dict[str, float],
    feedbacks: List[Dict[str, Any]],
) -> None:
    """
    Log combined RAGAS metrics and user feedback to MLflow.
    """
    try:
        from src.pipeline.rag import _load_config

        config = _load_config()

        mlflow.set_experiment("localnotebook-rag-sessions")

        with mlflow.start_run(run_name=f"session-{model_key}-{int(time.time())}"):
            # 1. Log RAGAS metrics
            for key, value in ragas_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"ragas_{key}", value)

            # 2. Log User Feedback
            if feedbacks:
                pos = sum(1 for f in feedbacks if f.get("is_positive"))
                total = len(feedbacks)
                satisfaction = pos / total if total > 0 else 0

                mlflow.log_metric("user_satisfaction", satisfaction)
                mlflow.log_metric("user_feedback_count", total)

                # Log feedback details as artifact
                with open("metrics/last_session_feedback.json", "w") as f:
                    json.dump(feedbacks, f, indent=2)
                mlflow.log_artifact("metrics/last_session_feedback.json")

            # 3. Log Params
            mlflow.log_param("model_name", model_key)
            mlflow.log_param("chunk_size", config["ingestion"]["chunk_size"])

        logger.info("Session results logged to MLflow")
    except Exception as e:
        logger.warning("Failed to log session to MLflow: %s", e)


def _log_to_mlflow(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    model_key: str,
) -> None:
    """Log evaluation results to MLflow."""
    try:
        mlflow.set_experiment("localnotebook-rag")

        with mlflow.start_run(run_name=f"eval-{model_key}-{int(time.time())}"):
            # Log parameters
            mlflow.log_param("model_name", model_key)
            mlflow.log_param("embedding_model", config["embeddings"]["model"])
            mlflow.log_param("chunk_size", config["ingestion"]["chunk_size"])
            mlflow.log_param("chunk_overlap", config["ingestion"]["chunk_overlap"])
            mlflow.log_param("top_k", config["retrieval"]["top_k"])
            mlflow.log_param("reranking", config["retrieval"]["reranking"])

            # Log metrics (only numeric values)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

            # Log metrics file as artifact
            metrics_file = METRICS_DIR / "ragas_results.json"
            if metrics_file.exists():
                mlflow.log_artifact(str(metrics_file))

        logger.info("Results logged to MLflow experiment 'localnotebook-rag'")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
