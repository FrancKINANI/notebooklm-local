"""
Ollama LLM client abstraction.

Provides a unified interface for generating text using local Ollama models.
Supports both streaming and non-streaming generation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Generator, Optional

import ollama

logger = logging.getLogger(__name__)

# Default RAG system prompt
RAG_SYSTEM_PROMPT = """You are a helpful research assistant. Answer the user's question based ONLY on the provided context.

Rules:
1. Only use information from the context below.
2. If the context doesn't contain enough information, say so clearly.
3. Cite the source document when possible.
4. Be concise and precise.

Context:
{context}"""


class OllamaLLM:
    """
    Ollama LLM client for local generation.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. "llama3.1:8b").
    base_url : str
        Ollama API base URL.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    top_k : int
        Top-k sampling parameter.
    repeat_penalty : float
        Repetition penalty.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_k: int = 50,
        repeat_penalty: float = 1.05,
    ):
        self.model = model
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

        self.client = ollama.Client(host=self.base_url)

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer using retrieved context.

        Parameters
        ----------
        query : str
            The user's question.
        context : str
            Retrieved context chunks joined as text.
        system_prompt : str, optional
            Custom system prompt (uses RAG_SYSTEM_PROMPT by default).

        Returns
        -------
        dict
            Keys: answer, model, latency_ms, tokens_per_second.
        """
        prompt = (system_prompt or RAG_SYSTEM_PROMPT).format(context=context)

        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        answer = response["message"]["content"]
        eval_count = response.get("eval_count", 0)
        tps = (eval_count / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        logger.info(
            "Generated %d tokens in %.0fms (%.1f t/s) — model=%s",
            eval_count,
            elapsed_ms,
            tps,
            self.model,
        )

        return {
            "answer": answer,
            "model": self.model,
            "latency_ms": round(elapsed_ms, 2),
            "tokens_per_second": round(tps, 2),
            "eval_count": eval_count,
        }

    def generate_stream(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens for a grounded answer.

        Yields
        ------
        str
            Individual token strings as they are generated.
        """
        prompt = (system_prompt or RAG_SYSTEM_PROMPT).format(context=context)

        stream = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
            },
            stream=True,
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            yield token

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            self.client.list()
            return True
        except Exception:
            return False
