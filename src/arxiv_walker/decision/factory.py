from __future__ import annotations

import os

from arxiv_walker.config import DecisionConfig
from arxiv_walker.decision.base import DecisionEngine, DecisionError
from arxiv_walker.decision.jev_openrouter import OpenRouterJevEngine
from arxiv_walker.decision.mock import MockDecisionEngine


def make_decision_engine(settings: DecisionConfig) -> DecisionEngine:
    backend = settings.backend
    if backend == "jev":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise DecisionError(
                "OPENROUTER_API_KEY environment variable is not set"
            )
        return OpenRouterJevEngine(
            api_key=api_key,
            model=settings.model,
            base_url=settings.base_url,
            timeout_sec=settings.timeout_sec,
            max_retries=settings.max_retries,
        )
    if backend == "mock":
        return MockDecisionEngine()
    raise ValueError(f"Unknown decision backend: {backend}")
