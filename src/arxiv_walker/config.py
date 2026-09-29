from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DecisionConfig(BaseModel):
    backend: str = "jev"
    model: str = "typesafe/jev-1.13"
    base_url: str = "https://openrouter.ai/api/alpha"
    timeout_sec: int = 60
    max_retries: int = 3


class YearBucketsConfig(BaseModel):
    recent_days: int = 730
    medium_days: int = 1825


class IngestConfig(BaseModel):
    batch_size: int = 50000
    year_buckets: YearBucketsConfig = Field(default_factory=YearBucketsConfig)


class SearchConfig(BaseModel):
    beam_width: int = 3
    max_depth: int = 5
    branch_threshold: float = 0.35
    candidate_cutoff: int = 200
    bm25_top_k: int = 100
    rerank_top_k: int = 50
    final_top_k: int = 25
    exploration_branches: int = 3
    extra_lexical_branch: bool = True


class Settings(BaseSettings):
    model_config = {"env_prefix": "ARXIV_WALKER_", "env_nested_delimiter": "__"}

    data_dir: str = "data"
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)


def load_settings(path: str | None = None) -> Settings:
    if path is None:
        path = str(Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml")
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Settings(**raw)
