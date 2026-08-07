from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    postgres_dsn: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_DSN", "postgresql://localhost:5432/discovery")
    )
    anthropic_api_key: str | None = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    embedding_dim: int = 128
    intent_heads: int = 4
    intent_head_collapse_cosine: float = 0.8
    session_length: int = 20
    retrieval_k_per_query: int = 150
    retrieval_merge_size: int = 500
    diversity_cap: float = 0.35
    latency_budget_total_ms: float = 80.0
    latency_budget_retrieval_ms: float = 12.0
    latency_budget_ranking_ms: float = 25.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
