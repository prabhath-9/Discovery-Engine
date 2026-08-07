from __future__ import annotations

import pytest

from src.shared.config import Settings, get_settings


def test_defaults_match_spec() -> None:
    settings = Settings(_env_file=None)
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.pg_dsn == "postgresql://localhost:5432/discovery"
    assert settings.anthropic_api_key is None
    assert settings.model_version == "dev"
    assert settings.embedding_dim == 128
    assert settings.intent_heads == 4
    assert settings.intent_head_collapse_cosine == 0.8
    assert settings.session_length == 20
    assert settings.retrieval_k_per_query == 150
    assert settings.retrieval_merge_size == 500
    assert settings.diversity_cap == 0.35
    assert settings.latency_budget_total_ms == 80.0
    assert settings.latency_budget_retrieval_ms == 12.0
    assert settings.latency_budget_ranking_ms == 25.0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://example:6380/1")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("PG_DSN", "postgresql://example:5432/prod")
    monkeypatch.setenv("MODEL_VERSION", "2026-08-07-a")
    settings = Settings(_env_file=None)
    assert settings.redis_url == "redis://example:6380/1"
    assert settings.environment == "production"
    assert settings.pg_dsn == "postgresql://example:5432/prod"
    assert settings.model_version == "2026-08-07-a"


def test_anthropic_key_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is None


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
