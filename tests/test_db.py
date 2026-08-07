from __future__ import annotations

from typing import Any

import pytest

import src.shared.db as db
from src.shared.config import Settings


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "_redis_client", None)
    monkeypatch.setattr(db, "_pg_connection", None)


class _FakePGConnection:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.closed = 0


def test_redis_client_parses_url_from_explicit_settings() -> None:
    settings = Settings(redis_url="redis://myhost:6390/2", _env_file=None)
    client = db.get_redis_client(settings)
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["host"] == "myhost"
    assert kwargs["port"] == 6390
    assert kwargs["db"] == 2
    assert kwargs["decode_responses"] is True


def test_redis_client_is_a_lazily_initialised_singleton() -> None:
    first = db.get_redis_client()
    second = db.get_redis_client()
    assert first is second


def test_redis_client_explicit_settings_bypasses_singleton() -> None:
    cached = db.get_redis_client()
    scoped = db.get_redis_client(Settings(redis_url="redis://other:6379/0", _env_file=None))
    assert scoped is not cached


def test_pg_connection_is_a_lazily_initialised_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_connect(dsn: str, **kwargs: Any) -> _FakePGConnection:
        calls.append(dsn)
        return _FakePGConnection(dsn)

    monkeypatch.setattr(db.psycopg2, "connect", fake_connect)

    first = db.get_pg_connection()
    second = db.get_pg_connection()

    assert first is second
    assert calls == [db.get_settings().pg_dsn]


def test_pg_connection_reconnects_if_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_connect(dsn: str, **kwargs: Any) -> _FakePGConnection:
        calls.append(dsn)
        return _FakePGConnection(dsn)

    monkeypatch.setattr(db.psycopg2, "connect", fake_connect)

    first = db.get_pg_connection()
    first.closed = 1
    second = db.get_pg_connection()

    assert second is not first
    assert len(calls) == 2


def test_pg_connection_explicit_settings_bypasses_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_connect(dsn: str, **kwargs: Any) -> _FakePGConnection:
        return _FakePGConnection(dsn)

    monkeypatch.setattr(db.psycopg2, "connect", fake_connect)

    cached = db.get_pg_connection()
    scoped = db.get_pg_connection(Settings(pg_dsn="postgresql://other:5432/x", _env_file=None))

    assert scoped is not cached
    assert scoped.dsn == "postgresql://other:5432/x"
