from __future__ import annotations

from src.shared.config import Settings
from src.shared.redis_client import get_redis_client


def test_client_parses_redis_url() -> None:
    settings = Settings(redis_url="redis://myhost:6390/2")
    client = get_redis_client(settings)
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["host"] == "myhost"
    assert kwargs["port"] == 6390
    assert kwargs["db"] == 2


def test_client_decodes_responses() -> None:
    settings = Settings(redis_url="redis://myhost:6390/2")
    client = get_redis_client(settings)
    assert client.connection_pool.connection_kwargs["decode_responses"] is True


def test_client_uses_default_settings_when_none_given() -> None:
    client = get_redis_client()
    assert client is not None
