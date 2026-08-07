from __future__ import annotations

import redis

from src.shared.config import Settings, get_settings


def get_redis_client(settings: Settings | None = None) -> redis.Redis:
    settings = settings or get_settings()
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)
