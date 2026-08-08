from __future__ import annotations

import csv
import random
import uuid
from pathlib import Path

import polars as pl
from locust import HttpUser, between, events, task
from locust.env import Environment

ITEMS_PATH = Path("data/processed/items.parquet")
RAW_LATENCIES_PATH = Path("bench/raw_latencies.csv")

REGIONS = ["IN-AP", "IN-KA", "IN-MH", "IN-DL", "IN-TN", "unknown"]
DEVICES = ["mobile", "desktop", "tablet"]
RETURNING_POOL_SIZE = 300
COLD_START_SHARE = 0.2
EVENT_BEFORE_FEED_SHARE = 0.35
FEED_LIMIT = 20

RequestSample = tuple[str, float, bool]


def load_catalog_ids(path: Path = ITEMS_PATH) -> list[int]:
    if not path.exists():
        return list(range(100_000, 100_200))
    return pl.read_parquet(path).select("article_id").to_series().to_list()


def load_categories(path: Path = ITEMS_PATH) -> list[str]:
    if not path.exists():
        return ["unknown"]
    return pl.read_parquet(path).select("category_l1").unique().to_series().to_list()


CATALOG_IDS = load_catalog_ids()
CATEGORIES = load_categories()
RETURNING_USER_IDS = [f"bench-user-{i}" for i in range(RETURNING_POOL_SIZE)]

_raw_samples: list[RequestSample] = []


def record_request(samples: list[RequestSample], name: str, response_time: float, exception: object) -> None:
    samples.append((name, response_time, exception is None))


def write_raw_samples(samples: list[RequestSample], path: Path = RAW_LATENCIES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "response_time_ms", "success"])
        writer.writerows(samples)


@events.request.add_listener
def _on_request(request_type: str, name: str, response_time: float, response_length: int, exception: object, **kwargs: object) -> None:
    record_request(_raw_samples, name, response_time, exception)


@events.test_stop.add_listener
def _on_test_stop(environment: Environment, **kwargs: object) -> None:
    write_raw_samples(_raw_samples)


class FeedUser(HttpUser):
    """Realistic mix: ~80% returning users drawn from a shared pool (who
    accumulate real session history as the run progresses), ~20% brand-new
    guests hitting the cold-start path. A third of feed requests are preceded
    by a fresh view event, mirroring "just looked at something" traffic."""

    wait_time = between(0.2, 1.5)

    def on_start(self) -> None:
        if random.random() < COLD_START_SHARE:
            self.user_id = f"bench-guest-{uuid.uuid4().hex[:10]}"
        else:
            self.user_id = random.choice(RETURNING_USER_IDS)
        self.session_id = f"{self.user_id}-s1"

    @task
    def feed(self) -> None:
        if random.random() < EVENT_BEFORE_FEED_SHARE:
            self._post_event()
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "limit": FEED_LIMIT,
            "context": {"device": random.choice(DEVICES), "region": random.choice(REGIONS)},
        }
        self.client.post("/v1/feed", json=payload, name="/v1/feed")

    def _post_event(self) -> None:
        article_id = random.choice(CATALOG_IDS)
        category = random.choice(CATEGORIES)
        self.client.post(
            "/v1/events",
            json={
                "user_id": self.user_id,
                "session_id": self.session_id,
                "article_id": int(article_id),
                "category_l1": category,
            },
            name="/v1/events",
        )
