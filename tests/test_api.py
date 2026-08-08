from __future__ import annotations

import uuid
from collections import Counter
from types import SimpleNamespace

import httpx
import json
import lightgbm as lgb
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import src.gateway.main as gateway_main
from retrieval_service.main import search_batch
from src.ranking.features import FEATURE_NAMES
from src.session.encoder import SessionEncoder
from src.shared.config import Settings
from src.shared.db import get_redis_client
from trainer.build_index import build_index
from trainer.train_towers import UserTower

CATEGORIES = ["tops", "bottoms", "shoes"]
N_ITEMS = 60
ARTICLE_BASE = 5000
TEST_EMBED_DIM = 8


def _build_catalog(seed: int = 0):
    rng = np.random.RandomState(seed)
    item_vectors = rng.rand(N_ITEMS, TEST_EMBED_DIM).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)

    article_ids = [ARTICLE_BASE + i for i in range(N_ITEMS)]
    article_id_to_index = {a: i for i, a in enumerate(article_ids)}
    index_to_article_id = {i: a for i, a in enumerate(article_ids)}
    category_by_article = {a: CATEGORIES[i % 3] for i, a in enumerate(article_ids)}

    item_meta = {
        a: gateway_main.ItemMeta(
            category_l1=category_by_article[a], popularity=0.5, ctr=0.1, avg_price=20.0, days_since_first_seen=100.0
        )
        for a in article_ids
    }

    category_centroid: dict[str, list[float]] = {}
    for category in CATEGORIES:
        idxs = [article_id_to_index[a] for a in article_ids if category_by_article[a] == category]
        category_centroid[category] = item_vectors[idxs].mean(axis=0).tolist()
    global_centroid = item_vectors.mean(axis=0)

    return SimpleNamespace(
        item_vectors=item_vectors,
        article_id_to_index=article_id_to_index,
        index_to_article_id=index_to_article_id,
        category_by_article=category_by_article,
        item_meta=item_meta,
        category_centroid=category_centroid,
        global_centroid=global_centroid,
        article_ids=article_ids,
    )


def _tiny_boosters(seed: int = 0) -> dict[str, lgb.Booster]:
    rng = np.random.RandomState(seed)
    n = 80
    X = rng.rand(n, len(FEATURE_NAMES)).astype(np.float32)
    params = {"objective": "binary", "verbosity": -1, "num_leaves": 7, "min_data_in_leaf": 5}
    boosters: dict[str, lgb.Booster] = {}
    for objective in gateway_main.OBJECTIVES:
        y = rng.randint(0, 2, size=n).astype(np.int32)
        boosters[objective] = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=10)
    return boosters


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        self.executed.append((query, params))

    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class _FakePGConnection:
    def __init__(self) -> None:
        self._cursor = _FakeCursor()
        self.committed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True


def _make_mock_transport(index, index_to_article_id: dict[int, int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        vectors = np.array(payload["vectors"], dtype=np.float32)
        k = payload["k"]
        exclude = set(payload.get("exclude", []))
        results = search_batch(index, index_to_article_id, vectors, k, exclude)
        body = {
            "results": [[{"id": r.id, "score": r.score} for r in row] for row in results],
            "took_ms": 0.1,
        }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.fixture()
def gateway_state(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    catalog = _build_catalog()

    torch.manual_seed(0)
    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=TEST_EMBED_DIM)
    user_tower.eval()
    session_encoder = SessionEncoder(embed_dim=TEST_EMBED_DIM, n_heads=2, n_layers=2, n_intents=4)
    session_encoder.eval()

    index = build_index(catalog.item_vectors)
    http_client = httpx.Client(transport=_make_mock_transport(index, catalog.index_to_article_id))

    redis_client = get_redis_client(Settings(_env_file=None))

    monkeypatch.setattr(
        gateway_main.state,
        "settings",
        Settings(retrieval_k_per_query=30, retrieval_merge_size=60, model_version="test", _env_file=None),
    )
    monkeypatch.setattr(gateway_main.state, "item_vectors", catalog.item_vectors)
    monkeypatch.setattr(gateway_main.state, "article_id_to_index", catalog.article_id_to_index)
    monkeypatch.setattr(gateway_main.state, "index_to_article_id", catalog.index_to_article_id)
    monkeypatch.setattr(gateway_main.state, "user_tower", user_tower)
    monkeypatch.setattr(gateway_main.state, "age_band_vocab", {"unknown": 0})
    monkeypatch.setattr(gateway_main.state, "region_vocab", {"unknown": 0})
    monkeypatch.setattr(gateway_main.state, "session_encoder", session_encoder)
    monkeypatch.setattr(gateway_main.state, "boosters", _tiny_boosters())
    monkeypatch.setattr(gateway_main.state, "item_meta", catalog.item_meta)
    monkeypatch.setattr(gateway_main.state, "category_centroid", catalog.category_centroid)
    monkeypatch.setattr(gateway_main.state, "global_centroid", catalog.global_centroid)
    monkeypatch.setattr(gateway_main.state, "category_vocab", CATEGORIES)
    monkeypatch.setattr(gateway_main.state, "http_client", http_client)
    monkeypatch.setattr(gateway_main.state, "redis_client", redis_client)
    monkeypatch.setattr(gateway_main.state, "pg_conn_factory", lambda: _FakePGConnection())

    return SimpleNamespace(article_ids=catalog.article_ids, category_by_article=catalog.category_by_article)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(gateway_main.app)


# --- healthz ---


def test_healthz_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- feed ---


def test_feed_returns_20_items_with_intents_and_respects_diversity_cap(
    client: TestClient, gateway_state: SimpleNamespace
) -> None:
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    session_id = f"sess-{uuid.uuid4().hex[:8]}"

    for article_id in gateway_state.article_ids[:5]:
        response = client.post(
            "/v1/events",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "article_id": article_id,
                "category_l1": gateway_state.category_by_article[article_id],
            },
        )
        assert response.status_code == 202

    response = client.post("/v1/feed", json={"user_id": user_id, "session_id": session_id, "limit": 20})
    assert response.status_code == 200
    body = response.json()

    assert len(body["items"]) == 20
    assert len(body["detected_intents"]) == 4
    for intent in body["detected_intents"]:
        assert set(intent) == {"label", "confidence", "slots"}

    counts = Counter(gateway_state.category_by_article[int(item["product_id"])] for item in body["items"])
    for count in counts.values():
        assert count / len(body["items"]) <= 0.35 + 1e-9

    assert body["degraded"] is None
    assert body["latency_ms"] >= 0
    assert all(item["reason"]["code"] for item in body["items"])


# --- events ---


def test_events_returns_202(client: TestClient, gateway_state: SimpleNamespace) -> None:
    response = client.post(
        "/v1/events",
        json={
            "user_id": f"user-{uuid.uuid4().hex[:8]}",
            "session_id": "sess-1",
            "article_id": gateway_state.article_ids[0],
        },
    )
    assert response.status_code == 202
    assert response.content == b""


# --- delete + cold start ---


def test_delete_user_then_feed_takes_cold_start_path(client: TestClient, gateway_state: SimpleNamespace) -> None:
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    session_id = f"sess-{uuid.uuid4().hex[:8]}"

    for article_id in gateway_state.article_ids[:3]:
        client.post("/v1/events", json={"user_id": user_id, "session_id": session_id, "article_id": article_id})

    warm = client.post("/v1/feed", json={"user_id": user_id, "session_id": session_id, "limit": 5})
    assert warm.status_code == 200
    assert warm.json()["detected_intents"][0]["label"] != "cold_start"

    redis_client = gateway_main.state.redis_client
    redis_client.delete("pop:unknown:unknown")
    redis_client.zadd("pop:unknown:unknown", {str(a): float(50 - i) for i, a in enumerate(gateway_state.article_ids[:10])})

    try:
        delete_response = client.delete(f"/v1/users/{user_id}")
        assert delete_response.status_code == 204
        assert delete_response.headers["X-Deletion-Receipt-Id"]

        cold_response = client.post("/v1/feed", json={"user_id": user_id, "session_id": session_id, "limit": 5})
        assert cold_response.status_code == 200
        body = cold_response.json()
        assert len(body["items"]) > 0
        assert body["detected_intents"] == [
            {"label": "cold_start", "confidence": 1.0, "slots": len(body["items"])}
        ]
        assert all(item["reason"]["code"] == "POPULAR" for item in body["items"])
    finally:
        redis_client.delete("pop:unknown:unknown")


# --- search ---


def test_search_matches_category_keyword_and_returns_items(client: TestClient, gateway_state: SimpleNamespace) -> None:
    response = client.post("/v1/search", json={"query": "tops", "limit": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["parsed_intent"]["category_l1"] == "tops"
    assert len(body["items"]) > 0
    assert all(item["reason"]["code"] == "SEARCH_MATCH" for item in body["items"])


# --- bundles ---


def test_bundles_returns_embedding_neighbours(client: TestClient, gateway_state: SimpleNamespace) -> None:
    seed_id = gateway_state.article_ids[0]
    response = client.post("/v1/bundles", json={"seed_product_id": seed_id, "limit": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) > 0
    assert all(item["reason"]["code"] in {"COMPLEMENT", "SIMILAR_STYLE"} for item in body["items"])


# --- guardrails: fail closed, never a fallback ---


def test_guardrails_failure_returns_503_never_a_fallback(
    client: TestClient, gateway_state: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args: object, **kwargs: object):
        raise RuntimeError("boom")

    monkeypatch.setattr(gateway_main, "apply_all", _raise)

    response = client.post(
        "/v1/feed", json={"user_id": f"user-{uuid.uuid4().hex[:8]}", "session_id": "sess-1", "limit": 5}
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "GUARDRAILS_UNAVAILABLE"
    assert body["error"]["retryable"] is True


# --- helper unit tests ---


def test_session_key_prefers_user_id_over_session_id() -> None:
    assert gateway_main._session_key("u1", "s1") == "sess:u1"
    assert gateway_main._session_key(None, "s1") == "sess:s1"
    assert gateway_main._session_key(None, None) == "sess:anonymous"


def test_merge_candidates_keeps_best_score_and_first_source_label() -> None:
    results = [
        [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}],
        [{"id": 1, "score": 0.95}, {"id": 3, "score": 0.7}],
    ]
    ids, scores, source = gateway_main._merge_candidates(results, ["a", "b"], merge_size=10)
    assert set(ids) == {1, 2, 3}
    assert scores[1] == pytest.approx(0.95)
    assert source[1] == "a"


def test_reason_for_cold_start_uses_popular_code() -> None:
    reason = gateway_main._reason_for(gateway_main.COLD_START_LABEL, 0)
    assert reason.code == "POPULAR"


def test_reason_for_session_affinity_when_category_seen() -> None:
    reason = gateway_main._reason_for("seasonal_browsing", 3)
    assert reason.code == "SESSION_AFFINITY"
    assert "3" in reason.text


def test_reason_for_intent_match_when_no_category_overlap() -> None:
    reason = gateway_main._reason_for("bargain_hunting", 0)
    assert reason.code == "INTENT_MATCH"
