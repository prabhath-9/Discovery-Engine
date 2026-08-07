from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from retrieval_service import main as service
from retrieval_service.main import search_batch
from trainer.build_index import build_index


def _random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vectors = rng.rand(n, dim).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture()
def small_index() -> tuple:
    vectors = _random_unit_vectors(50, 8)
    index = build_index(vectors)
    index_to_article_id = {i: 1000 + i for i in range(50)}
    return index, index_to_article_id


def test_search_batch_returns_k_results(small_index: tuple) -> None:
    index, index_to_article_id = small_index
    query = _random_unit_vectors(3, 8, seed=1)

    results = search_batch(index, index_to_article_id, query, k=5, exclude=set())

    assert len(results) == 3
    for row in results:
        assert len(row) == 5


def test_search_batch_latency_under_10ms_for_four_queries() -> None:
    vectors = _random_unit_vectors(3000, 128, seed=2)
    index = build_index(vectors)
    index_to_article_id = {i: i for i in range(3000)}
    query = vectors[:4]

    search_batch(index, index_to_article_id, query, k=20, exclude=set())  # warm up

    start = time.perf_counter()
    search_batch(index, index_to_article_id, query, k=20, exclude=set())
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 10.0


def test_search_batch_never_returns_excluded_ids() -> None:
    vectors = _random_unit_vectors(30, 8)
    index = build_index(vectors)
    index_to_article_id = {i: 100 + i for i in range(30)}

    query = vectors[[0, 5, 10]]
    exclude = {100, 101, 102, 105, 110}

    results = search_batch(index, index_to_article_id, query, k=5, exclude=exclude)

    returned_ids = {result.id for row in results for result in row}
    assert returned_ids.isdisjoint(exclude)


def test_healthz_always_ok() -> None:
    client = TestClient(service.app)
    response = client.get("/healthz")
    assert response.status_code == 200


def test_readyz_503_before_index_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.state, "index", None)
    client = TestClient(service.app)
    response = client.get("/readyz")
    assert response.status_code == 503


def test_readyz_200_after_index_loaded(monkeypatch: pytest.MonkeyPatch, small_index: tuple) -> None:
    index, index_to_article_id = small_index
    monkeypatch.setattr(service.state, "index", index)
    monkeypatch.setattr(service.state, "index_to_article_id", index_to_article_id)
    client = TestClient(service.app)
    response = client.get("/readyz")
    assert response.status_code == 200


def test_search_endpoint_end_to_end(monkeypatch: pytest.MonkeyPatch, small_index: tuple) -> None:
    index, index_to_article_id = small_index
    monkeypatch.setattr(service.state, "index", index)
    monkeypatch.setattr(service.state, "index_to_article_id", index_to_article_id)
    client = TestClient(service.app)

    query = _random_unit_vectors(2, 8, seed=3).tolist()
    response = client.post("/search", json={"vectors": query, "k": 4, "exclude": []})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert all(len(row) == 4 for row in body["results"])
    assert body["took_ms"] >= 0


def test_search_endpoint_503_before_index_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.state, "index", None)
    monkeypatch.setattr(service.state, "index_to_article_id", None)
    client = TestClient(service.app)

    response = client.post("/search", json={"vectors": [[0.1] * 8], "k": 4, "exclude": []})

    assert response.status_code == 503
