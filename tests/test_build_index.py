from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest

from trainer.build_index import build_index, save_index


def _random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vectors = rng.rand(n, dim).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_build_index_uses_inner_product_metric() -> None:
    index = build_index(_random_unit_vectors(20, 8))
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_build_index_sets_hnsw_params() -> None:
    index = build_index(_random_unit_vectors(20, 8), m=16, ef_construction=100, ef_search=32)
    assert index.hnsw.efConstruction == 100
    assert index.hnsw.efSearch == 32


def test_build_index_adds_all_vectors() -> None:
    index = build_index(_random_unit_vectors(37, 8))
    assert index.ntotal == 37


def test_build_index_search_finds_exact_match() -> None:
    vectors = _random_unit_vectors(50, 8)
    index = build_index(vectors)
    scores, indices = index.search(vectors[[5]], 1)
    assert indices[0][0] == 5
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)


def test_save_index_roundtrips(tmp_path: Path) -> None:
    vectors = _random_unit_vectors(10, 8)
    index = build_index(vectors)
    path = tmp_path / "index.faiss"
    save_index(index, path)

    loaded = faiss.read_index(str(path))
    assert loaded.ntotal == 10
    assert loaded.metric_type == faiss.METRIC_INNER_PRODUCT
