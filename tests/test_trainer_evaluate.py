from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trainer.build_index import build_index
from trainer.evaluate import evaluate, make_two_tower_recommend_fn, measure_latency_p50_ms, write_results_row
from trainer.train_towers import UserTower


@pytest.fixture()
def fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    train = pl.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1", "u2"],
            "article_id": [10, 20, 30, 40],
            "t_dat": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 1)],
        }
    )
    items = pl.DataFrame(
        {
            "article_id": [10, 20, 30, 40, 101, 102, 103],
            "category_l1": ["A", "A", "B", "B", "C", "C", "D"],
        }
    )

    train_path = tmp_path / "train.parquet"
    items_path = tmp_path / "items.parquet"
    train.write_parquet(train_path)
    items.write_parquet(items_path)
    return train_path, items_path


def _static_recommend_fn(user_id: str, history: list[int]) -> list[int]:
    return [101, 102, 103]


def test_evaluate_computes_expected_metrics(fixture_paths: tuple[Path, Path]) -> None:
    train_path, items_path = fixture_paths
    val_df = pl.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2"],
            "article_id": [101, 999, 102],
        }
    )

    metrics = evaluate(_static_recommend_fn, val_df, k=3, train_path=train_path, items_path=items_path)

    # u1: relevant={101, 999}, recommended=[101,102,103] -> hit=101 only
    u1_recall = 1 / 2
    u1_precision = 1 / 3
    u1_dcg = 1 / math.log2(2)
    u1_idcg = 1 / math.log2(2) + 1 / math.log2(3)
    u1_ndcg = u1_dcg / u1_idcg

    # u2: relevant={102}, recommended=[101,102,103] -> hit=102 only
    u2_recall = 1 / 1
    u2_precision = 1 / 3
    u2_dcg = 1 / math.log2(3)
    u2_idcg = 1 / math.log2(2)
    u2_ndcg = u2_dcg / u2_idcg

    entropy = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))  # categories C,C,D for both users

    assert metrics["recall@20"] == pytest.approx((u1_recall + u2_recall) / 2)
    assert metrics["precision@20"] == pytest.approx((u1_precision + u2_precision) / 2)
    assert metrics["ndcg@20"] == pytest.approx((u1_ndcg + u2_ndcg) / 2)
    assert metrics["catalog_coverage"] == pytest.approx(3 / 7)  # {101,102,103} of 7 catalog items
    assert metrics["mean_category_entropy"] == pytest.approx(entropy)


def test_evaluate_skips_users_with_no_train_history(fixture_paths: tuple[Path, Path]) -> None:
    train_path, items_path = fixture_paths
    val_df = pl.DataFrame({"customer_id": ["cold_start_user"], "article_id": [101]})

    metrics = evaluate(_static_recommend_fn, val_df, k=3, train_path=train_path, items_path=items_path)

    assert metrics == {
        "recall@20": 0.0,
        "precision@20": 0.0,
        "ndcg@20": 0.0,
        "catalog_coverage": 0.0,
        "mean_category_entropy": 0.0,
    }


def test_evaluate_perfect_recommendations_score_one(fixture_paths: tuple[Path, Path]) -> None:
    train_path, items_path = fixture_paths
    val_df = pl.DataFrame({"customer_id": ["u1"], "article_id": [101]})

    def perfect_fn(user_id: str, history: list[int]) -> list[int]:
        return [101]

    metrics = evaluate(perfect_fn, val_df, k=1, train_path=train_path, items_path=items_path)

    assert metrics["recall@20"] == pytest.approx(1.0)
    assert metrics["precision@20"] == pytest.approx(1.0)
    assert metrics["ndcg@20"] == pytest.approx(1.0)


def test_measure_latency_p50_ms_returns_non_negative_median() -> None:
    def recommend_fn(user_id: str, history: list[int]) -> list[int]:
        return [1, 2, 3]

    latency = measure_latency_p50_ms(recommend_fn, [[1], [1], [1]])
    assert latency >= 0.0


def test_measure_latency_p50_ms_empty_histories() -> None:
    def recommend_fn(user_id: str, history: list[int]) -> list[int]:
        return []

    assert measure_latency_p50_ms(recommend_fn, []) == 0.0


def test_write_results_row_creates_table_and_dedupes_on_rerun(tmp_path: Path) -> None:
    path = tmp_path / "results.md"
    metrics = {"recall@20": 0.1234, "ndcg@20": 0.5678, "catalog_coverage": 0.4321}

    write_results_row("Two-tower", metrics, latency_p50_ms=1.5, path=path)
    first = path.read_text()
    assert "| Two-tower | 0.1234 | 0.5678 | 0.4321 | 1.50ms |" in first
    assert first.count("| Two-tower |") == 1

    write_results_row("Two-tower", metrics, latency_p50_ms=2.5, path=path)
    second = path.read_text()
    assert second.count("| Two-tower |") == 1
    assert "2.50ms" in second


def test_write_results_row_preserves_other_model_rows(tmp_path: Path) -> None:
    path = tmp_path / "results.md"
    metrics = {"recall@20": 0.1, "ndcg@20": 0.2, "catalog_coverage": 0.3}

    write_results_row("Baseline", metrics, latency_p50_ms=1.0, path=path)
    write_results_row("Two-tower", metrics, latency_p50_ms=2.0, path=path)

    content = path.read_text()
    assert content.count("| Baseline |") == 1
    assert content.count("| Two-tower |") == 1


def test_make_two_tower_recommend_fn_excludes_history_and_respects_k() -> None:
    embed_dim = 8
    n_items = 20
    rng = np.random.RandomState(0)
    item_vectors = rng.rand(n_items, embed_dim).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)

    index = build_index(item_vectors)
    article_id_to_index = {100 + i: i for i in range(n_items)}
    index_to_article_id = {i: 100 + i for i in range(n_items)}

    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=embed_dim)
    user_tower.eval()

    demo_by_customer = {"u1": (0, 0)}
    recommend_fn = make_two_tower_recommend_fn(
        user_tower,
        demo_by_customer,
        item_vectors,
        article_id_to_index,
        index_to_article_id,
        index,
        session_length=20,
        k=5,
    )

    history = [100, 101, 102]
    recommended = recommend_fn("u1", history)

    assert len(recommended) == 5
    assert set(recommended).isdisjoint(history)


def test_make_two_tower_recommend_fn_handles_empty_history() -> None:
    embed_dim = 8
    n_items = 10
    item_vectors = np.random.RandomState(1).rand(n_items, embed_dim).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)

    index = build_index(item_vectors)
    article_id_to_index = {100 + i: i for i in range(n_items)}
    index_to_article_id = {i: 100 + i for i in range(n_items)}

    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=embed_dim)
    user_tower.eval()

    recommend_fn = make_two_tower_recommend_fn(
        user_tower, {}, item_vectors, article_id_to_index, index_to_article_id, index, session_length=20, k=3
    )

    assert len(recommend_fn("cold_start_user", [])) == 3
