from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from trainer.evaluate import evaluate


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
