from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

PROCESSED_DIR = Path("data/processed")


def _load(name: str) -> pl.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not found; run `python -m trainer.sample_data` first")
    return pl.read_parquet(path)


def test_val_max_date_after_train_max_date() -> None:
    train = _load("train")
    val = _load("val")
    assert val["t_dat"].max() > train["t_dat"].max()


def test_no_user_or_item_leakage_outside_the_sampled_catalog() -> None:
    train = _load("train")
    val = _load("val")
    items = _load("items")
    users = _load("users")

    transacted_articles = set(train["article_id"]) | set(val["article_id"])
    transacted_customers = set(train["customer_id"]) | set(val["customer_id"])

    assert transacted_articles <= set(items["article_id"])
    assert transacted_customers <= set(users["customer_id"])


def test_items_category_l1_has_no_nulls() -> None:
    items = _load("items")
    assert items["category_l1"].null_count() == 0
