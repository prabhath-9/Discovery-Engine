from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
ITEMS_PATH = PROCESSED_DIR / "items.parquet"

HISTORY_LENGTH = 20

RecommendFn = Callable[[str, list[int]], list[int]]


def _build_histories(train_path: Path, history_length: int = HISTORY_LENGTH) -> dict[str, list[int]]:
    train = pl.read_parquet(train_path).sort("t_dat")
    grouped = train.group_by("customer_id", maintain_order=True).agg(pl.col("article_id"))
    return {
        customer_id: articles[-history_length:]
        for customer_id, articles in zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list())
    }


def _load_categories(items_path: Path) -> dict[int, str]:
    items = pl.read_parquet(items_path)
    return dict(zip(items["article_id"].to_list(), items["category_l1"].to_list()))


def _dcg(relevances: list[int]) -> float:
    return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances)))


def _ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = _dcg([1 if item in relevant else 0 for item in recommended[:k]])
    idcg = _dcg([1] * min(len(relevant), k))
    return dcg / idcg if idcg > 0 else 0.0


def _category_entropy(categories: list[str]) -> float:
    if not categories:
        return 0.0
    counts = Counter(categories)
    total = len(categories)
    probs = np.array([count / total for count in counts.values()])
    return float(-(probs * np.log2(probs)).sum())


def evaluate(
    recommend_fn: RecommendFn,
    val_df: pl.DataFrame,
    k: int = 20,
    train_path: Path = TRAIN_PATH,
    items_path: Path = ITEMS_PATH,
) -> dict[str, float]:
    histories = _build_histories(train_path)
    category_by_item = _load_categories(items_path)
    catalog_size = len(category_by_item)

    relevant_by_user: dict[str, set[int]] = defaultdict(set)
    for customer_id, article_id in zip(val_df["customer_id"].to_list(), val_df["article_id"].to_list()):
        relevant_by_user[customer_id].add(article_id)

    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    entropies: list[float] = []
    recommended_items: set[int] = set()

    for user_id, relevant in relevant_by_user.items():
        history = histories.get(user_id)
        if not history:
            continue

        recommended = recommend_fn(user_id, history)[:k]
        recommended_items.update(recommended)

        hits = sum(1 for item in recommended if item in relevant)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(hits / k if k else 0.0)
        ndcgs.append(_ndcg_at_k(recommended, relevant, k))
        entropies.append(_category_entropy([category_by_item[i] for i in recommended if i in category_by_item]))

    n_evaluated = len(recalls)
    return {
        "recall@20": float(np.mean(recalls)) if n_evaluated else 0.0,
        "precision@20": float(np.mean(precisions)) if n_evaluated else 0.0,
        "ndcg@20": float(np.mean(ndcgs)) if n_evaluated else 0.0,
        "catalog_coverage": len(recommended_items) / catalog_size if catalog_size else 0.0,
        "mean_category_entropy": float(np.mean(entropies)) if n_evaluated else 0.0,
    }
