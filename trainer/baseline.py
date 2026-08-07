from __future__ import annotations

import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from trainer.evaluate import _build_histories, evaluate

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
ARTIFACT_PATH = Path("artifacts/covisit.pkl")
RESULTS_PATH = Path("docs/results.md")

COVISIT_WINDOW_DAYS = 7
MAX_NEIGHBORS_PER_ITEM = 50
HISTORY_WINDOW = 5

CoVisitMap = dict[int, list[tuple[int, int]]]


def build_covisitation(train: pl.DataFrame, window_days: int = COVISIT_WINDOW_DAYS) -> CoVisitMap:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    grouped = train.sort("t_dat").group_by("customer_id", maintain_order=True).agg(
        pl.col("article_id"), pl.col("t_dat")
    )

    for articles, dates in zip(grouped["article_id"].to_list(), grouped["t_dat"].to_list()):
        n = len(articles)
        for i in range(n):
            for j in range(i + 1, n):
                if (dates[j] - dates[i]).days > window_days:
                    break
                if articles[i] == articles[j]:
                    continue
                counts[articles[i]][articles[j]] += 1
                counts[articles[j]][articles[i]] += 1

    return {item: neighbors.most_common(MAX_NEIGHBORS_PER_ITEM) for item, neighbors in counts.items()}


def save_covisitation(covisit: CoVisitMap, path: Path = ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(covisit, f)


def load_covisitation(path: Path = ARTIFACT_PATH) -> CoVisitMap:
    with path.open("rb") as f:
        return pickle.load(f)


def recommend(covisit: CoVisitMap, user_history: list[int], k: int = 20) -> list[int]:
    recent = user_history[-HISTORY_WINDOW:]
    seen = set(user_history)
    scores: Counter[int] = Counter()
    for item in recent:
        for neighbor, count in covisit.get(item, []):
            if neighbor in seen:
                continue
            scores[neighbor] += count
    return [item for item, _ in scores.most_common(k)]


def make_recommend_fn(covisit: CoVisitMap, k: int = 20) -> Callable[[str, list[int]], list[int]]:
    return lambda user_id, history: recommend(covisit, history, k)


def measure_latency_p50_ms(recommend_fn: Callable[[str, list[int]], list[int]], histories: list[list[int]]) -> float:
    durations_ms: list[float] = []
    for history in histories:
        start = time.perf_counter()
        recommend_fn("_latency_probe", history)
        durations_ms.append((time.perf_counter() - start) * 1000)
    return float(np.median(durations_ms)) if durations_ms else 0.0


def _write_results_row(metrics: dict[str, float], latency_p50_ms: float, path: Path = RESULTS_PATH) -> None:
    header = "| Model | Recall@20 | NDCG@20 | Coverage | Latency p50 |\n"
    separator = "| --- | --- | --- | --- | --- |\n"
    row = (
        f"| Baseline | {metrics['recall@20']:.4f} | {metrics['ndcg@20']:.4f} | "
        f"{metrics['catalog_coverage']:.4f} | {latency_p50_ms:.2f}ms |\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    if "| Model |" not in existing:
        path.write_text(existing + header + separator + row)
        return

    lines = existing.splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith("| Baseline |")]
    insert_at = next((i for i, line in enumerate(lines) if line.startswith("| ---")), len(lines) - 1) + 1
    lines.insert(insert_at, row)
    path.write_text("".join(lines))


def main() -> None:
    train = pl.read_parquet(TRAIN_PATH)
    val = pl.read_parquet(VAL_PATH)

    covisit = build_covisitation(train)
    save_covisitation(covisit)
    print(f"covisitation built: {len(covisit)} items")

    recommend_fn = make_recommend_fn(covisit)
    metrics = evaluate(recommend_fn, val, k=20)

    histories = list(_build_histories(TRAIN_PATH).values())
    latency_p50_ms = measure_latency_p50_ms(recommend_fn, histories[:200])

    print("Model    | Recall@20 | Precision@20 | NDCG@20 | Coverage | Entropy | Latency p50")
    print(
        f"Baseline | {metrics['recall@20']:.4f}    | {metrics['precision@20']:.4f}       "
        f"| {metrics['ndcg@20']:.4f}  | {metrics['catalog_coverage']:.4f}   "
        f"| {metrics['mean_category_entropy']:.4f}  | {latency_p50_ms:.2f}ms"
    )

    _write_results_row(metrics, latency_p50_ms)


if __name__ == "__main__":
    main()
