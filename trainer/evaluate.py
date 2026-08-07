from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
ITEMS_PATH = PROCESSED_DIR / "items.parquet"
USERS_PATH = PROCESSED_DIR / "users.parquet"

ARTIFACTS_DIR = Path("artifacts")
TOWERS_PATH = ARTIFACTS_DIR / "towers.pt"
ITEM_VECTORS_PATH = ARTIFACTS_DIR / "item_vectors.npy"
ID_MAP_PATH = ARTIFACTS_DIR / "id_map.json"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
RESULTS_PATH = Path("docs/results.md")

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


def measure_latency_p50_ms(recommend_fn: RecommendFn, histories: list[list[int]]) -> float:
    durations_ms: list[float] = []
    for history in histories:
        start = time.perf_counter()
        recommend_fn("_latency_probe", history)
        durations_ms.append((time.perf_counter() - start) * 1000)
    return float(np.median(durations_ms)) if durations_ms else 0.0


def write_results_row(
    model_name: str, metrics: dict[str, float], latency_p50_ms: float, path: Path = RESULTS_PATH
) -> None:
    header = "| Model | Recall@20 | NDCG@20 | Coverage | Latency p50 |\n"
    separator = "| --- | --- | --- | --- | --- |\n"
    row = (
        f"| {model_name} | {metrics['recall@20']:.4f} | {metrics['ndcg@20']:.4f} | "
        f"{metrics['catalog_coverage']:.4f} | {latency_p50_ms:.2f}ms |\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    if "| Model |" not in existing:
        path.write_text(existing + header + separator + row)
        return

    lines = existing.splitlines(keepends=True)
    lines = [line for line in lines if not line.startswith(f"| {model_name} |")]
    insert_at = next((i for i, line in enumerate(lines) if line.startswith("| ---")), len(lines) - 1) + 1
    lines.insert(insert_at, row)
    path.write_text("".join(lines))


def load_towers(path: Path = TOWERS_PATH):
    import torch

    from trainer.train_towers import UserTower

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    user_tower = UserTower(
        len(checkpoint["age_band_vocab"]), len(checkpoint["region_vocab"]), embed_dim=config["embedding_dim"]
    )
    user_tower.load_state_dict(checkpoint["user_tower_state"])
    user_tower.eval()
    return user_tower, checkpoint["age_band_vocab"], checkpoint["region_vocab"]


def make_two_tower_recommend_fn(
    user_tower,
    demo_by_customer: dict[str, tuple[int, int]],
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
    index_to_article_id: dict[int, int],
    index,
    session_length: int = HISTORY_LENGTH,
    k: int = 20,
) -> RecommendFn:
    import torch

    def recommend(user_id: str, history: list[int]) -> list[int]:
        recent = history[-session_length:]
        idxs = [article_id_to_index[a] for a in recent if a in article_id_to_index]
        if idxs:
            history_emb = torch.tensor(item_vectors[idxs], dtype=torch.float32).unsqueeze(0)
            mask = torch.ones(1, len(idxs))
        else:
            history_emb = torch.zeros(1, 1, item_vectors.shape[1])
            mask = torch.zeros(1, 1)

        age_band_idx, region_idx = demo_by_customer.get(user_id, (0, 0))
        with torch.no_grad():
            user_emb = user_tower(history_emb, mask, torch.tensor([age_band_idx]), torch.tensor([region_idx]))
        query = user_emb.numpy().astype(np.float32)

        seen = set(history)
        fetch_k = min(index.ntotal, k + len(seen))
        _, indices = index.search(query, fetch_k)

        results: list[int] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            article_id = index_to_article_id[int(idx)]
            if article_id in seen:
                continue
            results.append(article_id)
            if len(results) >= k:
                break
        return results

    return recommend


def main() -> None:
    import faiss

    val = pl.read_parquet(VAL_PATH)
    users = pl.read_parquet(USERS_PATH)

    user_tower, age_band_vocab, region_vocab = load_towers()

    demo_by_customer = {
        customer_id: (age_band_vocab.get(age_band, 0), region_vocab.get(region, 0))
        for customer_id, age_band, region in zip(
            users["customer_id"].to_list(),
            users["age_band"].fill_null("unknown").to_list(),
            users["region"].fill_null("unknown").to_list(),
        )
    }

    item_vectors = np.load(ITEM_VECTORS_PATH)
    with ID_MAP_PATH.open() as f:
        id_map = json.load(f)
    article_id_to_index = {int(k): v for k, v in id_map["article_id_to_index"].items()}
    index_to_article_id = {int(k): v for k, v in id_map["index_to_article_id"].items()}

    index = faiss.read_index(str(INDEX_PATH))

    recommend_fn = make_two_tower_recommend_fn(
        user_tower, demo_by_customer, item_vectors, article_id_to_index, index_to_article_id, index
    )

    metrics = evaluate(recommend_fn, val, k=20)
    histories = list(_build_histories(TRAIN_PATH).values())
    latency_p50_ms = measure_latency_p50_ms(recommend_fn, histories[:200])

    print("Model     | Recall@20 | Precision@20 | NDCG@20 | Coverage | Entropy | Latency p50")
    print(
        f"Two-tower | {metrics['recall@20']:.4f}    | {metrics['precision@20']:.4f}       "
        f"| {metrics['ndcg@20']:.4f}  | {metrics['catalog_coverage']:.4f}   "
        f"| {metrics['mean_category_entropy']:.4f}  | {latency_p50_ms:.2f}ms"
    )

    write_results_row("Two-tower", metrics, latency_p50_ms)


if __name__ == "__main__":
    main()
