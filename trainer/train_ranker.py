from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

from src.ranking.features import FEATURE_NAMES, RankingCandidate, UserFeatures, build_features
from src.session.encoder import EMBED_DIM, N_INTENTS, SessionEncoder
from src.session.intents import COLD_START_LABEL, Intent, SessionEvent, infer_intents
from trainer.evaluate import RecommendFn, evaluate, measure_latency_p50_ms, write_results_row
from trainer.train_towers import UserTower

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
SESSION_ENCODER_PATH = ARTIFACTS_DIR / "session_encoder.pt"
RESULTS_PATH = Path("docs/results.md")

OBJECTIVES = ("click", "cart", "purchase", "wishlist")
COMBINED_WEIGHTS = {"click": 0.2, "cart": 0.3, "purchase": 0.5}

SESSION_LENGTH = 20
N_NEGATIVES = 4
MAX_TRAIN_SESSIONS = 8_000
MAX_VAL_SESSIONS = 2_000
CART_PROB = 0.4
WISHLIST_PROB = 0.2
CTR_SMOOTHING = 20.0
RANDOM_SEED = 0
RETRIEVAL_K = 150
LATENCY_CANDIDATES = 500
LATENCY_BUDGET_MS = 25.0


@dataclass(frozen=True, slots=True)
class Transaction:
    article_id: int
    t_dat: date
    price: float


@dataclass(frozen=True, slots=True)
class RankerExample:
    customer_id: str
    history: list[Transaction]
    target: Transaction


@dataclass(frozen=True, slots=True)
class ItemMeta:
    category_l1: str
    popularity: float
    ctr: float
    first_seen: date
    avg_price: float


def build_customer_histories(train: pl.DataFrame) -> dict[str, list[Transaction]]:
    grouped = train.sort("t_dat").group_by("customer_id", maintain_order=True).agg(
        pl.col("article_id"), pl.col("t_dat"), pl.col("price")
    )
    histories: dict[str, list[Transaction]] = {}
    for customer_id, article_ids, dates, prices in zip(
        grouped["customer_id"].to_list(),
        grouped["article_id"].to_list(),
        grouped["t_dat"].to_list(),
        grouped["price"].to_list(),
    ):
        histories[customer_id] = [Transaction(a, d, p) for a, d, p in zip(article_ids, dates, prices)]
    return histories


def build_train_examples(
    histories: dict[str, list[Transaction]], session_length: int = SESSION_LENGTH
) -> list[RankerExample]:
    examples: list[RankerExample] = []
    for customer_id, tx in histories.items():
        for i in range(1, len(tx)):
            history = tx[max(0, i - session_length) : i]
            examples.append(RankerExample(customer_id, history, tx[i]))
    return examples


def build_val_examples(
    train_histories: dict[str, list[Transaction]], val: pl.DataFrame, session_length: int = SESSION_LENGTH
) -> list[RankerExample]:
    examples: list[RankerExample] = []
    for customer_id, article_id, t_dat, price in zip(
        val["customer_id"].to_list(), val["article_id"].to_list(), val["t_dat"].to_list(), val["price"].to_list()
    ):
        history = train_histories.get(customer_id)
        if not history:
            continue
        examples.append(RankerExample(customer_id, history[-session_length:], Transaction(article_id, t_dat, price)))
    return examples


def subsample_examples(examples: list[RankerExample], max_examples: int, seed: int = RANDOM_SEED) -> list[RankerExample]:
    if len(examples) <= max_examples:
        return examples
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(examples), size=max_examples, replace=False)
    return [examples[i] for i in idx]


def build_item_meta(items: pl.DataFrame, train: pl.DataFrame) -> dict[int, ItemMeta]:
    max_n = items["n_interactions"].max() or 1
    category_by_item = dict(zip(items["article_id"].to_list(), items["category_l1"].to_list()))
    n_by_item = dict(zip(items["article_id"].to_list(), items["n_interactions"].to_list()))

    price_stats = train.group_by("article_id").agg(
        pl.col("t_dat").min().alias("first_seen"), pl.col("price").mean().alias("avg_price")
    )
    first_seen_by_item = dict(zip(price_stats["article_id"].to_list(), price_stats["first_seen"].to_list()))
    avg_price_by_item = dict(zip(price_stats["article_id"].to_list(), price_stats["avg_price"].to_list()))

    meta: dict[int, ItemMeta] = {}
    for article_id in items["article_id"].to_list():
        n = n_by_item.get(article_id, 0)
        meta[article_id] = ItemMeta(
            category_l1=category_by_item.get(article_id, "unknown"),
            popularity=n / max_n,
            ctr=n / (n + CTR_SMOOTHING),
            first_seen=first_seen_by_item.get(article_id, date(1970, 1, 1)),
            avg_price=avg_price_by_item.get(article_id, 0.0),
        )
    return meta


def _to_datetime(d: date) -> datetime:
    return datetime.combine(d, dt_time.min)


def build_session(history: list[Transaction], item_meta: dict[int, ItemMeta]) -> list[SessionEvent]:
    return [
        SessionEvent(
            article_id=tx.article_id,
            timestamp=_to_datetime(tx.t_dat),
            price=tx.price,
            category_l1=item_meta[tx.article_id].category_l1 if tx.article_id in item_meta else "unknown",
        )
        for tx in history
    ]


def sample_negatives(all_article_ids: list[int], exclude: set[int], n: int, rng: np.random.RandomState) -> list[int]:
    negatives: list[int] = []
    attempts = 0
    max_attempts = n * 20
    while len(negatives) < n and attempts < max_attempts:
        candidate = all_article_ids[rng.randint(len(all_article_ids))]
        attempts += 1
        if candidate in exclude or candidate in negatives:
            continue
        negatives.append(candidate)
    return negatives


def make_candidate(
    article_id: int,
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
    item_meta: dict[int, ItemMeta],
    now: date,
) -> RankingCandidate | None:
    idx = article_id_to_index.get(article_id)
    meta = item_meta.get(article_id)
    if idx is None or meta is None:
        return None
    days_since_first_seen = max(0.0, float((now - meta.first_seen).days))
    return RankingCandidate(
        article_id=article_id,
        category_l1=meta.category_l1,
        price=meta.avg_price,
        item_vector=item_vectors[idx].tolist(),
        popularity=meta.popularity,
        ctr=meta.ctr,
        days_since_first_seen=days_since_first_seen,
    )


def weak_labels(
    is_target: bool, category_l1: str, session_categories: set[str], rng: np.random.RandomState
) -> dict[str, int]:
    if is_target:
        return {
            "click": 1,
            "purchase": 1,
            "cart": 1 if rng.random() < CART_PROB else 0,
            "wishlist": 1 if rng.random() < WISHLIST_PROB else 0,
        }
    return {"click": 1 if category_l1 in session_categories else 0, "purchase": 0, "cart": 0, "wishlist": 0}


def compute_user_vector(
    user_tower: UserTower, history_idxs: list[int], item_vectors: np.ndarray, demo: tuple[int, int]
) -> list[float]:
    if history_idxs:
        history_emb = torch.tensor(item_vectors[history_idxs], dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, len(history_idxs))
    else:
        history_emb = torch.zeros(1, 1, item_vectors.shape[1])
        mask = torch.zeros(1, 1)
    with torch.no_grad():
        user_emb = user_tower(history_emb, mask, torch.tensor([demo[0]]), torch.tensor([demo[1]]))
    return user_emb[0].tolist()


def infer_session_intents(
    session: list[SessionEvent],
    encoder: SessionEncoder | None,
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
) -> list[Intent]:
    if encoder is None:
        return [
            Intent(vector=[0.0] * EMBED_DIM, weight=1.0 / N_INTENTS, label=COLD_START_LABEL) for _ in range(N_INTENTS)
        ]
    return infer_intents(session, encoder=encoder, item_vectors=item_vectors, article_id_to_index=article_id_to_index)


def build_dataset(
    examples: list[RankerExample],
    item_meta: dict[int, ItemMeta],
    all_article_ids: list[int],
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
    user_tower: UserTower,
    demo_by_customer: dict[str, tuple[int, int]],
    session_encoder: SessionEncoder | None,
    n_negatives: int = N_NEGATIVES,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.RandomState(seed)
    feature_rows: list[np.ndarray] = []
    labels: dict[str, list[int]] = {obj: [] for obj in OBJECTIVES}

    for example in examples:
        history_article_ids = [tx.article_id for tx in example.history]
        idxs = [article_id_to_index[a] for a in history_article_ids if a in article_id_to_index]
        session = build_session(example.history, item_meta)
        session_categories = {e.category_l1 for e in session}

        user_vector = compute_user_vector(user_tower, idxs, item_vectors, demo_by_customer.get(example.customer_id, (0, 0)))
        intents = infer_session_intents(session, session_encoder, item_vectors, article_id_to_index)
        user = UserFeatures(
            user_id=example.customer_id,
            user_vector=user_vector,
            intent_vectors=[i.vector for i in intents],
            intent_weights=[i.weight for i in intents],
        )

        exclude = set(history_article_ids) | {example.target.article_id}
        negative_ids = sample_negatives(all_article_ids, exclude, n_negatives, rng)
        candidate_ids = [example.target.article_id] + negative_ids

        candidates: list[RankingCandidate] = []
        is_target_flags: list[bool] = []
        for article_id in candidate_ids:
            c = make_candidate(article_id, item_vectors, article_id_to_index, item_meta, now=example.target.t_dat)
            if c is not None:
                candidates.append(c)
                is_target_flags.append(article_id == example.target.article_id)

        if not candidates:
            continue

        feature_rows.append(build_features(user, candidates, session))
        for c, is_target in zip(candidates, is_target_flags):
            row_labels = weak_labels(is_target, c.category_l1, session_categories, rng)
            for obj in OBJECTIVES:
                labels[obj].append(row_labels[obj])

    X = np.concatenate(feature_rows, axis=0) if feature_rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = {obj: np.array(vals, dtype=np.int32) for obj, vals in labels.items()}
    return X, y


def train_boosters(
    X_train: np.ndarray, y_train: dict[str, np.ndarray], X_val: np.ndarray, y_val: dict[str, np.ndarray]
) -> dict[str, lgb.Booster]:
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.05,
    }
    boosters: dict[str, lgb.Booster] = {}
    for objective in OBJECTIVES:
        train_set = lgb.Dataset(X_train, label=y_train[objective], feature_name=list(FEATURE_NAMES))
        val_set = lgb.Dataset(X_val, label=y_val[objective], reference=train_set)
        boosters[objective] = lgb.train(
            params,
            train_set,
            num_boost_round=100,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(0)],
        )
    return boosters


def combined_score(boosters: dict[str, lgb.Booster], X: np.ndarray) -> np.ndarray:
    return sum(weight * boosters[objective].predict(X) for objective, weight in COMBINED_WEIGHTS.items())


def save_boosters(boosters: dict[str, lgb.Booster], artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for objective, booster in boosters.items():
        booster.save_model(str(artifacts_dir / f"ranker_{objective}.txt"))


def measure_scoring_latency_ms(boosters: dict[str, lgb.Booster], n_candidates: int = LATENCY_CANDIDATES) -> float:
    rng = np.random.RandomState(0)
    X = rng.rand(n_candidates, len(FEATURE_NAMES)).astype(np.float32)
    combined_score(boosters, X)
    start = time.perf_counter()
    combined_score(boosters, X)
    return (time.perf_counter() - start) * 1000


def make_two_tower_ranker_recommend_fn(
    user_tower: UserTower,
    demo_by_customer: dict[str, tuple[int, int]],
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
    index_to_article_id: dict[int, int],
    index,
    item_meta: dict[int, ItemMeta],
    session_encoder: SessionEncoder | None,
    boosters: dict[str, lgb.Booster],
    reference_date: date,
    session_length: int = SESSION_LENGTH,
    retrieval_k: int = RETRIEVAL_K,
    k: int = 20,
) -> RecommendFn:
    def recommend(user_id: str, history: list[int]) -> list[int]:
        recent = history[-session_length:]
        idxs = [article_id_to_index[a] for a in recent if a in article_id_to_index]
        if not idxs:
            return []

        user_vector = compute_user_vector(user_tower, idxs, item_vectors, demo_by_customer.get(user_id, (0, 0)))

        seen = set(history)
        fetch_k = min(index.ntotal, retrieval_k + len(seen))
        query = np.array(user_vector, dtype=np.float32).reshape(1, -1)
        _, indices = index.search(query, fetch_k)

        candidate_ids: list[int] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            article_id = index_to_article_id[int(idx)]
            if article_id in seen:
                continue
            candidate_ids.append(article_id)
            if len(candidate_ids) >= retrieval_k:
                break
        if not candidate_ids:
            return []

        session = [
            SessionEvent(
                article_id=a,
                timestamp=_to_datetime(reference_date),
                price=item_meta[a].avg_price if a in item_meta else 0.0,
                category_l1=item_meta[a].category_l1 if a in item_meta else "unknown",
            )
            for a in recent
        ]
        intents = infer_session_intents(session, session_encoder, item_vectors, article_id_to_index)
        user = UserFeatures(
            user_id=user_id,
            user_vector=user_vector,
            intent_vectors=[i.vector for i in intents],
            intent_weights=[i.weight for i in intents],
        )

        candidates = [
            c
            for c in (
                make_candidate(a, item_vectors, article_id_to_index, item_meta, now=reference_date)
                for a in candidate_ids
            )
            if c is not None
        ]
        if not candidates:
            return []

        X = build_features(user, candidates, session)
        scores = combined_score(boosters, X)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [c.article_id for c, _ in ranked[:k]]

    return recommend


def load_session_encoder(path: Path = SESSION_ENCODER_PATH) -> SessionEncoder | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = SessionEncoder(**checkpoint["config"])
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()
    return encoder


def main() -> None:
    import faiss

    train = pl.read_parquet(TRAIN_PATH)
    val = pl.read_parquet(VAL_PATH)
    items = pl.read_parquet(ITEMS_PATH)
    users = pl.read_parquet(USERS_PATH)

    item_vectors = np.load(ITEM_VECTORS_PATH)
    with ID_MAP_PATH.open() as f:
        id_map = json.load(f)
    article_id_to_index = {int(k): v for k, v in id_map["article_id_to_index"].items()}
    index_to_article_id = {int(k): v for k, v in id_map["index_to_article_id"].items()}
    all_article_ids = items["article_id"].to_list()

    checkpoint = torch.load(TOWERS_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    user_tower = UserTower(
        len(checkpoint["age_band_vocab"]), len(checkpoint["region_vocab"]), embed_dim=config["embedding_dim"]
    )
    user_tower.load_state_dict(checkpoint["user_tower_state"])
    user_tower.eval()
    age_band_vocab, region_vocab = checkpoint["age_band_vocab"], checkpoint["region_vocab"]
    demo_by_customer = {
        customer_id: (age_band_vocab.get(age_band, 0), region_vocab.get(region, 0))
        for customer_id, age_band, region in zip(
            users["customer_id"].to_list(),
            users["age_band"].fill_null("unknown").to_list(),
            users["region"].fill_null("unknown").to_list(),
        )
    }

    session_encoder = load_session_encoder()
    print(f"session encoder loaded: {session_encoder is not None}")

    item_meta = build_item_meta(items, train)
    histories = build_customer_histories(train)

    train_examples = subsample_examples(build_train_examples(histories), MAX_TRAIN_SESSIONS)
    val_examples = subsample_examples(build_val_examples(histories, val), MAX_VAL_SESSIONS)
    print(f"ranker train sessions: {len(train_examples)}, val sessions: {len(val_examples)}")

    X_train, y_train = build_dataset(
        train_examples, item_meta, all_article_ids, item_vectors, article_id_to_index, user_tower, demo_by_customer, session_encoder
    )
    X_val, y_val = build_dataset(
        val_examples, item_meta, all_article_ids, item_vectors, article_id_to_index, user_tower, demo_by_customer, session_encoder
    )
    print(f"train rows: {X_train.shape[0]}, val rows: {X_val.shape[0]}, features: {X_train.shape[1]}")

    boosters = train_boosters(X_train, y_train, X_val, y_val)
    save_boosters(boosters)

    latency_ms = measure_scoring_latency_ms(boosters)
    print(f"scoring {LATENCY_CANDIDATES} candidates: {latency_ms:.3f}ms")
    assert latency_ms < LATENCY_BUDGET_MS, f"scoring latency {latency_ms:.3f}ms exceeds budget {LATENCY_BUDGET_MS}ms"

    index = faiss.read_index(str(INDEX_PATH))
    reference_date = train["t_dat"].max()
    recommend_fn = make_two_tower_ranker_recommend_fn(
        user_tower,
        demo_by_customer,
        item_vectors,
        article_id_to_index,
        index_to_article_id,
        index,
        item_meta,
        session_encoder,
        boosters,
        reference_date,
    )
    metrics = evaluate(recommend_fn, val, k=20)
    eval_histories = list(histories.values())
    eval_history_ids = [[tx.article_id for tx in h] for h in eval_histories[:200]]
    latency_p50_ms = measure_latency_p50_ms(recommend_fn, eval_history_ids)

    print(
        f"Two-tower + ranker | recall@20={metrics['recall@20']:.4f} ndcg@20={metrics['ndcg@20']:.4f} "
        f"coverage={metrics['catalog_coverage']:.4f} latency_p50={latency_p50_ms:.2f}ms"
    )
    write_results_row("Two-tower + ranker", metrics, latency_p50_ms, path=RESULTS_PATH)


if __name__ == "__main__":
    main()
