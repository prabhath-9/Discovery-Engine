from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import torch

from src.ranking.features import FEATURE_NAMES
from src.session.encoder import EMBED_DIM, N_INTENTS, SessionEncoder
from src.session.intents import SessionEvent
from trainer.build_index import build_index
from trainer.train_ranker import (
    OBJECTIVES,
    COMBINED_WEIGHTS,
    CTR_SMOOTHING,
    ItemMeta,
    RankerExample,
    Transaction,
    build_customer_histories,
    build_dataset,
    build_item_meta,
    build_session,
    build_train_examples,
    build_val_examples,
    combined_score,
    compute_user_vector,
    infer_session_intents,
    load_session_encoder,
    make_candidate,
    make_two_tower_ranker_recommend_fn,
    measure_scoring_latency_ms,
    sample_negatives,
    save_boosters,
    subsample_examples,
    train_boosters,
    weak_labels,
)
from trainer.train_towers import UserTower


# --- build_customer_histories ---


def test_build_customer_histories_sorts_by_date_per_customer() -> None:
    train = pl.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2"],
            "article_id": [20, 10, 30],
            "t_dat": [date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 1)],
            "price": [5.0, 4.0, 6.0],
        }
    )
    histories = build_customer_histories(train)
    assert [tx.article_id for tx in histories["u1"]] == [10, 20]
    assert histories["u2"][0].article_id == 30


# --- build_train_examples / build_val_examples ---


def test_build_train_examples_expands_history() -> None:
    tx = [Transaction(10, date(2024, 1, 1), 1.0), Transaction(20, date(2024, 1, 2), 2.0), Transaction(30, date(2024, 1, 3), 3.0)]
    examples = build_train_examples({"u1": tx}, session_length=20)
    assert len(examples) == 2
    assert examples[0].history == [tx[0]]
    assert examples[0].target == tx[1]
    assert examples[1].history == [tx[0], tx[1]]
    assert examples[1].target == tx[2]


def test_build_train_examples_caps_history_length() -> None:
    tx = [Transaction(i, date(2024, 1, 1), 1.0) for i in range(5)]
    examples = build_train_examples({"u1": tx}, session_length=2)
    assert examples[-1].history == tx[2:4]
    assert examples[-1].target == tx[4]


def test_build_val_examples_skips_customers_without_train_history() -> None:
    val = pl.DataFrame({"customer_id": ["cold"], "article_id": [1], "t_dat": [date(2024, 1, 1)], "price": [1.0]})
    assert build_val_examples({}, val) == []


def test_build_val_examples_uses_tail_of_train_history() -> None:
    tx = [Transaction(i, date(2024, 1, 1), 1.0) for i in range(5)]
    val = pl.DataFrame({"customer_id": ["u1"], "article_id": [99], "t_dat": [date(2024, 1, 10)], "price": [9.0]})
    examples = build_val_examples({"u1": tx}, val, session_length=2)
    assert examples[0].history == tx[-2:]
    assert examples[0].target.article_id == 99


# --- subsample_examples ---


def test_subsample_examples_returns_all_when_under_cap() -> None:
    examples = [RankerExample("u", [], Transaction(i, date(2024, 1, 1), 1.0)) for i in range(3)]
    assert subsample_examples(examples, max_examples=10) == examples


def test_subsample_examples_caps_size_deterministically() -> None:
    examples = [RankerExample("u", [], Transaction(i, date(2024, 1, 1), 1.0)) for i in range(100)]
    a = subsample_examples(examples, max_examples=10, seed=1)
    b = subsample_examples(examples, max_examples=10, seed=1)
    assert len(a) == 10
    assert a == b


# --- build_item_meta ---


def test_build_item_meta_computes_expected_stats() -> None:
    items = pl.DataFrame({"article_id": [1, 2], "category_l1": ["A", "B"], "n_interactions": [10, 40]})
    train = pl.DataFrame(
        {
            "article_id": [1, 1, 2],
            "t_dat": [date(2024, 1, 5), date(2024, 1, 1), date(2024, 2, 1)],
            "price": [10.0, 20.0, 5.0],
        }
    )
    meta = build_item_meta(items, train)
    assert meta[1].category_l1 == "A"
    assert meta[1].popularity == pytest.approx(10 / 40)
    assert meta[1].ctr == pytest.approx(10 / (10 + CTR_SMOOTHING))
    assert meta[1].first_seen == date(2024, 1, 1)
    assert meta[1].avg_price == pytest.approx(15.0)
    assert meta[2].popularity == pytest.approx(1.0)


# --- build_session ---


def test_build_session_maps_category_from_item_meta() -> None:
    item_meta = {10: ItemMeta("shoes", 0.1, 0.1, date(2024, 1, 1), 9.0)}
    history = [Transaction(10, date(2024, 1, 2), 9.5)]
    session = build_session(history, item_meta)
    assert session[0].category_l1 == "shoes"
    assert session[0].article_id == 10
    assert session[0].price == 9.5
    assert session[0].timestamp == datetime(2024, 1, 2, 0, 0, 0)


# --- sample_negatives ---


def test_sample_negatives_excludes_given_ids_and_returns_unique() -> None:
    rng = np.random.RandomState(0)
    negatives = sample_negatives([1, 2, 3, 4, 5], exclude={1, 2}, n=3, rng=rng)
    assert len(negatives) == 3
    assert set(negatives).isdisjoint({1, 2})
    assert len(set(negatives)) == 3


def test_sample_negatives_stops_when_pool_exhausted() -> None:
    rng = np.random.RandomState(0)
    negatives = sample_negatives([1, 2], exclude=set(), n=5, rng=rng)
    assert len(negatives) <= 2
    assert set(negatives) <= {1, 2}


# --- make_candidate ---


def test_make_candidate_clips_negative_days_since_first_seen_to_zero() -> None:
    item_vectors = np.array([[1.0, 2.0]], dtype=np.float32)
    article_id_to_index = {10: 0}
    item_meta = {10: ItemMeta(category_l1="A", popularity=0.5, ctr=0.1, first_seen=date(2024, 6, 1), avg_price=9.0)}
    candidate = make_candidate(10, item_vectors, article_id_to_index, item_meta, now=date(2024, 1, 1))
    assert candidate is not None
    assert candidate.days_since_first_seen == 0.0


def test_make_candidate_returns_none_for_unknown_article() -> None:
    assert make_candidate(999, np.zeros((1, 2)), {}, {}, now=date(2024, 1, 1)) is None


# --- weak_labels ---


def test_weak_labels_target_is_always_purchase_and_click() -> None:
    rng = np.random.RandomState(0)
    labels = weak_labels(is_target=True, category_l1="shoes", session_categories=set(), rng=rng)
    assert labels["purchase"] == 1
    assert labels["click"] == 1


def test_weak_labels_negative_click_true_when_category_seen_in_session() -> None:
    rng = np.random.RandomState(0)
    labels = weak_labels(is_target=False, category_l1="shoes", session_categories={"shoes"}, rng=rng)
    assert labels == {"click": 1, "purchase": 0, "cart": 0, "wishlist": 0}


def test_weak_labels_negative_click_false_when_category_unseen_in_session() -> None:
    rng = np.random.RandomState(0)
    labels = weak_labels(is_target=False, category_l1="bags", session_categories={"shoes"}, rng=rng)
    assert labels["click"] == 0


class _StubRng:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


def test_weak_labels_cart_and_wishlist_follow_probability_draw() -> None:
    hits = weak_labels(is_target=True, category_l1="shoes", session_categories=set(), rng=_StubRng([0.1, 0.1]))
    misses = weak_labels(is_target=True, category_l1="shoes", session_categories=set(), rng=_StubRng([0.9, 0.9]))
    assert hits["cart"] == 1 and hits["wishlist"] == 1
    assert misses["cart"] == 0 and misses["wishlist"] == 0


# --- compute_user_vector ---


def test_compute_user_vector_matches_direct_tower_forward() -> None:
    torch.manual_seed(0)
    tower = UserTower(n_age_band=1, n_region=1, embed_dim=8)
    tower.eval()
    item_vectors = np.random.RandomState(0).rand(4, 8).astype(np.float32)

    result = compute_user_vector(tower, [0, 2], item_vectors, (0, 0))

    history_emb = torch.tensor(item_vectors[[0, 2]]).unsqueeze(0)
    mask = torch.ones(1, 2)
    with torch.no_grad():
        expected = tower(history_emb, mask, torch.tensor([0]), torch.tensor([0]))
    assert result == pytest.approx(expected[0].tolist())


def test_compute_user_vector_handles_empty_history() -> None:
    tower = UserTower(n_age_band=1, n_region=1, embed_dim=8)
    tower.eval()
    item_vectors = np.zeros((1, 8), dtype=np.float32)
    result = compute_user_vector(tower, [], item_vectors, (0, 0))
    assert len(result) == 8


# --- infer_session_intents ---


def test_infer_session_intents_cold_start_when_no_encoder() -> None:
    intents = infer_session_intents([], None, np.zeros((1, EMBED_DIM)), {})
    assert len(intents) == N_INTENTS
    assert all(i.vector == [0.0] * EMBED_DIM for i in intents)
    assert sum(i.weight for i in intents) == pytest.approx(1.0)


def test_infer_session_intents_uses_real_encoder_when_provided() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder(embed_dim=8, n_heads=2, n_layers=2, n_intents=4)
    encoder.eval()
    rng = np.random.RandomState(0)
    item_vectors = rng.rand(5, 8).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)
    article_id_to_index = {100 + i: i for i in range(5)}
    events = [SessionEvent(article_id=100, timestamp=datetime(2026, 1, 1), price=10.0, category_l1="shoes")]

    intents = infer_session_intents(events, encoder, item_vectors, article_id_to_index)
    assert len(intents) == 4


# --- build_dataset ---


def test_build_dataset_produces_one_purchase_row_per_example() -> None:
    item_vectors = np.random.RandomState(0).rand(6, 4).astype(np.float32)
    article_id_to_index = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5}
    item_meta = {
        aid: ItemMeta(category_l1="A" if aid <= 30 else "B", popularity=0.5, ctr=0.1, first_seen=date(2024, 1, 1), avg_price=10.0)
        for aid in article_id_to_index
    }
    all_article_ids = list(article_id_to_index)
    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=4)
    user_tower.eval()
    demo_by_customer = {"u1": (0, 0)}

    examples = [
        RankerExample("u1", [Transaction(10, date(2024, 1, 1), 9.0)], Transaction(20, date(2024, 1, 2), 11.0))
    ]

    X, y = build_dataset(
        examples, item_meta, all_article_ids, item_vectors, article_id_to_index, user_tower, demo_by_customer,
        session_encoder=None, n_negatives=2, seed=0,
    )

    assert X.shape == (3, len(FEATURE_NAMES))
    assert y["purchase"].tolist() == [1, 0, 0]
    assert y["click"][0] == 1
    assert y["cart"][1:].tolist() == [0, 0]
    assert y["wishlist"][1:].tolist() == [0, 0]


# --- LightGBM training helpers ---


def _tiny_boosters(seed: int = 0) -> dict[str, lgb.Booster]:
    rng = np.random.RandomState(seed)
    n = 80
    X = rng.rand(n, len(FEATURE_NAMES)).astype(np.float32)
    y = {obj: rng.randint(0, 2, size=n).astype(np.int32) for obj in OBJECTIVES}
    return train_boosters(X, y, X, y)


def test_train_boosters_returns_one_booster_per_objective() -> None:
    boosters = _tiny_boosters()
    assert set(boosters) == set(OBJECTIVES)
    assert all(isinstance(b, lgb.Booster) for b in boosters.values())


def test_train_boosters_predictions_are_probabilities() -> None:
    boosters = _tiny_boosters()
    X = np.random.RandomState(1).rand(10, len(FEATURE_NAMES)).astype(np.float32)
    for booster in boosters.values():
        preds = booster.predict(X)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_combined_score_matches_weighted_sum_of_boosters() -> None:
    boosters = _tiny_boosters()
    X = np.random.RandomState(2).rand(5, len(FEATURE_NAMES)).astype(np.float32)
    expected = sum(weight * boosters[obj].predict(X) for obj, weight in COMBINED_WEIGHTS.items())
    assert np.allclose(combined_score(boosters, X), expected)


def test_save_boosters_writes_loadable_files(tmp_path: Path) -> None:
    boosters = _tiny_boosters()
    save_boosters(boosters, artifacts_dir=tmp_path)
    X = np.random.RandomState(3).rand(4, len(FEATURE_NAMES)).astype(np.float32)
    for objective in OBJECTIVES:
        path = tmp_path / f"ranker_{objective}.txt"
        assert path.exists()
        reloaded = lgb.Booster(model_file=str(path))
        assert np.allclose(reloaded.predict(X), boosters[objective].predict(X))


def test_measure_scoring_latency_ms_is_nonnegative() -> None:
    boosters = _tiny_boosters()
    latency = measure_scoring_latency_ms(boosters, n_candidates=50)
    assert latency >= 0.0


# --- make_two_tower_ranker_recommend_fn ---


def test_recommend_fn_excludes_history_and_respects_k() -> None:
    embed_dim, n_items = 8, 20
    rng = np.random.RandomState(0)
    item_vectors = rng.rand(n_items, embed_dim).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)

    index = build_index(item_vectors)
    article_id_to_index = {100 + i: i for i in range(n_items)}
    index_to_article_id = {i: 100 + i for i in range(n_items)}
    item_meta = {
        100 + i: ItemMeta(category_l1="A" if i % 2 == 0 else "B", popularity=0.5, ctr=0.1, first_seen=date(2024, 1, 1), avg_price=10.0)
        for i in range(n_items)
    }

    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=embed_dim)
    user_tower.eval()
    demo_by_customer = {"u1": (0, 0)}
    boosters = _tiny_boosters()

    recommend_fn = make_two_tower_ranker_recommend_fn(
        user_tower, demo_by_customer, item_vectors, article_id_to_index, index_to_article_id, index,
        item_meta, None, boosters, reference_date=date(2024, 6, 1), retrieval_k=10, k=5,
    )

    history = [100, 101, 102]
    result = recommend_fn("u1", history)

    assert len(result) == 5
    assert set(result).isdisjoint(history)


def test_recommend_fn_returns_empty_for_unmapped_history() -> None:
    embed_dim, n_items = 8, 10
    item_vectors = np.random.RandomState(1).rand(n_items, embed_dim).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)
    index = build_index(item_vectors)
    article_id_to_index = {100 + i: i for i in range(n_items)}
    index_to_article_id = {i: 100 + i for i in range(n_items)}
    item_meta = {
        100 + i: ItemMeta(category_l1="A", popularity=0.5, ctr=0.1, first_seen=date(2024, 1, 1), avg_price=10.0)
        for i in range(n_items)
    }
    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=embed_dim)
    user_tower.eval()
    boosters = _tiny_boosters()

    recommend_fn = make_two_tower_ranker_recommend_fn(
        user_tower, {}, item_vectors, article_id_to_index, index_to_article_id, index,
        item_meta, None, boosters, reference_date=date(2024, 6, 1), retrieval_k=5, k=3,
    )

    assert recommend_fn("cold_start_user", [999999]) == []


# --- load_session_encoder ---


def test_load_session_encoder_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_session_encoder(tmp_path / "missing.pt") is None


def test_load_session_encoder_loads_checkpoint(tmp_path: Path) -> None:
    encoder = SessionEncoder(embed_dim=8, n_heads=2, n_layers=2, session_length=5, n_intents=4)
    path = tmp_path / "session_encoder.pt"
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "config": {"embed_dim": 8, "n_heads": 2, "n_layers": 2, "session_length": 5, "n_intents": 4, "dropout": 0.1},
        },
        path,
    )
    loaded = load_session_encoder(path)
    assert loaded is not None
    assert loaded.training is False
