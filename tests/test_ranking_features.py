from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.ranking.features import (
    FEATURE_NAMES,
    NEW_ITEM_DAYS_THRESHOLD,
    NO_RECENT_EVENT_MINUTES,
    RankingCandidate,
    UserFeatures,
    build_features,
)
from src.session.intents import SessionEvent

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0)


def _user(user_vector: list[float] | None = None, intent_vectors: list[list[float]] | None = None, intent_weights: list[float] | None = None) -> UserFeatures:
    return UserFeatures(
        user_id="u1",
        user_vector=user_vector if user_vector is not None else [1.0, 0.0],
        intent_vectors=intent_vectors if intent_vectors is not None else [[1.0, 0.0], [0.0, 1.0]],
        intent_weights=intent_weights if intent_weights is not None else [0.5, 0.5],
    )


def _candidate(
    article_id: int = 1,
    category_l1: str = "shoes",
    price: float = 10.0,
    item_vector: list[float] | None = None,
    popularity: float = 0.5,
    ctr: float = 0.1,
    days_since_first_seen: float = 100.0,
) -> RankingCandidate:
    return RankingCandidate(
        article_id=article_id,
        category_l1=category_l1,
        price=price,
        item_vector=item_vector if item_vector is not None else [1.0, 0.0],
        popularity=popularity,
        ctr=ctr,
        days_since_first_seen=days_since_first_seen,
    )


def test_build_features_empty_candidates_returns_empty_array() -> None:
    result = build_features(_user(), [], [])
    assert result.shape == (0, len(FEATURE_NAMES))


def test_output_shape_matches_candidate_count_and_feature_count() -> None:
    candidates = [_candidate(article_id=1), _candidate(article_id=2)]
    result = build_features(_user(), candidates, [])
    assert result.shape == (2, len(FEATURE_NAMES))


def test_two_tower_dot_matches_dot_product() -> None:
    user = _user(user_vector=[1.0, 2.0])
    candidate = _candidate(item_vector=[3.0, 4.0])
    result = build_features(user, [candidate], [])
    assert result[0, IDX["two_tower_dot"]] == pytest.approx(1.0 * 3.0 + 2.0 * 4.0)


def test_intent_dot_is_weighted_sum_across_heads() -> None:
    user = _user(intent_vectors=[[1.0, 0.0], [0.0, 1.0]], intent_weights=[0.3, 0.7])
    candidate = _candidate(item_vector=[2.0, 5.0])
    result = build_features(user, [candidate], [])
    expected = 0.3 * (1.0 * 2.0 + 0.0 * 5.0) + 0.7 * (0.0 * 2.0 + 1.0 * 5.0)
    assert result[0, IDX["intent_dot"]] == pytest.approx(expected)


def test_popularity_and_ctr_pass_through_unchanged() -> None:
    candidate = _candidate(popularity=0.42, ctr=0.07)
    result = build_features(_user(), [candidate], [])
    assert result[0, IDX["item_popularity"]] == pytest.approx(0.42)
    assert result[0, IDX["item_ctr"]] == pytest.approx(0.07)


def test_price_percentile_ranks_cheapest_item_lowest_within_category() -> None:
    candidates = [
        _candidate(article_id=1, category_l1="shoes", price=10.0),
        _candidate(article_id=2, category_l1="shoes", price=20.0),
        _candidate(article_id=3, category_l1="shoes", price=30.0),
    ]
    result = build_features(_user(), candidates, [])
    assert result[0, IDX["price_percentile_in_category"]] == pytest.approx(0.0)
    assert result[1, IDX["price_percentile_in_category"]] == pytest.approx(0.5)
    assert result[2, IDX["price_percentile_in_category"]] == pytest.approx(1.0)


def test_price_percentile_is_scoped_per_category() -> None:
    candidates = [
        _candidate(article_id=1, category_l1="shoes", price=1000.0),
        _candidate(article_id=2, category_l1="bags", price=1.0),
        _candidate(article_id=3, category_l1="bags", price=2.0),
    ]
    result = build_features(_user(), candidates, [])
    assert result[1, IDX["price_percentile_in_category"]] == pytest.approx(0.0)
    assert result[2, IDX["price_percentile_in_category"]] == pytest.approx(1.0)


def test_price_percentile_single_item_category_is_midpoint() -> None:
    candidate = _candidate(article_id=1, category_l1="hats", price=50.0)
    result = build_features(_user(), [candidate], [])
    assert result[0, IDX["price_percentile_in_category"]] == pytest.approx(0.5)


def test_category_match_count_counts_session_events_in_same_category() -> None:
    session = [
        SessionEvent(article_id=1, timestamp=_now(), price=10.0, category_l1="shoes"),
        SessionEvent(article_id=2, timestamp=_now(), price=10.0, category_l1="shoes"),
        SessionEvent(article_id=3, timestamp=_now(), price=10.0, category_l1="bags"),
    ]
    candidate = _candidate(category_l1="shoes")
    result = build_features(_user(), [candidate], session)
    assert result[0, IDX["category_match_count"]] == pytest.approx(2.0)


def test_recency_is_sentinel_when_category_never_seen_in_session() -> None:
    session = [SessionEvent(article_id=1, timestamp=_now(), price=10.0, category_l1="bags")]
    candidate = _candidate(category_l1="shoes")
    result = build_features(_user(), [candidate], session)
    assert result[0, IDX["recency_same_category_minutes"]] == pytest.approx(NO_RECENT_EVENT_MINUTES)


def test_recency_is_sentinel_for_empty_session() -> None:
    result = build_features(_user(), [_candidate()], [])
    assert result[0, IDX["recency_same_category_minutes"]] == pytest.approx(NO_RECENT_EVENT_MINUTES)


def test_recency_measures_minutes_since_most_recent_same_category_event() -> None:
    session = [
        SessionEvent(article_id=1, timestamp=_now() - timedelta(minutes=30), price=10.0, category_l1="shoes"),
        SessionEvent(article_id=2, timestamp=_now() - timedelta(minutes=5), price=10.0, category_l1="shoes"),
        SessionEvent(article_id=3, timestamp=_now(), price=10.0, category_l1="bags"),
    ]
    candidate = _candidate(category_l1="shoes")
    result = build_features(_user(), [candidate], session)
    assert result[0, IDX["recency_same_category_minutes"]] == pytest.approx(5.0)


def test_session_length_reflects_event_count() -> None:
    session = [SessionEvent(article_id=i, timestamp=_now(), price=10.0, category_l1="shoes") for i in range(4)]
    result = build_features(_user(), [_candidate()], session)
    assert result[0, IDX["session_length"]] == pytest.approx(4.0)


def test_is_new_item_true_below_threshold() -> None:
    candidate = _candidate(days_since_first_seen=NEW_ITEM_DAYS_THRESHOLD - 1.0)
    result = build_features(_user(), [candidate], [])
    assert result[0, IDX["is_new_item"]] == pytest.approx(1.0)


def test_is_new_item_false_at_or_above_threshold() -> None:
    candidate = _candidate(days_since_first_seen=NEW_ITEM_DAYS_THRESHOLD)
    result = build_features(_user(), [candidate], [])
    assert result[0, IDX["is_new_item"]] == pytest.approx(0.0)


def test_days_since_first_seen_passes_through() -> None:
    candidate = _candidate(days_since_first_seen=365.0)
    result = build_features(_user(), [candidate], [])
    assert result[0, IDX["days_since_first_seen"]] == pytest.approx(365.0)


def test_output_dtype_is_float32() -> None:
    result = build_features(_user(), [_candidate()], [])
    assert result.dtype == np.float32
