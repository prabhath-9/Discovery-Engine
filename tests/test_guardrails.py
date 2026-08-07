from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.guardrails.interface import (
    GuardrailError,
    ScoredItem,
    category_shares,
    enforce_diversity_cap,
    violates_diversity_cap,
)


def _items(category_counts: dict[str, int], score_start: float = 0.0) -> list[ScoredItem]:
    items = []
    item_id = 0
    for category, count in category_counts.items():
        for rank in range(count):
            item_id += 1
            items.append(ScoredItem(item_id=item_id, category_l1=category, score=score_start - rank))
        score_start -= 1000
    return items


def test_selects_top_scored_items_within_each_category_when_abundant() -> None:
    candidates = _items({"shoes": 50, "bags": 50, "hats": 50}, score_start=10_000)
    result = enforce_diversity_cap(candidates, list_size=21, cap=0.35)

    assert len(result) == 21
    shares = category_shares(result)
    assert shares["shoes"] == pytest.approx(7 / 21)
    assert shares["bags"] == pytest.approx(7 / 21)
    assert shares["hats"] == pytest.approx(7 / 21)
    assert not violates_diversity_cap(result, cap=0.35)

    shoes_ids = {item.item_id for item in result if item.category_l1 == "shoes"}
    assert shoes_ids == {1, 2, 3, 4, 5, 6, 7}  # top 7 by score, not the low-ranked ones


def test_shrinks_and_backfills_when_one_category_dominates_the_pool() -> None:
    candidates = _items({"shoes": 100, "bags": 5, "hats": 5})
    result = enforce_diversity_cap(candidates, list_size=20, cap=0.35)

    assert len(result) == 15
    counts = {category: round(share * len(result)) for category, share in category_shares(result).items()}
    assert counts == {"shoes": 5, "bags": 5, "hats": 5}


def test_returns_empty_for_single_category_pool() -> None:
    candidates = _items({"shoes": 100})
    assert enforce_diversity_cap(candidates, list_size=20, cap=0.35) == []


def test_returns_empty_for_two_category_pool() -> None:
    # even split of 2 categories is 50% each, which always exceeds a 35% cap
    candidates = _items({"shoes": 50, "bags": 50})
    assert enforce_diversity_cap(candidates, list_size=20, cap=0.35) == []


def test_noop_when_cap_allows_everything() -> None:
    candidates = _items({"shoes": 5, "bags": 5})
    result = enforce_diversity_cap(candidates, list_size=len(candidates), cap=1.0)
    assert result == candidates


def test_empty_inputs() -> None:
    assert enforce_diversity_cap([], list_size=10) == []
    assert enforce_diversity_cap(_items({"shoes": 5}), list_size=0) == []


@pytest.mark.parametrize("cap", [0, -0.1, 1.1])
def test_rejects_invalid_cap(cap: float) -> None:
    with pytest.raises(GuardrailError):
        enforce_diversity_cap(_items({"shoes": 5}), list_size=5, cap=cap)


def test_rejects_negative_list_size() -> None:
    with pytest.raises(GuardrailError):
        enforce_diversity_cap(_items({"shoes": 5}), list_size=-1)


def test_rejects_missing_category() -> None:
    bad_items = [ScoredItem(item_id=1, category_l1="", score=1.0)]
    with pytest.raises(GuardrailError):
        enforce_diversity_cap(bad_items, list_size=1)


def test_category_shares_of_empty_list_is_empty_dict() -> None:
    assert category_shares([]) == {}


def test_category_shares_sums_to_one() -> None:
    shares = category_shares(_items({"shoes": 3, "bags": 1}))
    assert math.isclose(sum(shares.values()), 1.0)


def test_violates_diversity_cap_detects_skew() -> None:
    assert violates_diversity_cap(_items({"shoes": 8, "bags": 2}), cap=0.35) is True
    # a 2-way split is always >= 50% for the majority category, so use 3 balanced categories
    assert violates_diversity_cap(_items({"shoes": 3, "bags": 3, "hats": 3}), cap=0.35) is False


@given(
    counts=st.dictionaries(
        keys=st.sampled_from(["a", "b", "c", "d", "e"]),
        values=st.integers(min_value=0, max_value=50),
        min_size=1,
        max_size=5,
    ),
    list_size=st.integers(min_value=1, max_value=30),
)
def test_property_never_violates_cap_regardless_of_pool_shape(counts: dict[str, int], list_size: int) -> None:
    candidates = _items(counts)
    result = enforce_diversity_cap(candidates, list_size=list_size, cap=0.35)
    assert len(result) <= list_size
    assert not violates_diversity_cap(result, cap=0.35)
