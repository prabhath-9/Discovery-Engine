from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.guardrails.pipeline import GuardrailReport, apply_all
from src.guardrails.rules import (
    cap_category,
    filter_availability,
    filter_policy,
    filter_seen,
    filter_sensitive,
    mmr_diversify,
)
from src.guardrails.types import Candidate


def _candidate(
    product_id: str,
    score: float = 1.0,
    category_l1: str = "shoes",
    in_stock: bool = True,
    age_restricted: bool = False,
    embedding: list[float] | None = None,
) -> Candidate:
    return Candidate(
        product_id=product_id,
        score=score,
        category_l1=category_l1,
        in_stock=in_stock,
        age_restricted=age_restricted,
        embedding=embedding if embedding is not None else [1.0, 0.0],
    )


def _candidates(category_counts: dict[str, int], score_start: float = 1000.0) -> list[Candidate]:
    candidates = []
    idx = 0
    for category, count in category_counts.items():
        for rank in range(count):
            idx += 1
            candidates.append(
                _candidate(product_id=str(idx), score=score_start - rank, category_l1=category)
            )
        score_start -= 100_000
    return candidates


# --- filter_availability ---


def test_filter_availability_drops_out_of_stock() -> None:
    candidates = [_candidate("1", in_stock=True), _candidate("2", in_stock=False)]
    assert [c.product_id for c in filter_availability(candidates)] == ["1"]


# --- filter_policy ---


@pytest.mark.parametrize("age_band", ["18_25", "26_35", "36_50", "over_50"])
def test_age_restricted_item_survives_for_adult_age_bands(age_band: str) -> None:
    candidates = [_candidate("1", age_restricted=True)]
    assert filter_policy(candidates, age_band) == candidates


def test_age_restricted_item_never_survives_for_under_18() -> None:
    candidates = [_candidate("1", age_restricted=True), _candidate("2", age_restricted=False)]
    result = filter_policy(candidates, "under_18")
    assert [c.product_id for c in result] == ["2"]


# --- filter_sensitive ---


def test_filter_sensitive_drops_blocked_categories() -> None:
    candidates = [_candidate("1", category_l1="health"), _candidate("2", category_l1="shoes")]
    result = filter_sensitive(candidates, blocked_categories={"health"})
    assert [c.product_id for c in result] == ["2"]


# --- filter_seen ---


def test_filter_seen_drops_impressed_items() -> None:
    candidates = [_candidate("1"), _candidate("2")]
    result = filter_seen(candidates, seen_ids={"1"})
    assert [c.product_id for c in result] == ["2"]


# --- cap_category ---


def test_cap_category_shrinks_when_one_category_dominates() -> None:
    candidates = _candidates({"shoes": 100, "bags": 5, "hats": 5})
    result = cap_category(candidates, limit=20, max_share=0.35)
    counts: dict[str, int] = {}
    for c in result:
        counts[c.category_l1] = counts.get(c.category_l1, 0) + 1
    assert counts == {"shoes": 5, "bags": 5, "hats": 5}


def test_cap_category_returns_single_item_for_single_category_pool() -> None:
    candidates = _candidates({"shoes": 100})
    result = cap_category(candidates, limit=20, max_share=0.35)
    assert len(result) == 1


def test_cap_category_fills_limit_when_pool_is_diverse_enough() -> None:
    candidates = _candidates({"shoes": 50, "bags": 50, "hats": 50})
    result = cap_category(candidates, limit=21, max_share=0.35)
    assert len(result) == 21


def test_cap_category_prefers_higher_scored_items_within_a_category() -> None:
    candidates = _candidates({"shoes": 50, "bags": 50, "hats": 50}, score_start=10_000)
    result = cap_category(candidates, limit=21, max_share=0.35)
    shoes_ids = {c.product_id for c in result if c.category_l1 == "shoes"}
    assert shoes_ids == {"1", "2", "3", "4", "5", "6", "7"}


def test_cap_category_empty_input() -> None:
    assert cap_category([], limit=10) == []


def test_cap_category_zero_limit() -> None:
    assert cap_category(_candidates({"shoes": 5}), limit=0) == []


# --- mmr_diversify ---


def test_mmr_diversify_respects_limit() -> None:
    candidates = [_candidate(str(i), score=float(i), embedding=[float(i), 1.0]) for i in range(10)]
    result = mmr_diversify(candidates, limit=4)
    assert len(result) == 4


def test_mmr_diversify_penalizes_near_duplicate_embeddings() -> None:
    # Two near-identical top-scored items in the same direction, one distinct.
    candidates = [
        _candidate("a", score=10.0, embedding=[1.0, 0.0]),
        _candidate("b", score=9.0, embedding=[1.0, 0.001]),
        _candidate("c", score=8.9, embedding=[0.0, 1.0]),
    ]
    result = mmr_diversify(candidates, limit=2, lam=0.5)
    ids = [c.product_id for c in result]
    assert ids[0] == "a"
    assert "c" in ids  # the distinct item beats the near-duplicate under redundancy penalty


# --- pipeline ---


def test_apply_all_runs_filters_in_order_and_reports_drops() -> None:
    candidates = [
        _candidate("1", in_stock=False),
        _candidate("2", age_restricted=True),
        _candidate("3", category_l1="health"),
        _candidate("4"),
        _candidate("5"),
    ]
    result, report = apply_all(
        candidates,
        age_band="under_18",
        blocked_categories={"health"},
        seen_ids={"5"},
        limit=10,
    )
    assert [c.product_id for c in result] == ["4"]
    assert isinstance(report, GuardrailReport)
    assert report.dropped["filter_availability"] == 1
    assert report.dropped["filter_policy"] == 1
    assert report.dropped["filter_sensitive"] == 1
    assert report.dropped["filter_seen"] == 1


def test_guardrail_report_counts_match_actual_drop_count() -> None:
    candidates = _candidates({"shoes": 20, "bags": 2})
    result, report = apply_all(
        candidates,
        age_band="26_35",
        blocked_categories=set(),
        seen_ids=set(),
        limit=10,
    )
    assert sum(report.dropped.values()) == len(candidates) - len(result)


# --- hypothesis properties ---

_category_st = st.sampled_from(["a", "b", "c", "d", "e"])
_candidates_st = st.lists(
    st.builds(
        _candidate,
        product_id=st.uuids().map(str),
        score=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        category_l1=_category_st,
        in_stock=st.just(True),
        age_restricted=st.just(False),
    ),
    min_size=0,
    max_size=60,
    unique_by=lambda c: c.product_id,
)


@given(candidates=_candidates_st, limit=st.integers(min_value=0, max_value=40))
def test_cap_category_never_exceeds_share_except_single_item_edge_case(
    candidates: list[Candidate], limit: int
) -> None:
    result = cap_category(candidates, limit=limit, max_share=0.35)
    if len(result) > 1:
        counts: dict[str, int] = {}
        for c in result:
            counts[c.category_l1] = counts.get(c.category_l1, 0) + 1
        for count in counts.values():
            assert count / len(result) <= 0.35


@given(candidates=_candidates_st, limit=st.integers(min_value=0, max_value=40))
def test_cap_category_output_length_never_exceeds_limit(candidates: list[Candidate], limit: int) -> None:
    result = cap_category(candidates, limit=limit, max_share=0.35)
    assert len(result) <= limit


@given(candidates=_candidates_st, limit=st.integers(min_value=0, max_value=40))
def test_cap_category_output_is_subset_with_no_duplicates(candidates: list[Candidate], limit: int) -> None:
    result = cap_category(candidates, limit=limit, max_share=0.35)
    input_ids = {c.product_id for c in candidates}
    result_ids = [c.product_id for c in result]
    assert len(result_ids) == len(set(result_ids))
    assert set(result_ids) <= input_ids


@given(candidates=_candidates_st, limit=st.integers(min_value=0, max_value=40))
def test_mmr_diversify_output_length_never_exceeds_limit(candidates: list[Candidate], limit: int) -> None:
    result = mmr_diversify(candidates, limit=limit)
    assert len(result) <= limit


@given(candidates=_candidates_st, limit=st.integers(min_value=0, max_value=40))
def test_mmr_diversify_output_is_subset_with_no_duplicates(candidates: list[Candidate], limit: int) -> None:
    result = mmr_diversify(candidates, limit=limit)
    input_ids = {c.product_id for c in candidates}
    result_ids = [c.product_id for c in result]
    assert len(result_ids) == len(set(result_ids))
    assert set(result_ids) <= input_ids
