from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from trainer.baseline import (
    _write_results_row,
    build_covisitation,
    make_recommend_fn,
    measure_latency_p50_ms,
    recommend,
)


def _train_df(rows: list[tuple[str, int, date]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "customer_id": [r[0] for r in rows],
            "article_id": [r[1] for r in rows],
            "t_dat": [r[2] for r in rows],
        }
    )


def test_covisitation_pairs_within_window_only() -> None:
    train = _train_df(
        [
            ("u1", 1, date(2024, 1, 1)),
            ("u1", 2, date(2024, 1, 2)),  # 1 day after item 1: within window
            ("u1", 3, date(2024, 1, 11)),  # 9 days after item 2: outside window
            ("u2", 2, date(2024, 1, 1)),
            ("u2", 3, date(2024, 1, 3)),  # 2 days after: within window
        ]
    )

    covisit = build_covisitation(train)

    assert covisit[1] == [(2, 1)]
    assert dict(covisit[2]) == {1: 1, 3: 1}
    assert covisit[3] == [(2, 1)]


def test_covisitation_caps_neighbors_per_item() -> None:
    rows: list[tuple[str, int, date]] = []
    for i in range(55):
        customer_id = f"u{i}"
        rows.append((customer_id, 1, date(2024, 1, 1)))
        rows.append((customer_id, 100 + i, date(2024, 1, 2)))

    covisit = build_covisitation(_train_df(rows))

    assert len(covisit[1]) == 50


def test_recommend_excludes_already_purchased() -> None:
    covisit = {
        1: [(10, 5), (20, 3)],
        2: [(10, 2), (30, 4)],
    }

    assert recommend(covisit, [1, 2], k=20) == [10, 30, 20]
    assert recommend(covisit, [1, 10], k=20) == [20]


def test_recommend_respects_k() -> None:
    covisit = {1: [(10, 5), (20, 4), (30, 3)]}
    assert recommend(covisit, [1], k=2) == [10, 20]


def test_recommend_only_uses_last_five_items() -> None:
    covisit = {1: [(99, 1)], 2: [(88, 1)]}
    # item 1 falls outside the last-5 window and should be ignored
    assert recommend(covisit, [1, 2, 3, 4, 5, 6], k=20) == [88]


def test_make_recommend_fn_matches_signature() -> None:
    covisit = {1: [(10, 5)]}
    recommend_fn = make_recommend_fn(covisit, k=20)
    assert recommend_fn("any_user", [1]) == [10]


def test_measure_latency_p50_ms_returns_non_negative_median() -> None:
    recommend_fn = make_recommend_fn({1: [(10, 1)]}, k=20)
    latency = measure_latency_p50_ms(recommend_fn, [[1], [1], [1]])
    assert latency >= 0.0


def test_measure_latency_p50_ms_empty_histories() -> None:
    recommend_fn = make_recommend_fn({}, k=20)
    assert measure_latency_p50_ms(recommend_fn, []) == 0.0


def test_write_results_row_creates_table_and_dedupes_on_rerun(tmp_path: Path) -> None:
    path = tmp_path / "results.md"
    metrics = {"recall@20": 0.1234, "ndcg@20": 0.5678, "catalog_coverage": 0.4321}

    _write_results_row(metrics, latency_p50_ms=1.5, path=path)
    first = path.read_text()
    assert "| Baseline | 0.1234 | 0.5678 | 0.4321 | 1.50ms |" in first
    assert first.count("| Baseline |") == 1

    _write_results_row(metrics, latency_p50_ms=2.5, path=path)
    second = path.read_text()
    assert second.count("| Baseline |") == 1
    assert "2.50ms" in second
