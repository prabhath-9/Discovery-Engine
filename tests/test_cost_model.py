from __future__ import annotations

from pathlib import Path

import pytest

from bench.cost_model import (
    CapacityResult,
    api_pods_needed,
    compute,
    format_report,
    hourly_cost,
    llm_comparison,
    offline_cost_per_1000,
    online_cost_per_1000,
    read_mean_latency_ms,
    retrieval_pods_needed,
)

# --- api_pods_needed ---


def test_api_pods_needed_matches_architecture_doc_reference_point() -> None:
    # architecture.md §10: 35ms mean, 16 workers/pod, 40% headroom -> ~32 pods
    throughput, pods = api_pods_needed(mean_latency_ms=35.0, target_rps=10_000, workers_per_pod=16, headroom=0.40)
    assert throughput == pytest.approx(16 / 0.035)
    assert pods == 31  # ceil(10_000 * 0.035 / 16 * 1.40) = ceil(30.625)


def test_api_pods_needed_scales_up_with_higher_latency() -> None:
    _, pods_fast = api_pods_needed(mean_latency_ms=20.0)
    _, pods_slow = api_pods_needed(mean_latency_ms=80.0)
    assert pods_slow > pods_fast


def test_api_pods_needed_is_always_an_integer_ceiling() -> None:
    _, pods = api_pods_needed(mean_latency_ms=1.0, target_rps=1, workers_per_pod=16, headroom=0.0)
    assert isinstance(pods, int)
    assert pods >= 1


# --- retrieval_pods_needed ---


def test_retrieval_pods_needed_matches_architecture_doc_reference_point() -> None:
    # 10,000 * 4 queries * 2ms = 80 cores / 4 vCPU = 20 pods
    assert retrieval_pods_needed() == 20


# --- hourly_cost ---


def test_hourly_cost_matches_architecture_doc_reference_point() -> None:
    cost = hourly_cost(api_pods=32, retrieval_pods=20)
    assert cost == pytest.approx(9.52, abs=0.01)


# --- online / offline cost per 1000 ---


def test_online_cost_per_1000_matches_architecture_doc_reference_point() -> None:
    assert online_cost_per_1000(cost_per_hour=9.52) == pytest.approx(0.00026, abs=0.00001)


def test_offline_cost_per_1000_matches_architecture_doc_reference_point() -> None:
    assert offline_cost_per_1000() == pytest.approx(0.0000023, abs=0.0000001)


# --- llm_comparison ---


def test_llm_comparison_matches_architecture_doc_headline() -> None:
    llm_cost_per_1000, multiplier = llm_comparison(cost_per_1000=0.00026)
    assert llm_cost_per_1000 == pytest.approx(0.50)
    assert multiplier == pytest.approx(1_923, abs=5)


def test_llm_comparison_handles_zero_cost() -> None:
    _, multiplier = llm_comparison(cost_per_1000=0.0)
    assert multiplier == float("inf")


# --- compute (end to end) ---


def test_compute_reproduces_architecture_doc_headline_at_reference_latency() -> None:
    result = compute(mean_latency_ms=35.0)
    assert isinstance(result, CapacityResult)
    assert result.cost_per_million == pytest.approx(0.27, abs=0.02)  # doc headline: ~$0.26/million
    assert result.llm_multiplier > 1_000


def test_compute_higher_latency_costs_more_per_million() -> None:
    cheap = compute(mean_latency_ms=20.0)
    expensive = compute(mean_latency_ms=100.0)
    assert expensive.cost_per_million > cheap.cost_per_million


def test_compute_concurrent_in_flight_uses_littles_law() -> None:
    result = compute(mean_latency_ms=40.0, target_rps=1_000)
    assert result.concurrent_in_flight == pytest.approx(40.0)


# --- read_mean_latency_ms ---


def test_read_mean_latency_ms_finds_named_endpoint_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "results_stats.csv"
    csv_path.write_text(
        "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time\n"
        "POST,/v1/feed,1000,0,40,42.5\n"
        "POST,/v1/events,1000,0,5,5.5\n"
        ",Aggregated,2000,0,20,24.0\n"
    )
    assert read_mean_latency_ms(csv_path, "/v1/feed") == pytest.approx(42.5)


def test_read_mean_latency_ms_falls_back_to_aggregated(tmp_path: Path) -> None:
    csv_path = tmp_path / "results_stats.csv"
    csv_path.write_text(
        "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time\n"
        ",Aggregated,2000,0,20,24.0\n"
    )
    assert read_mean_latency_ms(csv_path, "/v1/feed") == pytest.approx(24.0)


def test_read_mean_latency_ms_raises_when_nothing_matches(tmp_path: Path) -> None:
    csv_path = tmp_path / "results_stats.csv"
    csv_path.write_text("Type,Name,Average Response Time\nPOST,/other,1.0\n")
    with pytest.raises(ValueError):
        read_mean_latency_ms(csv_path, "/v1/feed")


# --- format_report ---


def test_format_report_includes_key_numbers() -> None:
    result = compute(mean_latency_ms=35.0)
    report = format_report(result)
    assert "API pods needed" in report
    assert "Cost per hour" in report
    assert "Cost per 1,000 recommendations" in report
    assert "more expensive" in report
