from __future__ import annotations

from pathlib import Path

from bench.plot_latency import build_histogram, read_latencies


def test_read_latencies_filters_by_name_and_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw_latencies.csv"
    csv_path.write_text(
        "name,response_time_ms,success\n"
        "/v1/feed,40.0,True\n"
        "/v1/feed,55.5,True\n"
        "/v1/feed,90.0,False\n"
        "/v1/events,5.0,True\n"
    )
    latencies = read_latencies(csv_path, "/v1/feed")
    assert latencies == [40.0, 55.5]


def test_read_latencies_empty_file_returns_empty_list(tmp_path: Path) -> None:
    csv_path = tmp_path / "raw_latencies.csv"
    csv_path.write_text("name,response_time_ms,success\n")
    assert read_latencies(csv_path, "/v1/feed") == []


def test_build_histogram_writes_a_nonempty_png(tmp_path: Path) -> None:
    output_path = tmp_path / "latency.png"
    result = build_histogram([10.0, 20.0, 30.0, 45.0, 90.0, 120.0], budget_ms=80.0, output_path=output_path)
    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_histogram_creates_missing_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "dir" / "latency.png"
    build_histogram([1.0, 2.0, 3.0], output_path=output_path)
    assert output_path.exists()
