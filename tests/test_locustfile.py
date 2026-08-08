from __future__ import annotations

from pathlib import Path

import polars as pl

from bench.locustfile import (
    CATALOG_IDS,
    CATEGORIES,
    RETURNING_USER_IDS,
    RETURNING_POOL_SIZE,
    load_categories,
    load_catalog_ids,
    record_request,
    write_raw_samples,
)


def test_load_catalog_ids_falls_back_when_file_missing(tmp_path: Path) -> None:
    ids = load_catalog_ids(tmp_path / "missing.parquet")
    assert ids == list(range(100_000, 100_200))


def test_load_categories_falls_back_when_file_missing(tmp_path: Path) -> None:
    assert load_categories(tmp_path / "missing.parquet") == ["unknown"]


def test_load_catalog_ids_reads_real_file(tmp_path: Path) -> None:
    path = tmp_path / "items.parquet"
    pl.DataFrame({"article_id": [1, 2, 3], "category_l1": ["a", "b", "a"]}).write_parquet(path)
    assert load_catalog_ids(path) == [1, 2, 3]


def test_load_categories_reads_unique_values(tmp_path: Path) -> None:
    path = tmp_path / "items.parquet"
    pl.DataFrame({"article_id": [1, 2, 3], "category_l1": ["a", "b", "a"]}).write_parquet(path)
    assert set(load_categories(path)) == {"a", "b"}


def test_module_level_catalog_loaded_from_real_processed_data() -> None:
    # data/processed/items.parquet exists in this repo checkout, so the
    # module-level globals should be populated from it, not the fallback.
    assert len(CATALOG_IDS) > 200
    assert len(CATEGORIES) > 1
    assert len(RETURNING_USER_IDS) == RETURNING_POOL_SIZE


def test_record_request_appends_success_flag() -> None:
    samples: list[tuple[str, float, bool]] = []
    record_request(samples, "/v1/feed", 42.5, None)
    record_request(samples, "/v1/feed", 100.0, RuntimeError("boom"))
    assert samples == [("/v1/feed", 42.5, True), ("/v1/feed", 100.0, False)]


def test_write_raw_samples_writes_csv_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    write_raw_samples([("/v1/feed", 10.0, True), ("/v1/events", 2.0, False)], path)
    content = path.read_text()
    lines = content.splitlines()
    assert lines[0] == "name,response_time_ms,success"
    assert lines[1] == "/v1/feed,10.0,True"
    assert lines[2] == "/v1/events,2.0,False"


def test_write_raw_samples_creates_missing_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "raw.csv"
    write_raw_samples([("/v1/feed", 1.0, True)], path)
    assert path.exists()
