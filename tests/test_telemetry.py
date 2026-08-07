from __future__ import annotations

import time

from src.shared.telemetry import LATENCY, Timer, get_logger


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("discovery.test")
    assert logger.name == "discovery.test"
    assert len(logger.handlers) == 1


def test_get_logger_does_not_duplicate_handlers() -> None:
    logger_a = get_logger("discovery.dup")
    logger_b = get_logger("discovery.dup")
    assert logger_a is logger_b
    assert len(logger_b.handlers) == 1


def test_timer_measures_elapsed_time() -> None:
    with Timer("test.block") as timer:
        time.sleep(0.01)
    assert timer.elapsed_ms >= 10


def test_timer_records_into_a_custom_target_dict() -> None:
    samples: dict[str, list[float]] = {}
    with Timer("retrieval", target=samples):
        time.sleep(0.005)
    assert len(samples["retrieval"]) == 1
    assert samples["retrieval"][0] > 0


def test_timer_defaults_to_module_level_latency_histogram() -> None:
    LATENCY.clear()
    with Timer("ranking"):
        time.sleep(0.005)
    assert len(LATENCY["ranking"]) == 1


def test_timer_appends_multiple_samples_under_the_same_name() -> None:
    samples: dict[str, list[float]] = {}
    with Timer("gateway.request", target=samples):
        pass
    with Timer("gateway.request", target=samples):
        pass
    assert len(samples["gateway.request"]) == 2


def test_timer_as_decorator() -> None:
    @Timer("work")
    def work() -> int:
        time.sleep(0.005)
        return 42

    assert work() == 42
