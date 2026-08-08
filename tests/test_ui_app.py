from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

import ui.app as ui_app


def test_intent_color_assigns_fixed_order_and_falls_back_to_muted() -> None:
    assert ui_app.intent_color(ui_app.LABELS[0]) == ui_app.CATEGORICAL_PALETTE[0]
    assert ui_app.intent_color(ui_app.COLD_START_LABEL) == ui_app.MUTED_INK
    assert ui_app.intent_color("unknown_label") == ui_app.MUTED_INK
    assert ui_app.intent_color(None) == ui_app.MUTED_INK


def test_build_category_colors_is_stable_and_cycles_palette() -> None:
    colors = ui_app.build_category_colors(["shoes", "bags", "shoes", "hats"])
    assert colors["shoes"] == ui_app.CATEGORICAL_PALETTE[0]
    assert colors["bags"] == ui_app.CATEGORICAL_PALETTE[1]
    assert colors["hats"] == ui_app.CATEGORICAL_PALETTE[2]
    assert len(colors) == 3


def test_truncate_leaves_short_text_untouched() -> None:
    assert ui_app.truncate("Linen Shirt", length=40) == "Linen Shirt"


def test_truncate_shortens_long_text_with_ellipsis() -> None:
    result = ui_app.truncate("a" * 50, length=10)
    assert len(result) == 10
    assert result.endswith("…")


def test_truncate_handles_none() -> None:
    assert ui_app.truncate(None) == "Unknown item"


def test_category_share_frame_computes_shares_per_category() -> None:
    items = [{"product_id": "1"}, {"product_id": "2"}, {"product_id": "3"}, {"product_id": "4"}]
    catalog = {
        1: {"category_l1": "shoes"},
        2: {"category_l1": "shoes"},
        3: {"category_l1": "bags"},
        4: {"category_l1": "bags"},
    }
    df = ui_app.category_share_frame(items, catalog)
    shares = dict(zip(df["category"], df["share"]))
    assert shares == {"shoes": pytest.approx(0.5), "bags": pytest.approx(0.5)}


def test_category_share_frame_empty_items_returns_empty_frame() -> None:
    df = ui_app.category_share_frame([], {})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# --- gateway client helpers, exercised against httpx.MockTransport ---


def _client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gateway.test")


def test_fetch_feed_returns_parsed_json_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/feed"
        body = json.loads(request.content)
        assert body == {"user_id": "u1", "session_id": "s1", "limit": 20}
        return httpx.Response(200, json={"items": [], "detected_intents": [], "guardrails": {}})

    with _client_for(handler) as client:
        result = ui_app.fetch_feed(client, "u1", "s1")
    assert result == {"items": [], "detected_intents": [], "guardrails": {}}


def test_fetch_feed_returns_none_on_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "X"}})

    with _client_for(handler) as client:
        assert ui_app.fetch_feed(client, "u1", "s1") is None


def test_fetch_feed_returns_none_on_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client_for(handler) as client:
        assert ui_app.fetch_feed(client, "u1", "s1") is None


def test_post_event_returns_true_on_202() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/events"
        return httpx.Response(202)

    with _client_for(handler) as client:
        assert ui_app.post_event(client, "u1", "s1", 1234, "shoes") is True


def test_post_event_returns_false_on_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with _client_for(handler) as client:
        assert ui_app.post_event(client, "u1", "s1", 1234, "shoes") is False


def test_delete_user_returns_receipt_id_on_204() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/users/u1"
        return httpx.Response(204, headers={"X-Deletion-Receipt-Id": "abc-123"})

    with _client_for(handler) as client:
        ok, receipt = ui_app.delete_user(client, "u1")
    assert ok is True
    assert receipt == "abc-123"


def test_delete_user_reports_failure_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with _client_for(handler) as client:
        ok, message = ui_app.delete_user(client, "u1")
    assert ok is False
    assert "503" in message


def test_gateway_reachable_true_when_healthz_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    with _client_for(handler) as client:
        assert ui_app.gateway_reachable(client) is True


def test_gateway_reachable_false_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client_for(handler) as client:
        assert ui_app.gateway_reachable(client) is False


# --- full page smoke test via Streamlit's AppTest ---


def test_app_renders_gateway_unreachable_error_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GATEWAY_URL", "http://127.0.0.1:1")  # nothing listens here
    at = AppTest.from_file("ui/app.py")
    at.run(timeout=30)

    assert not at.exception
    assert any("Could not reach the gateway" in e.value for e in at.error)
