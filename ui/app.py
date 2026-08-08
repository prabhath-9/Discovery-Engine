from __future__ import annotations

import os
import uuid
from collections import Counter
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import polars as pl
import streamlit as st

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
ITEMS_PATH = Path("data/processed/items.parquet")

SIMULATOR_SAMPLE_SIZE = 12
DIVERSITY_CAP = 0.35
FEED_LIMIT = 20

# Mirrors src/session/intents.py's LABELS and COLD_START_LABEL. Duplicated
# (not imported) so this script stays runnable standalone via `streamlit
# run` — Streamlit's script runner doesn't add the project root to
# sys.path the way `python -m` does, and the UI only needs the label
# strings the gateway API already returns, not the backend module itself.
LABELS = ("urgent_replacement", "seasonal_browsing", "complementary", "bargain_hunting")
COLD_START_LABEL = "cold_start"

DEMO_USERS = [f"demo-user-{i}" for i in range(1, 6)]
NEW_USER_LABEL = "New guest (cold start)"

# Fixed categorical order, validated for adjacent-pair colorblind separation.
CATEGORICAL_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED_INK = "#898781"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

INTENT_COLORS = {label: CATEGORICAL_PALETTE[i] for i, label in enumerate(LABELS)}
INTENT_COLORS[COLD_START_LABEL] = MUTED_INK
INTENT_COLORS["search"] = CATEGORICAL_PALETTE[0]


def intent_color(label: str | None) -> str:
    return INTENT_COLORS.get(label or "", MUTED_INK)


def build_category_colors(categories: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(categories))
    return {cat: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, cat in enumerate(unique)}


def truncate(text: str | None, length: int = 40) -> str:
    text = text or "Unknown item"
    return text if len(text) <= length else text[: length - 1] + "…"


def category_share_frame(items: list[dict], catalog_by_id: dict[int, dict]) -> pd.DataFrame:
    categories = [catalog_by_id.get(int(item["product_id"]), {}).get("category_l1", "unknown") for item in items]
    counts = Counter(categories)
    total = sum(counts.values()) or 1
    colors = build_category_colors(list(counts))
    return pd.DataFrame(
        {
            "category": list(counts),
            "share": [count / total for count in counts.values()],
            "color": [colors[c] for c in counts],
        }
    )


# --- gateway client ---


def make_client(base_url: str, timeout: float = 5.0) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)


def fetch_feed(client: httpx.Client, user_id: str, session_id: str, limit: int = FEED_LIMIT) -> dict | None:
    try:
        response = client.post("/v1/feed", json={"user_id": user_id, "session_id": session_id, "limit": limit})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def post_event(client: httpx.Client, user_id: str, session_id: str, article_id: int, category_l1: str) -> bool:
    try:
        response = client.post(
            "/v1/events",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "article_id": article_id,
                "event_type": "click",
                "category_l1": category_l1,
            },
        )
        return response.status_code == 202
    except Exception:
        return False


def fetch_search(client: httpx.Client, query: str, user_id: str, session_id: str, limit: int = 12) -> dict | None:
    try:
        response = client.post(
            "/v1/search", json={"query": query, "user_id": user_id, "session_id": session_id, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_bundles(client: httpx.Client, seed_product_id: int, user_id: str, limit: int = 8) -> dict | None:
    try:
        response = client.post("/v1/bundles", json={"seed_product_id": seed_product_id, "user_id": user_id, "limit": limit})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def delete_user(client: httpx.Client, user_id: str) -> tuple[bool, str]:
    try:
        response = client.delete(f"/v1/users/{user_id}")
        if response.status_code == 204:
            return True, response.headers.get("X-Deletion-Receipt-Id", "(no receipt id)")
        return False, f"gateway returned status {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def gateway_reachable(client: httpx.Client) -> bool:
    try:
        return client.get("/healthz").status_code == 200
    except Exception:
        return False


# --- catalog cache (session_state, loaded once) ---


def ensure_catalog_loaded() -> None:
    if "catalog_by_id" in st.session_state:
        return
    if not ITEMS_PATH.exists():
        st.session_state["catalog_by_id"] = {}
        st.session_state["catalog_ids"] = []
        return
    df = pl.read_parquet(ITEMS_PATH).select("article_id", "title", "category_l1", "colour")
    st.session_state["catalog_by_id"] = {row["article_id"]: row for row in df.to_dicts()}
    st.session_state["catalog_ids"] = df["article_id"].to_list()


def ensure_simulator_sample() -> None:
    if "simulator_sample" in st.session_state:
        return
    ids = st.session_state["catalog_ids"]
    if not ids:
        st.session_state["simulator_sample"] = []
        return
    # a fixed seed keeps the grid stable across reruns within a browser session
    step = max(1, len(ids) // SIMULATOR_SAMPLE_SIZE)
    st.session_state["simulator_sample"] = ids[::step][:SIMULATOR_SAMPLE_SIZE]


# --- render sections ---


def render_sidebar(client: httpx.Client) -> tuple[str, str]:
    st.sidebar.title("Demo controls")

    if "flash" in st.session_state:
        kind, message = st.session_state.pop("flash")
        getattr(st.sidebar, kind)(message)

    status = "🟢 reachable" if gateway_reachable(client) else "🔴 unreachable"
    st.sidebar.caption(f"Gateway `{GATEWAY_URL}` — {status}")
    st.sidebar.divider()

    st.sidebar.subheader("User")
    choice = st.sidebar.selectbox("Choose a user", DEMO_USERS + [NEW_USER_LABEL])
    if choice == NEW_USER_LABEL:
        if "guest_id" not in st.session_state or st.sidebar.button("Start a fresh guest session"):
            st.session_state["guest_id"] = f"guest-{uuid.uuid4().hex[:8]}"
        user_id = st.session_state["guest_id"]
        st.sidebar.caption(f"`{user_id}` — starts cold, click products below to build a session")
    else:
        user_id = choice
    session_id = f"{user_id}-s1"

    st.sidebar.divider()
    st.sidebar.subheader("Complete the Look")
    sample_ids = st.session_state.get("simulator_sample", [])
    catalog = st.session_state.get("catalog_by_id", {})
    if sample_ids:
        seed_id = st.sidebar.selectbox(
            "Seed product", sample_ids, format_func=lambda a: truncate(catalog.get(a, {}).get("title"), 30)
        )
        if st.sidebar.button("Find bundle"):
            st.session_state["bundle_results"] = fetch_bundles(client, seed_id, user_id)
    else:
        st.sidebar.caption("Catalog not loaded.")

    st.sidebar.divider()
    st.sidebar.subheader("Search")
    query = st.sidebar.text_input("Search products", key="search_query")
    if st.sidebar.button("Search") and query.strip():
        st.session_state["search_results"] = fetch_search(client, query.strip(), user_id, session_id)

    st.sidebar.divider()
    st.sidebar.subheader("Privacy (DPDP)")
    if st.sidebar.button("🗑️ Delete my data"):
        ok, info = delete_user(client, user_id)
        if ok:
            st.session_state["flash"] = ("success", f"Deleted. Receipt: `{info}`")
            st.session_state.pop("search_results", None)
            st.session_state.pop("bundle_results", None)
        else:
            st.session_state["flash"] = ("error", f"Delete failed: {info}")
        st.rerun()

    return user_id, session_id


def render_simulator(client: httpx.Client, user_id: str, session_id: str) -> None:
    st.subheader("1. Session simulator")
    st.caption("Click a product to simulate a view. Watch the intent panel and feed react — this is the cold-start demo.")

    sample_ids = st.session_state.get("simulator_sample", [])
    catalog = st.session_state.get("catalog_by_id", {})
    if not sample_ids:
        st.info("No catalog available — run `trainer.sample_data` first.")
        return

    cols = st.columns(4)
    for i, article_id in enumerate(sample_ids):
        item = catalog.get(article_id, {})
        with cols[i % 4], st.container(border=True):
            st.markdown(f"**{truncate(item.get('title'))}**")
            st.caption(item.get("category_l1", "unknown"))
            if st.button("👁️ View", key=f"sim-{article_id}"):
                post_event(client, user_id, session_id, article_id, item.get("category_l1", "unknown"))


def render_item_cards(items: list[dict], catalog_by_id: dict[int, dict], columns: int = 4) -> None:
    cols = st.columns(columns)
    for i, item in enumerate(items):
        article_id = int(item["product_id"])
        meta = catalog_by_id.get(article_id, {})
        with cols[i % columns], st.container(border=True):
            st.markdown(f"**{truncate(meta.get('title'))}**")
            st.caption(meta.get("category_l1", "unknown"))
            label = item.get("intent") or "n/a"
            color = intent_color(item.get("intent"))
            st.markdown(
                f'<span style="background:{color};color:#ffffff;padding:2px 9px;border-radius:10px;'
                f'font-size:0.72rem;font-weight:600;">{label}</span>',
                unsafe_allow_html=True,
            )
            st.caption(item["reason"]["text"])
            st.caption(f"score {item['score']:.3f}")


def render_feed(items: list[dict]) -> None:
    st.subheader("2. Feed")
    if not items:
        st.info("No items returned for this feed.")
        return
    render_item_cards(items, st.session_state.get("catalog_by_id", {}))


def render_intent_panel(intents: list[dict]) -> None:
    st.subheader("3. Detected intents")
    if not intents:
        st.info("No intents detected.")
        return

    df = pd.DataFrame(intents)
    df["pct"] = df["confidence"] * 100
    df["color"] = df["label"].map(intent_color)
    df["display"] = df.apply(lambda r: f"{r['pct']:.0f}%  ·  {int(r['slots'])} items in feed", axis=1)

    base = alt.Chart(df).encode(y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=15, labelFontWeight="bold")))
    bars = base.mark_bar(cornerRadiusEnd=4, size=32).encode(
        x=alt.X("confidence:Q", title="confidence", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color(
            "label:N", scale=alt.Scale(domain=df["label"].tolist(), range=df["color"].tolist()), legend=None
        ),
        tooltip=[alt.Tooltip("label:N", title="Intent"), alt.Tooltip("pct:Q", title="Confidence %", format=".0f"), alt.Tooltip("slots:Q", title="Items in feed")],
    )
    labels = base.mark_text(align="left", dx=8, fontSize=14, fontWeight="bold", color="#0b0b0b").encode(
        x=alt.X("confidence:Q"), text=alt.Text("display:N")
    )
    chart = (bars + labels).properties(height=max(200, 70 * len(df)))
    st.altair_chart(chart, use_container_width=True)

    top = df.iloc[0]
    st.markdown(f"**Reading this feed as:** `{top['label']}` — {top['pct']:.0f}% confidence, {int(top['slots'])} items")


def render_guardrail_panel(guardrails: dict, items: list[dict]) -> None:
    st.subheader("4. Guardrails")
    left, right = st.columns([1, 1.3])

    with left:
        st.metric("Items dropped by guardrails", guardrails["items_dropped"])
        filters_fired = guardrails.get("filters_fired", {})
        if filters_fired:
            st.markdown("**Filters that fired:**")
            for name, count in sorted(filters_fired.items(), key=lambda kv: -kv[1]):
                st.markdown(f"- `{name}` dropped **{count}**")
        else:
            st.success("No filters dropped any candidates this round.")
        if guardrails["diversity_cap_applied"]:
            st.warning("The 35% diversity cap was actively enforced on this list.")

    with right:
        df = category_share_frame(items, st.session_state.get("catalog_by_id", {}))
        if df.empty:
            st.caption("No items to chart.")
            return
        df["over_cap"] = df["share"] > DIVERSITY_CAP
        y_max = max(0.4, float(df["share"].max()) * 1.15)

        bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("category:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("share:Q", axis=alt.Axis(format="%", title="share of feed"), scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color(
                "category:N", scale=alt.Scale(domain=df["category"].tolist(), range=df["color"].tolist()), legend=None
            ),
            tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("share:Q", title="Share", format=".1%")],
        )
        cap_line = alt.Chart(pd.DataFrame({"y": [DIVERSITY_CAP]})).mark_rule(color=CRITICAL, strokeDash=[5, 3], size=2).encode(y="y:Q")
        cap_label = (
            alt.Chart(pd.DataFrame({"y": [DIVERSITY_CAP], "label": ["35% cap"]}))
            .mark_text(align="left", dx=4, dy=-8, color=CRITICAL, fontWeight="bold")
            .encode(y="y:Q", text="label:N")
        )
        st.altair_chart((bars + cap_line + cap_label).properties(height=300), use_container_width=True)

        if df["over_cap"].any():
            st.error("A category exceeds the 35% cap in this view.")
        else:
            st.success("Every category is within the 35% cap.")


def render_search_results(data: dict) -> None:
    st.divider()
    st.subheader(f"🔎 Search results for “{data['query']}”")
    if not data["items"]:
        st.info("No matches.")
        return
    render_item_cards(data["items"], st.session_state.get("catalog_by_id", {}))


def render_bundle_results(data: dict) -> None:
    st.divider()
    st.subheader(f"👗 Complete the look — seed {data['seed_product_id']}")
    if not data["items"]:
        st.info("No complements or similar items found.")
        return
    render_item_cards(data["items"], st.session_state.get("catalog_by_id", {}))


def main() -> None:
    st.set_page_config(page_title="Discovery Engine", layout="wide")
    ensure_catalog_loaded()
    ensure_simulator_sample()

    st.title("Discovery Engine")
    st.caption("Multi-intent recommendation, live against the gateway.")

    with make_client(GATEWAY_URL) as client:
        user_id, session_id = render_sidebar(client)

        feed_data = fetch_feed(client, user_id, session_id, limit=FEED_LIMIT)
        if feed_data is None:
            st.error(f"Could not reach the gateway at `{GATEWAY_URL}`. Start it and reload.")
            st.stop()

        if feed_data.get("degraded"):
            st.warning(f"Serving in degraded mode: `{feed_data['degraded']}`")

        render_simulator(client, user_id, session_id)
        st.divider()
        render_feed(feed_data["items"])
        st.divider()
        render_intent_panel(feed_data["detected_intents"])
        st.divider()
        render_guardrail_panel(feed_data["guardrails"], feed_data["items"])

        if st.session_state.get("search_results"):
            render_search_results(st.session_state["search_results"])
        if st.session_state.get("bundle_results"):
            render_bundle_results(st.session_state["bundle_results"])


if __name__ == "__main__":
    main()
