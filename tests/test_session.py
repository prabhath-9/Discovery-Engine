from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.session.encoder import (
    EMBED_DIM,
    SessionEncoder,
    orthogonality_penalty,
    pairwise_head_cosine,
)
from src.session.intents import (
    LABELS,
    COLD_START_LABEL,
    Intent,
    SessionEvent,
    _label_scores,
    _pad_embeddings,
    infer_intents,
)


def _now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0)


# --- pairwise_head_cosine / orthogonality_penalty ---


def test_pairwise_head_cosine_returns_six_pairs_for_four_heads() -> None:
    intents = nn_normalize(torch.randn(3, 4, 8))
    pairwise = pairwise_head_cosine(intents)
    assert pairwise.shape == (3, 6)


def test_orthogonality_penalty_low_for_orthogonal_vectors() -> None:
    intents = torch.zeros(1, 4, 4)
    for i in range(4):
        intents[0, i, i] = 1.0
    assert orthogonality_penalty(intents).item() == pytest.approx(0.0, abs=1e-6)


def test_orthogonality_penalty_high_for_identical_vectors() -> None:
    base = nn_normalize(torch.randn(1, 1, 8))
    intents = base.expand(1, 4, 8)
    assert orthogonality_penalty(intents).item() == pytest.approx(1.0, abs=1e-5)


def nn_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


# --- SessionEncoder ---


def test_session_encoder_output_shapes() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder(embed_dim=16, n_heads=2, n_layers=2, session_length=5, n_intents=4)
    encoder.eval()

    embeddings = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5)

    intents, weights = encoder(embeddings, mask)

    assert intents.shape == (3, 4, 16)
    assert weights.shape == (3, 4)


def test_session_encoder_produces_four_distinct_intent_vectors() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder(embed_dim=16, n_heads=2, n_layers=2, session_length=5, n_intents=4)
    encoder.eval()

    embeddings = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5)

    intents, _ = encoder(embeddings, mask)
    pairwise = pairwise_head_cosine(intents)

    assert pairwise.max().item() < 0.9


def test_session_encoder_weights_sum_to_one() -> None:
    torch.manual_seed(1)
    encoder = SessionEncoder(embed_dim=16, n_heads=2, n_layers=2, session_length=5, n_intents=4)
    encoder.eval()

    embeddings = torch.randn(4, 5, 16)
    mask = torch.ones(4, 5)

    _, weights = encoder(embeddings, mask)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_padded_positions_do_not_affect_output() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder(embed_dim=8, n_heads=2, n_layers=2, session_length=5, n_intents=4)
    encoder.eval()

    base = torch.randn(1, 5, 8)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])

    variant = base.clone()
    variant[0, 3:] = torch.randn(2, 8) * 100  # scramble the masked-out tail

    with torch.no_grad():
        intents_a, weights_a = encoder(base, mask)
        intents_b, weights_b = encoder(variant, mask)

    assert torch.allclose(intents_a, intents_b, atol=1e-5)
    assert torch.allclose(weights_a, weights_b, atol=1e-5)


# --- _pad_embeddings ---


def test_pad_embeddings_pads_short_sequences() -> None:
    embeddings = np.ones((2, 4), dtype=np.float32)
    padded, mask = _pad_embeddings(embeddings, length=5)

    assert padded.shape == (1, 5, 4)
    assert mask.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0]]
    assert torch.allclose(padded[0, 2:], torch.zeros(3, 4))


def test_pad_embeddings_truncates_long_sequences() -> None:
    embeddings = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    padded, mask = _pad_embeddings(embeddings, length=3)

    assert padded.shape == (1, 3, 4)
    assert mask.tolist() == [[1.0, 1.0, 1.0]]
    assert torch.allclose(padded[0], torch.tensor(embeddings[:3]))


# --- infer_intents: cold start ---


def test_infer_intents_empty_session_returns_cold_start() -> None:
    intents = infer_intents([])

    assert len(intents) == 1
    assert intents[0].label == COLD_START_LABEL
    assert intents[0].weight == 1.0
    assert intents[0].vector == [0.0] * EMBED_DIM


def test_infer_intents_all_unmapped_items_returns_cold_start() -> None:
    events = [SessionEvent(article_id=999, timestamp=_now(), price=10.0, category_l1="shoes")]

    intents = infer_intents(events, encoder=object(), item_vectors=np.zeros((1, 8)), article_id_to_index={})

    assert len(intents) == 1
    assert intents[0].label == COLD_START_LABEL


# --- infer_intents: real (injected) model ---


def _fake_catalog(n_items: int, dim: int, seed: int = 0) -> tuple[np.ndarray, dict[int, int]]:
    rng = np.random.RandomState(seed)
    vectors = rng.rand(n_items, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, {100 + i: i for i in range(n_items)}


def test_infer_intents_returns_four_intents_ranked_by_weight() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder()
    encoder.eval()
    item_vectors, article_id_to_index = _fake_catalog(10, EMBED_DIM)

    events = [
        SessionEvent(article_id=100, timestamp=_now(), price=10.0, category_l1="shoes"),
        SessionEvent(article_id=101, timestamp=_now() + timedelta(minutes=1), price=12.0, category_l1="shoes"),
    ]

    intents = infer_intents(events, encoder=encoder, item_vectors=item_vectors, article_id_to_index=article_id_to_index)

    assert len(intents) == 4
    assert all(isinstance(i, Intent) for i in intents)
    assert {i.label for i in intents} == set(LABELS)
    assert all(len(i.vector) == EMBED_DIM for i in intents)

    weights = [i.weight for i in intents]
    assert weights == sorted(weights, reverse=True)
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


# --- _label_scores heuristics ---


def test_label_scores_favor_urgent_replacement_for_rapid_single_category_session() -> None:
    events = [
        SessionEvent(article_id=1, timestamp=_now(), price=20.0, category_l1="phone_chargers"),
        SessionEvent(article_id=2, timestamp=_now() + timedelta(seconds=30), price=22.0, category_l1="phone_chargers"),
        SessionEvent(article_id=3, timestamp=_now() + timedelta(seconds=60), price=19.0, category_l1="phone_chargers"),
    ]
    scores = _label_scores(events)
    assert max(scores, key=scores.get) == "urgent_replacement"


def test_label_scores_favor_seasonal_browsing_for_wide_spread_leisurely_session() -> None:
    categories = ["shoes", "bags", "hats", "dresses", "coats"]
    events = [
        SessionEvent(article_id=i, timestamp=_now() + timedelta(minutes=i * 20), price=30.0, category_l1=categories[i])
        for i in range(5)
    ]
    scores = _label_scores(events)
    assert max(scores, key=scores.get) == "seasonal_browsing"


def test_label_scores_favor_complementary_for_related_items_bought_together() -> None:
    events = [
        SessionEvent(article_id=1, timestamp=_now(), price=40.0, category_l1="dresses"),
        SessionEvent(article_id=2, timestamp=_now() + timedelta(minutes=1), price=15.0, category_l1="shoes"),
        SessionEvent(article_id=3, timestamp=_now() + timedelta(minutes=2), price=20.0, category_l1="bags"),
    ]
    scores = _label_scores(events)
    assert max(scores, key=scores.get) == "complementary"


def test_label_scores_favor_bargain_hunting_for_heavily_discounted_session() -> None:
    events = [
        SessionEvent(
            article_id=i,
            timestamp=_now() + timedelta(minutes=i),
            price=float(10 + i * 5),
            category_l1="misc",
            discounted=True,
        )
        for i in range(5)
    ]
    scores = _label_scores(events)
    assert max(scores, key=scores.get) == "bargain_hunting"
