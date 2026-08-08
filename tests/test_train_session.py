from __future__ import annotations

import numpy as np
import pytest
import torch

from src.session.encoder import SessionEncoder
from trainer.build_index import build_index
from trainer.train_session import (
    SessionDataset,
    make_collate_fn,
    make_multi_intent_recommend_fn,
    run_batch,
    session_loss,
)


def _random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vectors = rng.rand(n, dim).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --- SessionDataset ---


def test_session_dataset_drops_customer_id() -> None:
    examples = [("u1", [1, 2], 3), ("u2", [4], 5)]
    dataset = SessionDataset(examples)

    assert len(dataset) == 2
    assert dataset[0] == ([1, 2], 3)
    assert dataset[1] == ([4], 5)


# --- make_collate_fn ---


def test_collate_fn_builds_padded_embeddings_and_mask() -> None:
    item_vectors_ext = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0]], dtype=np.float32)  # pad row = idx 3
    collate = make_collate_fn(item_vectors_ext, pad_idx=3, session_length=3)

    batch = [([0, 1], 2), ([2], 0)]
    embeddings, mask, targets = collate(batch)

    assert embeddings.shape == (2, 3, 2)
    assert torch.allclose(embeddings[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(embeddings[0, 1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(embeddings[0, 2], torch.tensor([0.0, 0.0]))  # padded slot
    assert mask.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert targets.tolist() == [2, 0]


# --- session_loss ---


def test_session_loss_lower_when_responsible_head_matches_target() -> None:
    torch.manual_seed(0)
    batch_size, k, dim = 4, 4, 8
    target_emb = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1)

    matching = torch.zeros(batch_size, k, dim)
    matching[:, 0, :] = target_emb  # head 0 matches perfectly
    weights_favor_matching = torch.zeros(batch_size, k)
    weights_favor_matching[:, 0] = 1.0

    mismatched = torch.nn.functional.normalize(torch.randn(batch_size, k, dim), dim=-1)
    weights_uniform = torch.full((batch_size, k), 1.0 / k)

    good_loss = session_loss(matching, weights_favor_matching, target_emb)
    bad_loss = session_loss(mismatched, weights_uniform, target_emb)

    assert good_loss.item() < bad_loss.item()


def test_session_loss_is_scalar() -> None:
    torch.manual_seed(0)
    intents = torch.nn.functional.normalize(torch.randn(5, 4, 8), dim=-1)
    weights = torch.softmax(torch.randn(5, 4), dim=-1)
    target_emb = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)

    loss = session_loss(intents, weights, target_emb)

    assert loss.dim() == 0


# --- run_batch ---


def test_run_batch_returns_finite_loss_and_correct_intent_shape() -> None:
    torch.manual_seed(0)
    encoder = SessionEncoder(embed_dim=8, n_heads=2, n_layers=2, session_length=4, n_intents=4)
    item_vectors_ext = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)

    embeddings = item_vectors_ext[[0, 1, 2, 5]].unsqueeze(0)  # last row (5) stands in for the pad row
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    target_idx = torch.tensor([3])
    batch = (embeddings, mask, target_idx)

    loss, intents = run_batch(encoder, item_vectors_ext, batch)

    assert torch.isfinite(loss)
    assert intents.shape == (1, 4, 8)


# --- make_multi_intent_recommend_fn ---


def test_multi_intent_recommend_fn_excludes_seen_and_respects_k() -> None:
    torch.manual_seed(0)
    embed_dim, n_items = 8, 20
    item_vectors = _random_unit_vectors(n_items, embed_dim)
    index = build_index(item_vectors)
    article_id_to_index = {100 + i: i for i in range(n_items)}
    index_to_article_id = {i: 100 + i for i in range(n_items)}

    encoder = SessionEncoder(embed_dim=embed_dim, n_heads=2, n_layers=2, session_length=20, n_intents=4)
    encoder.eval()

    recommend_fn = make_multi_intent_recommend_fn(
        encoder, item_vectors, article_id_to_index, index_to_article_id, index, k=5
    )

    history = [100, 101, 102]
    recommended = recommend_fn("u1", history)

    assert len(recommended) == 5
    assert set(recommended).isdisjoint(history)


def test_multi_intent_recommend_fn_returns_empty_for_unmapped_history() -> None:
    embed_dim, n_items = 8, 10
    item_vectors = _random_unit_vectors(n_items, embed_dim)
    index = build_index(item_vectors)
    article_id_to_index = {100 + i: i for i in range(n_items)}
    index_to_article_id = {i: 100 + i for i in range(n_items)}

    encoder = SessionEncoder(embed_dim=embed_dim, n_heads=2, n_layers=2, session_length=20, n_intents=4)
    encoder.eval()

    recommend_fn = make_multi_intent_recommend_fn(
        encoder, item_vectors, article_id_to_index, index_to_article_id, index, k=5
    )

    assert recommend_fn("cold_start_user", [999999]) == []
