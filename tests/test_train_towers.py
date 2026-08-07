from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from trainer.train_towers import (
    ItemFeatures,
    ItemTower,
    TwoTowerDataset,
    UserTower,
    build_item_features,
    build_train_examples,
    build_val_examples,
    build_vocab,
    compute_val_loss,
    encode_item_text,
    gather_item_embeddings,
    make_collate_fn,
    masked_mean_pool,
    pad_and_mask,
    two_tower_loss,
    warmup_cosine_decay,
)


def test_build_vocab_is_sorted_and_deterministic() -> None:
    assert build_vocab(["b", "a", "a", "c"]) == {"a": 0, "b": 1, "c": 2}


def test_encode_item_text_uses_cache_without_loading_model(tmp_path: Path) -> None:
    cached = np.random.rand(3, 8).astype(np.float32)
    path = tmp_path / "item_text.npy"
    np.save(path, cached)

    items = pl.DataFrame({"title": ["a", "b", "c"]})
    result = encode_item_text(items, path=path)

    assert np.array_equal(result, cached)


def test_pad_and_mask_pads_short_history() -> None:
    padded, mask = pad_and_mask([1, 2], length=5, pad_idx=99)
    assert padded == [1, 2, 99, 99, 99]
    assert mask == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_pad_and_mask_trims_long_history() -> None:
    padded, mask = pad_and_mask([1, 2, 3, 4, 5], length=3, pad_idx=99)
    assert padded == [3, 4, 5]
    assert mask == [1.0, 1.0, 1.0]


def test_build_train_examples_uses_expanding_history() -> None:
    examples = build_train_examples({"u1": [10, 20, 30]}, session_length=20)
    assert examples == [("u1", [10], 20), ("u1", [10, 20], 30)]


def test_build_train_examples_caps_history_at_session_length() -> None:
    examples = build_train_examples({"u1": [1, 2, 3, 4, 5]}, session_length=2)
    assert examples[-1] == ("u1", [3, 4], 5)


def test_build_val_examples_skips_customers_without_train_history() -> None:
    val = pl.DataFrame({"customer_id": ["cold_start"], "article_id": [111]})
    examples = build_val_examples({}, val, {111: 0})
    assert examples == []


def test_build_val_examples_skips_unknown_articles() -> None:
    val = pl.DataFrame({"customer_id": ["u1"], "article_id": [999]})
    examples = build_val_examples({"u1": [0]}, val, {111: 0})
    assert examples == []


def test_build_val_examples_uses_full_train_history() -> None:
    val = pl.DataFrame({"customer_id": ["u1"], "article_id": [111]})
    examples = build_val_examples({"u1": [5, 6, 7]}, val, {111: 0}, session_length=2)
    assert examples == [("u1", [6, 7], 0)]


def test_masked_mean_pool_ignores_masked_positions() -> None:
    embeddings = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    pooled = masked_mean_pool(embeddings, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_masked_mean_pool_empty_history_does_not_divide_by_zero() -> None:
    embeddings = torch.zeros((1, 3, 2))
    mask = torch.zeros((1, 3))
    pooled = masked_mean_pool(embeddings, mask)
    assert torch.allclose(pooled, torch.zeros((1, 2)))


def test_item_tower_output_is_l2_normalized_and_shaped() -> None:
    tower = ItemTower(n_category_l1=3, n_category_l2=3, n_colour=3, n_dept=3, text_dim=8, embed_dim=16)
    out = tower(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.rand(2, 8),
    )
    assert out.shape == (2, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_user_tower_output_is_l2_normalized_and_shaped() -> None:
    tower = UserTower(n_age_band=2, n_region=2, embed_dim=16)
    out = tower(
        torch.rand(4, 5, 16),
        torch.ones(4, 5),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
    )
    assert out.shape == (4, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_warmup_cosine_decay_ramps_up_then_decays_to_zero() -> None:
    total_steps, warmup_steps = 100, 10
    assert warmup_cosine_decay(0, total_steps, warmup_steps) == pytest.approx(0.0)
    assert warmup_cosine_decay(5, total_steps, warmup_steps) == pytest.approx(0.5)
    assert warmup_cosine_decay(warmup_steps, total_steps, warmup_steps) == pytest.approx(1.0)
    assert warmup_cosine_decay(total_steps, total_steps, warmup_steps) == pytest.approx(0.0, abs=1e-9)

    mid = warmup_cosine_decay(55, total_steps, warmup_steps)
    late = warmup_cosine_decay(90, total_steps, warmup_steps)
    assert 0.0 < late < mid < 1.0


def test_two_tower_loss_is_lower_for_matching_embeddings() -> None:
    user_emb = nn_normalize(torch.eye(4))
    matching = user_emb.clone()
    shuffled = matching[[1, 2, 3, 0]]

    matching_loss = two_tower_loss(user_emb, matching, temperature=0.05)
    shuffled_loss = two_tower_loss(user_emb, shuffled, temperature=0.05)

    assert matching_loss.item() < shuffled_loss.item()


def nn_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)


def test_make_collate_fn_builds_padded_batch() -> None:
    collate = make_collate_fn(pad_idx=9, session_length=3)
    batch = [([1, 2], 5, 0, 1), ([3], 6, 1, 0)]

    hist, mask, age_band, region, target = collate(batch)

    assert hist.tolist() == [[1, 2, 9], [3, 9, 9]]
    assert mask.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert age_band.tolist() == [0, 1]
    assert region.tolist() == [1, 0]
    assert target.tolist() == [5, 6]


def _tiny_item_features() -> ItemFeatures:
    items = pl.DataFrame(
        {
            "category_l1": ["A", "B"],
            "category_l2": ["A1", "B1"],
            "colour": ["red", "blue"],
            "dept": ["d1", "d2"],
        }
    )
    text_embeddings = np.random.rand(2, 4).astype(np.float32)
    vocab = {"A": {"A": 0, "B": 1}, "A1": {"A1": 0, "B1": 1}, "red": {"blue": 0, "red": 1}, "d1": {"d1": 0, "d2": 1}}
    return build_item_features(items, text_embeddings, vocab["A"], vocab["A1"], vocab["red"], vocab["d1"])


def test_build_item_features_appends_masked_pad_row() -> None:
    features = _tiny_item_features()
    assert features.category_l1.shape == (3,)
    assert features.text.shape == (3, 4)
    assert torch.allclose(features.text[-1], torch.zeros(4))


def test_gather_item_embeddings_matches_direct_forward() -> None:
    features = _tiny_item_features()
    tower = ItemTower(n_category_l1=2, n_category_l2=2, n_colour=2, n_dept=2, text_dim=4, embed_dim=8)

    direct = tower(
        features.category_l1[[0, 1]],
        features.category_l2[[0, 1]],
        features.colour[[0, 1]],
        features.dept[[0, 1]],
        features.text[[0, 1]],
    )
    gathered = gather_item_embeddings(tower, features, torch.tensor([1, 0, 1]))

    assert torch.allclose(gathered[0], direct[1], atol=1e-6)
    assert torch.allclose(gathered[1], direct[0], atol=1e-6)
    assert torch.allclose(gathered[2], direct[1], atol=1e-6)


def test_compute_val_loss_is_nonnegative_and_deterministic_in_eval_mode() -> None:
    features = _tiny_item_features()
    item_tower = ItemTower(n_category_l1=2, n_category_l2=2, n_colour=2, n_dept=2, text_dim=4, embed_dim=8)
    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=8)

    examples = [("u1", [0], 1), ("u1", [1], 0)]
    demo_by_customer = {"u1": (0, 0)}
    loader = DataLoader(
        TwoTowerDataset(examples, demo_by_customer),
        batch_size=2,
        collate_fn=make_collate_fn(pad_idx=2, session_length=3),
    )

    loss_a = compute_val_loss(item_tower, user_tower, features, loader)
    loss_b = compute_val_loss(item_tower, user_tower, features, loader)

    assert loss_a >= 0.0
    assert loss_a == pytest.approx(loss_b)


def test_compute_val_loss_returns_zero_for_empty_loader() -> None:
    features = _tiny_item_features()
    item_tower = ItemTower(n_category_l1=2, n_category_l2=2, n_colour=2, n_dept=2, text_dim=4, embed_dim=8)
    user_tower = UserTower(n_age_band=1, n_region=1, embed_dim=8)
    loader = DataLoader(
        TwoTowerDataset([], {}), batch_size=2, collate_fn=make_collate_fn(pad_idx=2, session_length=3)
    )

    assert compute_val_loss(item_tower, user_tower, features, loader) == 0.0
