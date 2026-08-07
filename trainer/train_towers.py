from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
ITEMS_PATH = PROCESSED_DIR / "items.parquet"
USERS_PATH = PROCESSED_DIR / "users.parquet"

ARTIFACTS_DIR = Path("artifacts")
ITEM_TEXT_PATH = ARTIFACTS_DIR / "item_text.npy"
TOWERS_PATH = ARTIFACTS_DIR / "towers.pt"
ITEM_VECTORS_PATH = ARTIFACTS_DIR / "item_vectors.npy"
ID_MAP_PATH = ARTIFACTS_DIR / "id_map.json"

CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
CLIP_BATCH_SIZE = 256

EMBED_DIM = 128
CATEGORICAL_EMBED_DIM = 32
DEMO_EMBED_DIM = 16
SESSION_LENGTH = 20
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 5
WARMUP_FRACTION = 0.1
TEMPERATURE = 0.05

Example = tuple[str, list[int], int]


def encode_item_text(items: pl.DataFrame, path: Path = ITEM_TEXT_PATH) -> np.ndarray:
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(CLIP_MODEL_NAME)
    titles = items["title"].fill_null("").to_list()
    embeddings = model.encode(titles, batch_size=CLIP_BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings


def build_vocab(values: list[str]) -> dict[str, int]:
    return {value: idx for idx, value in enumerate(sorted(set(values)))}


@dataclass
class ItemFeatures:
    category_l1: torch.Tensor
    category_l2: torch.Tensor
    colour: torch.Tensor
    dept: torch.Tensor
    text: torch.Tensor


def build_item_features(
    items: pl.DataFrame,
    text_embeddings: np.ndarray,
    category_l1_vocab: dict[str, int],
    category_l2_vocab: dict[str, int],
    colour_vocab: dict[str, int],
    dept_vocab: dict[str, int],
) -> ItemFeatures:
    def _idx(col: str, vocab: dict[str, int]) -> torch.Tensor:
        values = items[col].fill_null("unknown").to_list()
        # pad row (appended index) reuses vocab index 0; it is always masked out before use
        return torch.tensor([vocab[v] for v in values] + [0], dtype=torch.long)

    pad_row = np.zeros((1, text_embeddings.shape[1]), dtype=np.float32)
    text = np.vstack([text_embeddings, pad_row])

    return ItemFeatures(
        category_l1=_idx("category_l1", category_l1_vocab),
        category_l2=_idx("category_l2", category_l2_vocab),
        colour=_idx("colour", colour_vocab),
        dept=_idx("dept", dept_vocab),
        text=torch.tensor(text, dtype=torch.float32),
    )


class ItemTower(nn.Module):
    def __init__(
        self,
        n_category_l1: int,
        n_category_l2: int,
        n_colour: int,
        n_dept: int,
        text_dim: int,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.category_l1_emb = nn.Embedding(n_category_l1, CATEGORICAL_EMBED_DIM)
        self.category_l2_emb = nn.Embedding(n_category_l2, CATEGORICAL_EMBED_DIM)
        self.colour_emb = nn.Embedding(n_colour, CATEGORICAL_EMBED_DIM)
        self.dept_emb = nn.Embedding(n_dept, CATEGORICAL_EMBED_DIM)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        input_dim = CATEGORICAL_EMBED_DIM * 4 + embed_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, embed_dim))

    def forward(
        self,
        category_l1: torch.Tensor,
        category_l2: torch.Tensor,
        colour: torch.Tensor,
        dept: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.category_l1_emb(category_l1),
                self.category_l2_emb(category_l2),
                self.colour_emb(colour),
                self.dept_emb(dept),
                self.text_proj(text_embedding),
            ],
            dim=-1,
        )
        return nn.functional.normalize(self.mlp(features), dim=-1)


def masked_mean_pool(item_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    summed = (item_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class UserTower(nn.Module):
    def __init__(self, n_age_band: int, n_region: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.age_band_emb = nn.Embedding(n_age_band, DEMO_EMBED_DIM)
        self.region_emb = nn.Embedding(n_region, DEMO_EMBED_DIM)
        input_dim = embed_dim + DEMO_EMBED_DIM * 2
        self.mlp = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, embed_dim))

    def forward(
        self,
        history_item_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        age_band: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        pooled = masked_mean_pool(history_item_embeddings, history_mask)
        demo = torch.cat([self.age_band_emb(age_band), self.region_emb(region)], dim=-1)
        return nn.functional.normalize(self.mlp(torch.cat([pooled, demo], dim=-1)), dim=-1)


def warmup_cosine_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / remaining)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def two_tower_loss(user_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    logits = user_emb @ item_emb.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def build_sequences(train: pl.DataFrame, article_id_to_idx: dict[int, int]) -> dict[str, list[int]]:
    grouped = train.sort("t_dat").group_by("customer_id", maintain_order=True).agg(pl.col("article_id"))
    return {
        customer_id: [article_id_to_idx[a] for a in articles]
        for customer_id, articles in zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list())
    }


def build_train_examples(sequences: dict[str, list[int]], session_length: int = SESSION_LENGTH) -> list[Example]:
    examples: list[Example] = []
    for customer_id, items in sequences.items():
        for i in range(1, len(items)):
            history = items[max(0, i - session_length) : i]
            examples.append((customer_id, history, items[i]))
    return examples


def build_val_examples(
    train_sequences: dict[str, list[int]],
    val: pl.DataFrame,
    article_id_to_idx: dict[int, int],
    session_length: int = SESSION_LENGTH,
) -> list[Example]:
    examples: list[Example] = []
    for customer_id, article_id in zip(val["customer_id"].to_list(), val["article_id"].to_list()):
        history = train_sequences.get(customer_id)
        item_idx = article_id_to_idx.get(article_id)
        if not history or item_idx is None:
            continue
        examples.append((customer_id, history[-session_length:], item_idx))
    return examples


class TwoTowerDataset(Dataset):
    def __init__(self, examples: list[Example], demo_by_customer: dict[str, tuple[int, int]]) -> None:
        self.examples = examples
        self.demo_by_customer = demo_by_customer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], int, int, int]:
        customer_id, history, target = self.examples[idx]
        age_band_idx, region_idx = self.demo_by_customer[customer_id]
        return history, target, age_band_idx, region_idx


def pad_and_mask(history: list[int], length: int, pad_idx: int) -> tuple[list[int], list[float]]:
    trimmed = history[-length:]
    padding = length - len(trimmed)
    return trimmed + [pad_idx] * padding, [1.0] * len(trimmed) + [0.0] * padding


def make_collate_fn(
    pad_idx: int, session_length: int = SESSION_LENGTH
) -> Callable[[list[tuple[list[int], int, int, int]]], tuple[torch.Tensor, ...]]:
    def collate(batch: list[tuple[list[int], int, int, int]]) -> tuple[torch.Tensor, ...]:
        histories, targets, age_bands, regions = zip(*batch)
        padded, masks = zip(*(pad_and_mask(h, session_length, pad_idx) for h in histories))
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(age_bands, dtype=torch.long),
            torch.tensor(regions, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    return collate


def gather_item_embeddings(item_tower: ItemTower, item_features: ItemFeatures, indices: torch.Tensor) -> torch.Tensor:
    unique_idx, inverse = torch.unique(indices, return_inverse=True)
    embeddings = item_tower(
        item_features.category_l1[unique_idx],
        item_features.category_l2[unique_idx],
        item_features.colour[unique_idx],
        item_features.dept[unique_idx],
        item_features.text[unique_idx],
    )
    return embeddings[inverse]


def run_batch(
    item_tower: ItemTower,
    user_tower: UserTower,
    item_features: ItemFeatures,
    batch: tuple[torch.Tensor, ...],
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    hist_idx, mask, age_band_idx, region_idx, target_idx = batch
    batch_size, length = hist_idx.shape
    flat_emb = gather_item_embeddings(item_tower, item_features, torch.cat([hist_idx.reshape(-1), target_idx]))
    history_emb = flat_emb[: batch_size * length].view(batch_size, length, -1)
    target_emb = flat_emb[batch_size * length :]
    user_emb = user_tower(history_emb, mask, age_band_idx, region_idx)
    return two_tower_loss(user_emb, target_emb, temperature)


def compute_val_loss(
    item_tower: ItemTower, user_tower: UserTower, item_features: ItemFeatures, val_loader: DataLoader
) -> float:
    item_tower.eval()
    user_tower.eval()
    with torch.no_grad():
        losses = [run_batch(item_tower, user_tower, item_features, batch).item() for batch in val_loader]
    return float(np.mean(losses)) if losses else 0.0


GRAD_CLIP_NORM = 5.0


def main() -> None:
    torch.manual_seed(0)

    items = pl.read_parquet(ITEMS_PATH)
    users = pl.read_parquet(USERS_PATH)
    train = pl.read_parquet(TRAIN_PATH)
    val = pl.read_parquet(VAL_PATH)

    text_embeddings = encode_item_text(items)

    article_ids = items["article_id"].to_list()
    article_id_to_idx = {article_id: idx for idx, article_id in enumerate(article_ids)}
    n_items = len(article_ids)

    category_l1_vocab = build_vocab(items["category_l1"].fill_null("unknown").to_list())
    category_l2_vocab = build_vocab(items["category_l2"].fill_null("unknown").to_list())
    colour_vocab = build_vocab(items["colour"].fill_null("unknown").to_list())
    dept_vocab = build_vocab(items["dept"].fill_null("unknown").to_list())
    item_features = build_item_features(
        items, text_embeddings, category_l1_vocab, category_l2_vocab, colour_vocab, dept_vocab
    )

    age_band_vocab = build_vocab(users["age_band"].fill_null("unknown").to_list())
    region_vocab = build_vocab(users["region"].fill_null("unknown").to_list())
    demo_by_customer = {
        customer_id: (age_band_vocab[age_band], region_vocab[region])
        for customer_id, age_band, region in zip(
            users["customer_id"].to_list(),
            users["age_band"].fill_null("unknown").to_list(),
            users["region"].fill_null("unknown").to_list(),
        )
    }

    train_sequences = build_sequences(train, article_id_to_idx)
    train_examples = build_train_examples(train_sequences)
    val_examples = build_val_examples(train_sequences, val, article_id_to_idx)
    print(f"train examples: {len(train_examples)}, val examples: {len(val_examples)}, items: {n_items}")

    pad_idx = n_items
    collate = make_collate_fn(pad_idx)
    train_loader = DataLoader(
        TwoTowerDataset(train_examples, demo_by_customer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        TwoTowerDataset(val_examples, demo_by_customer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    item_tower = ItemTower(
        len(category_l1_vocab), len(category_l2_vocab), len(colour_vocab), len(dept_vocab), text_embeddings.shape[1]
    )
    user_tower = UserTower(len(age_band_vocab), len(region_vocab))
    optimizer = torch.optim.Adam(list(item_tower.parameters()) + list(user_tower.parameters()), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_cosine_decay(step, total_steps, warmup_steps)
    )

    initial_val_loss = compute_val_loss(item_tower, user_tower, item_features, val_loader)
    print(f"initial val_loss={initial_val_loss:.4f}")

    best_val_loss = initial_val_loss
    best_item_state = copy.deepcopy(item_tower.state_dict())
    best_user_state = copy.deepcopy(user_tower.state_dict())

    for epoch in range(1, EPOCHS + 1):
        item_tower.train()
        user_tower.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = run_batch(item_tower, user_tower, item_features, batch)
            loss.backward()
            # the temperature-0.05 contrastive loss produces sharp gradients that can spike Adam updates
            nn.utils.clip_grad_norm_(
                list(item_tower.parameters()) + list(user_tower.parameters()), GRAD_CLIP_NORM
            )
            optimizer.step()
            scheduler.step()

        val_loss = compute_val_loss(item_tower, user_tower, item_features, val_loader)
        print(f"epoch {epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_item_state = copy.deepcopy(item_tower.state_dict())
            best_user_state = copy.deepcopy(user_tower.state_dict())

    assert best_val_loss < initial_val_loss, f"best val loss {best_val_loss} not below initial {initial_val_loss}"

    item_tower.load_state_dict(best_item_state)
    user_tower.load_state_dict(best_user_state)

    item_tower.eval()
    with torch.no_grad():
        final_item_vectors = gather_item_embeddings(item_tower, item_features, torch.arange(n_items)).numpy()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ITEM_VECTORS_PATH, final_item_vectors.astype(np.float32))
    with ID_MAP_PATH.open("w") as f:
        json.dump(
            {
                "article_id_to_index": {str(a): i for a, i in article_id_to_idx.items()},
                "index_to_article_id": {str(i): a for a, i in article_id_to_idx.items()},
            },
            f,
        )

    torch.save(
        {
            "item_tower_state": item_tower.state_dict(),
            "user_tower_state": user_tower.state_dict(),
            "category_l1_vocab": category_l1_vocab,
            "category_l2_vocab": category_l2_vocab,
            "colour_vocab": colour_vocab,
            "dept_vocab": dept_vocab,
            "age_band_vocab": age_band_vocab,
            "region_vocab": region_vocab,
            "config": {
                "embedding_dim": EMBED_DIM,
                "categorical_embed_dim": CATEGORICAL_EMBED_DIM,
                "demo_embed_dim": DEMO_EMBED_DIM,
                "session_length": SESSION_LENGTH,
                "text_dim": int(text_embeddings.shape[1]),
            },
        },
        TOWERS_PATH,
    )


if __name__ == "__main__":
    main()
