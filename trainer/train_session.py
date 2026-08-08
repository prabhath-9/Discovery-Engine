from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.session.encoder import (
    COLLAPSE_THRESHOLD,
    EMBED_DIM,
    N_HEADS,
    N_INTENTS,
    N_LAYERS,
    ORTHOGONALITY_WEIGHT,
    SESSION_LENGTH,
    SessionEncoder,
    orthogonality_penalty,
    pairwise_head_cosine,
)
from trainer.evaluate import RecommendFn, _build_histories, evaluate, measure_latency_p50_ms, write_results_row
from trainer.train_towers import Example, build_sequences, build_train_examples, build_val_examples, pad_and_mask

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"

ARTIFACTS_DIR = Path("artifacts")
ITEM_VECTORS_PATH = ARTIFACTS_DIR / "item_vectors.npy"
ID_MAP_PATH = ARTIFACTS_DIR / "id_map.json"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"
SESSION_ENCODER_PATH = ARTIFACTS_DIR / "session_encoder.pt"

BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 3
TEMPERATURE = 0.05
DROPOUT = 0.1


class SessionDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        _, history, target = self.examples[idx]
        return history, target


def make_collate_fn(item_vectors_ext: np.ndarray, pad_idx: int, session_length: int = SESSION_LENGTH):
    def collate(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        histories, targets = zip(*batch)
        padded, masks = zip(*(pad_and_mask(h, session_length, pad_idx) for h in histories))
        embeddings = item_vectors_ext[np.array(padded)]
        return (
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(masks, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        )

    return collate


def session_loss(
    intents: torch.Tensor, weights: torch.Tensor, target_emb: torch.Tensor, temperature: float = TEMPERATURE
) -> torch.Tensor:
    batch_size, k, _ = intents.shape
    labels = torch.arange(batch_size, device=intents.device)
    per_head = torch.stack(
        [
            nn.functional.cross_entropy(intents[:, head, :] @ target_emb.T / temperature, labels, reduction="none")
            for head in range(k)
        ],
        dim=1,
    )
    return (per_head * weights).sum(dim=1).mean()


def run_batch(
    encoder: SessionEncoder, item_vectors_ext: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings, mask, target_idx = batch
    intents, weights = encoder(embeddings, mask)
    target_emb = item_vectors_ext[target_idx]
    loss = session_loss(intents, weights, target_emb) + ORTHOGONALITY_WEIGHT * orthogonality_penalty(intents)
    return loss, intents


def make_multi_intent_recommend_fn(
    encoder: SessionEncoder,
    item_vectors: np.ndarray,
    article_id_to_index: dict[int, int],
    index_to_article_id: dict[int, int],
    index: faiss.Index,
    session_length: int = SESSION_LENGTH,
    k: int = 20,
) -> RecommendFn:
    pad_idx = item_vectors.shape[0]
    item_vectors_ext = np.vstack([item_vectors, np.zeros((1, item_vectors.shape[1]), dtype=np.float32)])

    def recommend(user_id: str, history: list[int]) -> list[int]:
        recent = history[-session_length:]
        idxs = [article_id_to_index[a] for a in recent if a in article_id_to_index]
        if not idxs:
            return []

        padded, mask = pad_and_mask(idxs, session_length, pad_idx)
        embeddings = torch.tensor(item_vectors_ext[padded], dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            intents, weights = encoder(embeddings, mask_t)

        seen = set(history)
        per_head_k = min(index.ntotal, k + len(seen))
        scored: dict[int, float] = {}
        for head in range(intents.shape[1]):
            query = intents[0, head].numpy().reshape(1, -1).astype(np.float32)
            head_weight = float(weights[0, head])
            scores, ids = index.search(query, per_head_k)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                article_id = index_to_article_id[int(idx)]
                if article_id in seen:
                    continue
                weighted_score = float(score) * head_weight
                if article_id not in scored or weighted_score > scored[article_id]:
                    scored[article_id] = weighted_score

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return [article_id for article_id, _ in ranked[:k]]

    return recommend


def main() -> None:
    torch.manual_seed(0)

    train = pl.read_parquet(TRAIN_PATH)
    val = pl.read_parquet(VAL_PATH)

    item_vectors = np.load(ITEM_VECTORS_PATH)
    with ID_MAP_PATH.open() as f:
        id_map = json.load(f)
    article_id_to_index = {int(k): v for k, v in id_map["article_id_to_index"].items()}
    index_to_article_id = {int(k): v for k, v in id_map["index_to_article_id"].items()}

    n_items = item_vectors.shape[0]
    pad_idx = n_items
    item_vectors_ext = np.vstack([item_vectors, np.zeros((1, item_vectors.shape[1]), dtype=np.float32)])
    item_vectors_ext_t = torch.tensor(item_vectors_ext, dtype=torch.float32)

    sequences = build_sequences(train, article_id_to_index)
    train_examples = build_train_examples(sequences)
    val_examples = build_val_examples(sequences, val, article_id_to_index)
    print(f"train examples: {len(train_examples)}, val examples: {len(val_examples)}")

    collate = make_collate_fn(item_vectors_ext, pad_idx)
    train_loader = DataLoader(
        SessionDataset(train_examples), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        SessionDataset(val_examples), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0
    )

    encoder = SessionEncoder(
        embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS, session_length=SESSION_LENGTH, n_intents=N_INTENTS,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = run_batch(encoder, item_vectors_ext_t, batch)
            loss.backward()
            optimizer.step()

        encoder.eval()
        val_losses: list[float] = []
        pairwise_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                loss, intents = run_batch(encoder, item_vectors_ext_t, batch)
                val_losses.append(loss.item())
                pairwise_batches.append(pairwise_head_cosine(intents))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        if pairwise_batches:
            pairwise = torch.cat(pairwise_batches, dim=0)
            pair_means = pairwise.mean(dim=0)
            mean_cosine = pair_means.mean().item()
            max_pair_cosine = pair_means.max().item()
        else:
            mean_cosine = 0.0
            max_pair_cosine = 0.0

        print(f"epoch {epoch} val_loss={val_loss:.4f} mean_pairwise_cosine={mean_cosine:.4f}")
        if max_pair_cosine > COLLAPSE_THRESHOLD:
            print(f"!!! COLLAPSE WARNING: max pairwise cosine {max_pair_cosine:.4f} exceeds {COLLAPSE_THRESHOLD} !!!")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "config": {
                "embed_dim": EMBED_DIM,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "session_length": SESSION_LENGTH,
                "n_intents": N_INTENTS,
                "dropout": DROPOUT,
            },
        },
        SESSION_ENCODER_PATH,
    )

    index = faiss.read_index(str(INDEX_PATH))
    recommend_fn = make_multi_intent_recommend_fn(
        encoder, item_vectors, article_id_to_index, index_to_article_id, index
    )
    metrics = evaluate(recommend_fn, val, k=20)
    histories = list(_build_histories(TRAIN_PATH).values())
    latency_p50_ms = measure_latency_p50_ms(recommend_fn, histories[:200])

    print(
        f"Multi-intent | recall@20={metrics['recall@20']:.4f} ndcg@20={metrics['ndcg@20']:.4f} "
        f"coverage={metrics['catalog_coverage']:.4f} latency_p50={latency_p50_ms:.2f}ms"
    )
    write_results_row("Multi-intent", metrics, latency_p50_ms)


if __name__ == "__main__":
    main()
