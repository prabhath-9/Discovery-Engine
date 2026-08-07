from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
ITEM_VECTORS_PATH = ARTIFACTS_DIR / "item_vectors.npy"
INDEX_PATH = ARTIFACTS_DIR / "index.faiss"

HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64


def build_index(
    vectors: np.ndarray,
    m: int = HNSW_M,
    ef_construction: int = EF_CONSTRUCTION,
    ef_search: int = EF_SEARCH,
) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def save_index(index: faiss.Index, path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def main() -> None:
    vectors = np.load(ITEM_VECTORS_PATH)
    index = build_index(vectors)
    save_index(index)
    print(f"index built: {index.ntotal} vectors, dim {vectors.shape[1]}")


if __name__ == "__main__":
    main()
