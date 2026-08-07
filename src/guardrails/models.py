from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredItem:
    item_id: int
    category_l1: str
    score: float
