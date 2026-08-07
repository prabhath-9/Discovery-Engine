from __future__ import annotations

import math

from src.guardrails.models import ScoredItem


class GuardrailError(ValueError):
    pass


def _validate(candidates: list[ScoredItem], list_size: int, cap: float) -> None:
    if list_size < 0:
        raise GuardrailError(f"list_size must be >= 0, got {list_size}")
    if not 0 < cap <= 1:
        raise GuardrailError(f"cap must be in (0, 1], got {cap}")
    for item in candidates:
        if not item.category_l1:
            raise GuardrailError(f"item {item.item_id} missing category_l1")


def _max_pickable(available_by_category: dict[str, int], limit: int) -> int:
    return sum(min(available, limit) for available in available_by_category.values())


def enforce_diversity_cap(candidates: list[ScoredItem], list_size: int, cap: float = 0.35) -> list[ScoredItem]:
    _validate(candidates, list_size, cap)
    if list_size == 0 or not candidates:
        return []

    available_by_category: dict[str, int] = {}
    for item in candidates:
        available_by_category[item.category_l1] = available_by_category.get(item.category_l1, 0) + 1

    target = 0
    for m in range(list_size, 0, -1):
        if _max_pickable(available_by_category, math.floor(cap * m)) >= m:
            target = m
            break
    if target == 0:
        return []

    limit = math.floor(cap * target)
    counts: dict[str, int] = {}
    selected: list[ScoredItem] = []
    for item in candidates:
        if counts.get(item.category_l1, 0) < limit:
            selected.append(item)
            counts[item.category_l1] = counts.get(item.category_l1, 0) + 1
    return selected[:target]


def category_shares(items: list[ScoredItem]) -> dict[str, float]:
    if not items:
        return {}
    n = len(items)
    counts: dict[str, int] = {}
    for item in items:
        counts[item.category_l1] = counts.get(item.category_l1, 0) + 1
    return {category: count / n for category, count in counts.items()}


def violates_diversity_cap(items: list[ScoredItem], cap: float = 0.35) -> bool:
    return any(share > cap for share in category_shares(items).values())
