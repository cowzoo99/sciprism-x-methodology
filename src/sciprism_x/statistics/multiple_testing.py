from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HolmResult:
    comparison_id: str
    raw_p: float
    adjusted_p: float
    reject: bool


def holm_bonferroni(
    *, comparisons: Sequence[tuple[str, float]], alpha: float
) -> tuple[HolmResult, ...]:
    """Adjust one saved table/figure comparison family with Holm's procedure."""
    values = tuple((str(name), float(p_value)) for name, p_value in comparisons)
    if not values or any(not name for name, _p_value in values):
        raise ValueError("comparisons must contain non-empty IDs")
    if len({name for name, _p_value in values}) != len(values):
        raise ValueError("comparison IDs must be unique")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if any(not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0 for _, p_value in values):
        raise ValueError("raw p-values must be finite and in [0, 1]")

    order = sorted(range(len(values)), key=lambda index: (values[index][1], values[index][0]))
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index][1]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return tuple(
        HolmResult(
            comparison_id=name,
            raw_p=p_value,
            adjusted_p=adjusted[index],
            reject=adjusted[index] <= alpha,
        )
        for index, (name, p_value) in enumerate(values)
    )
