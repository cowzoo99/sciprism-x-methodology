from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_tp(transport_plan: np.ndarray, frozen_reference: np.ndarray) -> float:
    """Transport mass on the frozen reference correspondence set."""
    plan = np.asarray(transport_plan, dtype=np.float64)
    reference = np.asarray(frozen_reference, dtype=np.bool_)
    if plan.ndim != 2 or reference.shape != plan.shape or plan.size == 0:
        raise ValueError("transport plan and frozen reference must have one equal matrix shape")
    if not np.isfinite(plan).all() or np.any(plan < 0.0):
        raise ValueError("transport plan must be finite and nonnegative")
    total = float(plan.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("transport plan must carry positive mass")
    return float(plan[reference].sum(dtype=np.float64) / total)


def _validated_ranks(ranks: Sequence[int]) -> np.ndarray:
    values = np.asarray(tuple(ranks), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ranks must be a non-empty finite one-dimensional sequence")
    if np.any(values < 1.0) or np.any(values != np.floor(values)):
        raise ValueError("ranks must be positive integers")
    return values


def compute_hits_at_1(ranks: Sequence[int]) -> float:
    values = _validated_ranks(ranks)
    return float(np.mean(values == 1.0, dtype=np.float64))


def compute_mrr(ranks: Sequence[int]) -> float:
    values = _validated_ranks(ranks)
    return float(np.mean(1.0 / values, dtype=np.float64))


def compute_scs(structural_correspondence_scores: Sequence[float]) -> float:
    values = np.asarray(tuple(structural_correspondence_scores), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("SCS requires a non-empty finite score sequence")
    return float(np.mean(values, dtype=np.float64))
