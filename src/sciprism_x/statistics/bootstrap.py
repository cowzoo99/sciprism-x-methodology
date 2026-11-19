from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    observed_difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    resamples: int
    confidence_level: float
    seed: int
    direction: Literal["higher", "lower"]
    pair_ids: tuple[str, ...]
    paired_differences: tuple[float, ...]
    bootstrap_differences: tuple[float, ...]


def paired_bootstrap(
    *,
    pair_ids: Sequence[str],
    proposed: np.ndarray,
    baseline: np.ndarray,
    resamples: int,
    confidence_level: float,
    seed: int,
    direction: Literal["higher", "lower"],
) -> BootstrapResult:
    """Paired ID-level percentile bootstrap with empirical two-sided p-value."""
    if resamples != 1000:
        raise ValueError("method contract requires exactly 1,000 bootstrap resamples")
    if confidence_level != 0.95:
        raise ValueError("method contract requires a 0.95 confidence level")
    ids = tuple(str(pair_id) for pair_id in pair_ids)
    proposed_values = np.asarray(proposed, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if not ids or any(not pair_id for pair_id in ids):
        raise ValueError("pair_ids must be non-empty strings")
    if proposed_values.shape != (len(ids),) or baseline_values.shape != (len(ids),):
        raise ValueError("paired values must be vectors aligned with pair_ids")
    if not np.isfinite(proposed_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("paired values must be finite")
    if direction not in ("higher", "lower"):
        raise ValueError("direction must be 'higher' or 'lower'")

    row_differences = (
        proposed_values - baseline_values
        if direction == "higher"
        else baseline_values - proposed_values
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair_id, difference in zip(ids, row_differences, strict=True):
        grouped[pair_id].append(float(difference))
    unique_ids = tuple(grouped)
    differences = np.asarray(
        [np.mean(grouped[pair_id], dtype=np.float64) for pair_id in unique_ids],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        differences.size,
        size=(resamples, differences.size),
        endpoint=False,
    )
    bootstrap = differences[indices].mean(axis=1, dtype=np.float64)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap,
        [tail, 1.0 - tail],
        method="linear",
    )
    p_value = min(
        1.0,
        2.0
        * min(
            float(np.mean(bootstrap <= 0.0, dtype=np.float64)),
            float(np.mean(bootstrap >= 0.0, dtype=np.float64)),
        ),
    )
    return BootstrapResult(
        observed_difference=float(np.mean(differences, dtype=np.float64)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        p_value=p_value,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
        direction=direction,
        pair_ids=unique_ids,
        paired_differences=tuple(float(value) for value in differences),
        bootstrap_differences=tuple(float(value) for value in bootstrap),
    )
