from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    index: int
    lower: float
    upper: float
    count: int
    fraction: float
    mean_confidence: float | None
    observed_validity: float | None
    absolute_gap: float | None
    sparse: bool


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    sc: float
    ece: float
    bins: tuple[CalibrationBin, ...]


def compute_sc(
    *,
    confidence: np.ndarray,
    structural_validity: np.ndarray,
    bins: int,
    minimum_bin_samples: int,
    minimum_bin_fraction: float,
) -> CalibrationResult:
    """Compute Appendix E.2 structural calibration as one minus ECE."""
    predictions = np.asarray(confidence, dtype=np.float64)
    observed = np.asarray(structural_validity, dtype=np.float64)
    if predictions.ndim != 1 or observed.shape != predictions.shape or predictions.size == 0:
        raise ValueError("confidence and structural_validity must be non-empty equal 1-D shapes")
    if not np.isfinite(predictions).all() or np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.isin(observed, [0.0, 1.0]).all():
        raise ValueError("structural_validity must contain only zero or one")
    if bins <= 0 or minimum_bin_samples < 0:
        raise ValueError("bin counts must be valid")
    if not 0.0 <= minimum_bin_fraction <= 1.0:
        raise ValueError("minimum_bin_fraction must be in [0, 1]")

    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    assignments = np.minimum(np.floor(predictions * bins).astype(int), bins - 1)
    results: list[CalibrationBin] = []
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        fraction = count / predictions.size
        sparse = count < minimum_bin_samples or fraction < minimum_bin_fraction
        if count:
            mean_confidence = float(predictions[mask].mean())
            observed_validity = float(observed[mask].mean())
            gap = abs(mean_confidence - observed_validity)
            ece += fraction * gap
        else:
            mean_confidence = None
            observed_validity = None
            gap = None
        results.append(
            CalibrationBin(
                index=index,
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                fraction=fraction,
                mean_confidence=mean_confidence,
                observed_validity=observed_validity,
                absolute_gap=gap,
                sparse=sparse,
            )
        )
    return CalibrationResult(sc=1.0 - ece, ece=ece, bins=tuple(results))
