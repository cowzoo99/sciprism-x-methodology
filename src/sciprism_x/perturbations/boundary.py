from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    rho_star: float
    bsi: float
    right_censored: bool
    evaluated_radii: tuple[float, ...]


def fit_mad_scale(validation_deltas: np.ndarray, floor: float) -> np.ndarray:
    """Fit per-dimension validation MAD without a Gaussian consistency factor."""
    values = np.asarray(validation_deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("validation_deltas must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("validation_deltas must be finite")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and positive")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    return np.maximum(mad, floor)


def scientifically_scaled_norm(
    delta: np.ndarray,
    mad_scale: np.ndarray,
    scientific_weights: np.ndarray,
) -> float:
    """Compute the paper's quadratic perturbation norm δᵀWδ after MAD scaling."""
    vector = np.asarray(delta, dtype=np.float64)
    scale = np.asarray(mad_scale, dtype=np.float64)
    weights = np.asarray(scientific_weights, dtype=np.float64)
    if vector.ndim != 1 or scale.shape != vector.shape or weights.shape != vector.shape:
        raise ValueError("delta, mad_scale, and scientific_weights must be equal 1-D shapes")
    if not np.isfinite(vector).all() or not np.isfinite(scale).all():
        raise ValueError("delta and mad_scale must be finite")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("scientific_weights must be finite and nonnegative")
    if np.any(scale <= 0.0):
        raise ValueError("mad_scale must be positive")
    standardized = vector / scale
    return float(np.sum(weights * standardized * standardized, dtype=np.float64))


def find_boundary(
    radii: Sequence[float],
    is_valid: Callable[[float], bool],
    rho_max: float,
) -> BoundaryResult:
    """Find the first radius that changes the frozen diagnostic assignment.

    ``is_valid`` returns True while the original regime or diagnostic state is
    preserved. If it never changes, the observation is right-censored at
    ``rho_max`` as specified in Appendix C.3 and E.2.
    """
    ordered = tuple(float(radius) for radius in radii)
    if not np.isfinite(rho_max) or rho_max <= 0.0:
        raise ValueError("rho_max must be finite and positive")
    if not ordered:
        raise ValueError("radii must not be empty")
    if any(not np.isfinite(radius) or radius <= 0.0 for radius in ordered):
        raise ValueError("radii must be finite and positive")
    if any(left >= right for left, right in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("radii must be strictly increasing")
    if ordered[-1] > rho_max:
        raise ValueError("radii cannot exceed rho_max")

    evaluated: list[float] = []
    for radius in ordered:
        evaluated.append(radius)
        if not bool(is_valid(radius)):
            rho_star = radius
            bsi = 1.0 - min(rho_star, rho_max) / rho_max
            return BoundaryResult(rho_star, bsi, False, tuple(evaluated))

    return BoundaryResult(rho_max, 0.0, True, tuple(evaluated))
