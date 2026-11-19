from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sciprism_x.open_set.calibration import CalibrationResult, compute_sc
from sciprism_x.open_set.grouping import compute_updq
from sciprism_x.residuals.scores import compute_pcs, compute_tsr


def compute_ccs(
    reference_structural_scores: Sequence[float],
    context_structural_scores: Sequence[float],
) -> float:
    """Appendix E.2: one minus mean absolute matched-context score change."""
    reference = np.asarray(tuple(reference_structural_scores), dtype=np.float64)
    contexts = np.asarray(tuple(context_structural_scores), dtype=np.float64)
    if reference.ndim != 1 or contexts.shape != reference.shape or reference.size == 0:
        raise ValueError("CCS inputs must be non-empty equal one-dimensional sequences")
    if not np.isfinite(reference).all() or not np.isfinite(contexts).all():
        raise ValueError("CCS inputs must be finite")
    return float(1.0 - np.mean(np.abs(reference - contexts), dtype=np.float64))


def compute_bsi(*, rho_star: float, rho_max: float) -> float:
    """Appendix E.2: 1 - min(ρ*, ρmax) / ρmax."""
    values = np.asarray([rho_star, rho_max], dtype=np.float64)
    if not np.isfinite(values).all() or rho_star < 0.0 or rho_max <= 0.0:
        raise ValueError("rho_star must be nonnegative and rho_max must be positive")
    return float(1.0 - min(rho_star, rho_max) / rho_max)


__all__ = [
    "CalibrationResult",
    "compute_bsi",
    "compute_ccs",
    "compute_pcs",
    "compute_sc",
    "compute_tsr",
    "compute_updq",
]
