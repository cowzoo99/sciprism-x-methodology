from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from sciprism_x.perturbations.boundary import BoundaryResult, find_boundary


@dataclass(frozen=True, slots=True)
class BoundaryEvaluation:
    case_id: str
    family: str
    seed: int
    radius: float
    original_regime: int
    perturbed_regime: int
    valid: bool


@dataclass(frozen=True, slots=True)
class BoundaryMetricRow:
    case_id: str
    family: str
    seed: int
    crossed: bool
    rho_star: float
    rho_max: float
    bsi_raw: float
    bsi_stability: float
    right_censored: bool
    evaluated_radii: tuple[float, ...]


def boundary_metric(result: BoundaryResult, *, rho_max: float) -> tuple[float, float]:
    bounded_rho = min(result.rho_star, rho_max)
    bsi_raw = 1.0 - bounded_rho / rho_max
    return bsi_raw, 1.0 - bsi_raw


def compute_boundary_rows(
    evaluations: tuple[BoundaryEvaluation, ...],
    *,
    rho_max: float,
) -> tuple[BoundaryMetricRow, ...]:
    if not evaluations:
        raise ValueError("boundary computation requires evaluated perturbations")
    if not math.isfinite(rho_max) or rho_max <= 0.0:
        raise ValueError("rho_max must be finite and positive")
    if any(not row.valid for row in evaluations):
        raise ValueError("BSI requires validity-preserving perturbations")
    grouped: dict[tuple[str, str, int], list[BoundaryEvaluation]] = defaultdict(list)
    for row in evaluations:
        grouped[(row.case_id, row.family, row.seed)].append(row)

    output: list[BoundaryMetricRow] = []
    for (case_id, family, seed), group in sorted(grouped.items()):
        original_regimes = {row.original_regime for row in group}
        if len(original_regimes) != 1:
            raise ValueError("boundary group has inconsistent original regimes")
        ordered = sorted(group, key=lambda row: row.radius)
        radii = tuple(row.radius for row in ordered)
        if len(radii) != len(set(radii)):
            raise ValueError("boundary group contains duplicate radii")
        original = next(iter(original_regimes))
        regime_by_radius = {row.radius: row.perturbed_regime for row in ordered}
        result = find_boundary(
            radii,
            is_valid=lambda radius: regime_by_radius[radius] == original,
            rho_max=rho_max,
        )
        bsi_raw, bsi_stability = boundary_metric(result, rho_max=rho_max)
        output.append(
            BoundaryMetricRow(
                case_id=case_id,
                family=family,
                seed=seed,
                crossed=not result.right_censored,
                rho_star=result.rho_star,
                rho_max=rho_max,
                bsi_raw=bsi_raw,
                bsi_stability=bsi_stability,
                right_censored=result.right_censored,
                evaluated_radii=result.evaluated_radii,
            )
        )
    return tuple(output)
