import math

import pytest

from sciprism_x.formal.boundaries import BoundaryEvaluation, compute_boundary_rows


def _known_crossing_series() -> tuple[BoundaryEvaluation, ...]:
    rows = []
    for radius, regime in ((0.25, 0), (0.50, 1), (1.00, 2)):
        rows.append(
            BoundaryEvaluation(
                case_id="case-a",
                family="evidence_removal",
                seed=42,
                radius=radius,
                original_regime=0,
                perturbed_regime=regime,
                valid=True,
            )
        )
    for radius in (0.25, 0.50, 1.00):
        rows.append(
            BoundaryEvaluation(
                case_id="case-b",
                family="context_composition_shift",
                seed=42,
                radius=radius,
                original_regime=1,
                perturbed_regime=1,
                valid=True,
            )
        )
    return tuple(rows)


def test_boundary_rows_are_computed_from_regime_crossings() -> None:
    rows = compute_boundary_rows(_known_crossing_series(), rho_max=1.0)

    assert rows[0].crossed is True
    assert rows[0].rho_star == pytest.approx(0.5)
    assert rows[0].bsi_raw == pytest.approx(0.5)
    assert rows[0].bsi_stability == pytest.approx(0.5)
    assert rows[1].right_censored is True
    assert rows[1].rho_star == pytest.approx(1.0)
    assert rows[1].bsi_raw == pytest.approx(0.0)
    assert rows[1].bsi_stability == pytest.approx(1.0)
    assert all(math.isfinite(row.bsi_raw) for row in rows)


def test_invalid_perturbation_is_rejected() -> None:
    invalid = _known_crossing_series()[0]

    with pytest.raises(ValueError, match="validity-preserving"):
        compute_boundary_rows((invalid.__class__(
            case_id=invalid.case_id,
            family=invalid.family,
            seed=invalid.seed,
            radius=invalid.radius,
            original_regime=invalid.original_regime,
            perturbed_regime=invalid.perturbed_regime,
            valid=False,
        ),), rho_max=1.0)
