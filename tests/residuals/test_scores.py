import math

import numpy as np
import pytest


def test_pcs_uses_printed_positive_node_sign() -> None:
    from sciprism_x.residuals.scores import compute_pcs

    actual = compute_pcs(
        node_loss=0.2,
        structure_loss=0.3,
        constraint_loss=0.4,
        eta_node=1.0,
        eta_structure=2.0,
        eta_constraint=3.0,
    )
    expected = math.exp(0.2 - 2.0 * 0.3 - 3.0 * 0.4)
    assert actual == pytest.approx(expected)


def test_residual_consistency_is_literal_exponential() -> None:
    from sciprism_x.residuals.scores import compute_residual_consistency

    residual = np.array([0.1, 0.2, 0.3, 0.0, 0.4, 0.5])
    beta = np.full(6, -0.5)
    assert compute_residual_consistency(residual, beta) == pytest.approx(
        math.exp(float(beta @ residual))
    )


def test_direction_audit_warns_without_changing_scores() -> None:
    from sciprism_x.residuals.scores import direction_audit

    warnings = direction_audit(
        eta_node=1.0,
        eta_structure=2.0,
        eta_constraint=3.0,
        beta=np.full(6, 0.5),
    )

    assert "pcs_node_loss_increases_score" in warnings
    assert "positive_beta_increases_consistency_with_residual" in warnings


def test_tsr_is_mean_of_binary_endpoint_success() -> None:
    from sciprism_x.residuals.scores import compute_tsr

    assert compute_tsr((True, False, True, True)) == 0.75
    with pytest.raises(ValueError):
        compute_tsr(())
