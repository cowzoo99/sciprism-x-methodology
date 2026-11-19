import math

import numpy as np
import pytest


def test_ccs_and_tp_hand_computed() -> None:
    from sciprism_x.metrics.scientific import compute_ccs
    from sciprism_x.metrics.structural import compute_tp

    assert compute_ccs([0.8, 0.6], [0.7, 0.5]) == pytest.approx(0.9)
    plan = np.array([[0.4, 0.1], [0.2, 0.3]], dtype=np.float64)
    frozen = np.array([[True, False], [False, True]])
    assert compute_tp(plan, frozen) == pytest.approx(0.7)


def test_complete_scientific_metric_formulas() -> None:
    from sciprism_x.metrics.scientific import (
        compute_bsi,
        compute_pcs,
        compute_sc,
        compute_tsr,
        compute_updq,
    )

    assert compute_tsr([True, False, True, True]) == pytest.approx(0.75)
    assert compute_pcs(
        node_loss=0.2,
        structure_loss=0.3,
        constraint_loss=0.4,
        eta_node=1.0,
        eta_structure=2.0,
        eta_constraint=3.0,
    ) == pytest.approx(math.exp(0.2 - 0.6 - 1.2))
    assert compute_bsi(rho_star=0.5, rho_max=1.0) == pytest.approx(0.5)
    assert compute_updq(
        purity=0.9,
        stability=0.8,
        separation=0.7,
        weights=(1 / 3, 1 / 3, 1 / 3),
    ) == pytest.approx(0.8)
    calibration = compute_sc(
        confidence=np.array([0.1, 0.4, 0.6, 0.9]),
        structural_validity=np.array([0, 1, 1, 1]),
        bins=2,
        minimum_bin_samples=20,
        minimum_bin_fraction=0.01,
    )
    assert calibration.sc == pytest.approx(0.75)
    assert len(calibration.bins) == 2
    assert all(bin_result.sparse for bin_result in calibration.bins)


def test_dbp15k_ranking_metrics_hand_computed() -> None:
    from sciprism_x.metrics.structural import (
        compute_hits_at_1,
        compute_mrr,
        compute_scs,
    )

    ranks = [1, 2, 4, 1]
    assert compute_hits_at_1(ranks) == pytest.approx(0.5)
    assert compute_mrr(ranks) == pytest.approx((1.0 + 0.5 + 0.25 + 1.0) / 4)
    assert compute_scs([0.8, 0.6, 1.0]) == pytest.approx(0.8)


def test_transport_purity_normalizes_nonunit_mass() -> None:
    from sciprism_x.metrics.structural import compute_tp

    plan = np.array([[2.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    frozen = np.eye(2, dtype=np.bool_)
    assert compute_tp(plan, frozen) == pytest.approx(0.75)
