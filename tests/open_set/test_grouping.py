import numpy as np
import pytest


def test_unknownness_is_normalized_component_mean() -> None:
    from sciprism_x.open_set.grouping import compute_unknownness

    values = compute_unknownness(
        support_distance=np.array([0.0, 1.0]),
        residual_novelty=np.array([0.3, 0.6]),
        motif_direction_novelty=np.array([0.6, 0.2]),
        weights=(1 / 3, 1 / 3, 1 / 3),
    )

    np.testing.assert_allclose(values, [0.3, 0.6])


def test_novelty_normalization_uses_validation_support_only() -> None:
    from sciprism_x.open_set.grouping import compute_novelty_from_validation

    validation = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    candidates = np.array([[1.0], [20.0]], dtype=np.float64)
    result = compute_novelty_from_validation(
        candidate_states=candidates,
        candidate_residuals=candidates,
        candidate_motif_directions=candidates,
        validation_states=validation,
        validation_residuals=validation,
        validation_motif_directions=validation,
        mad_floor=1.0e-12,
        normalization_quantile=0.95,
    )

    assert result.support_distance == pytest.approx((0.0, 1.0))
    assert result.residual_novelty == pytest.approx((0.0, 1.0))
    assert result.motif_direction_novelty == pytest.approx((0.0, 1.0))


def test_cluster_unknowns_searches_counts_deterministically() -> None:
    from sciprism_x.open_set.grouping import cluster_unknowns

    state = np.array(
        [[0.0], [0.1], [-0.1], [10.0], [10.1], [9.9]],
        dtype=np.float64,
    )
    residuals = np.zeros((6, 1), dtype=np.float64)
    directions = np.zeros((6, 1), dtype=np.float64)

    first = cluster_unknowns(
        state_vectors=state,
        residuals=residuals,
        motif_directions=directions,
        cluster_counts=range(2, 11),
        init="k-means++",
        n_init=20,
        max_iterations=300,
        tolerance=1.0e-4,
        algorithm="lloyd",
        random_state=42,
        mad_floor=1.0e-12,
    )
    second = cluster_unknowns(
        state_vectors=state,
        residuals=residuals,
        motif_directions=directions,
        cluster_counts=range(2, 11),
        init="k-means++",
        n_init=20,
        max_iterations=300,
        tolerance=1.0e-4,
        algorithm="lloyd",
        random_state=42,
        mad_floor=1.0e-12,
    )

    assert first.cluster_count == 2
    assert first.labels == second.labels
    assert first.silhouette == pytest.approx(second.silhouette)
    assert first.candidate_counts == (2, 3, 4, 5)


def test_updq_uses_equal_weighted_purity_stability_and_separation() -> None:
    from sciprism_x.open_set.grouping import compute_updq

    result = compute_updq(
        purity=0.9,
        stability=0.8,
        separation=0.7,
        weights=(1 / 3, 1 / 3, 1 / 3),
    )

    assert result == pytest.approx(0.8)


def test_structural_calibration_keeps_and_flags_sparse_bins() -> None:
    from sciprism_x.open_set.calibration import compute_sc

    result = compute_sc(
        confidence=np.array([0.1, 0.4, 0.6, 0.9]),
        structural_validity=np.array([0, 1, 1, 1]),
        bins=2,
        minimum_bin_samples=20,
        minimum_bin_fraction=0.01,
    )

    assert result.sc == pytest.approx(0.75)
    assert result.ece == pytest.approx(0.25)
    assert len(result.bins) == 2
    assert [bin_result.count for bin_result in result.bins] == [2, 2]
    assert all(bin_result.sparse for bin_result in result.bins)
def test_structural_calibration_reports_empty_bins_without_dropping_them() -> None:
    from sciprism_x.open_set.calibration import compute_sc

    result = compute_sc(
        confidence=np.array([0.05, 0.95]),
        structural_validity=np.array([0, 1]),
        bins=10,
        minimum_bin_samples=20,
        minimum_bin_fraction=0.01,
    )

    assert len(result.bins) == 10
    assert sum(bin_result.count == 0 for bin_result in result.bins) == 8
    assert all(bin_result.sparse for bin_result in result.bins)
