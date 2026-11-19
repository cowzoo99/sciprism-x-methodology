import math

import numpy as np
import pytest

from sciprism_x.formal.open_set import OpenSetBundle, run_open_set_analysis

from .helpers import run_config


def _bundle(points_per_cluster: int, *, offset: float) -> OpenSetBundle:
    states = []
    residuals = []
    motifs = []
    validity = []
    for cluster, center in enumerate((0.0, 5.0, 10.0)):
        for index in range(points_per_cluster):
            jitter = (index - points_per_cluster / 2) * 0.01
            states.append(np.full(15, center + offset + jitter))
            residual = np.zeros(6)
            residual[cluster] = 0.8
            residuals.append(residual)
            motif = np.zeros(6)
            motif[cluster] = 1.0
            motifs.append(motif)
            validity.append(cluster != 2)
    residual_matrix = np.asarray(residuals, dtype=np.float64)
    return OpenSetBundle(
        states=np.asarray(states, dtype=np.float64),
        residuals=residual_matrix,
        motif_mismatch=np.asarray(motifs, dtype=np.float64),
        structural_confidence=np.clip(1.0 - residual_matrix.mean(axis=1), 0.0, 1.0),
        structural_validity=np.asarray(validity, dtype=np.int64),
        reference_groups=np.asarray(
            [
                cluster
                for cluster in range(3)
                for _ in range(points_per_cluster)
            ],
            dtype=np.int64,
        ),
    )


def test_open_set_analysis_computes_all_components() -> None:
    result = run_open_set_analysis(
        _bundle(5, offset=0.0),
        _bundle(4, offset=0.25),
        run_config(),
    )

    assert len(result.seedwise_labels) == 5
    assert 2 <= result.selected_cluster_count <= 10
    assert 0.0 <= result.purity <= 1.0
    assert 0.0 <= result.stability <= 1.0
    assert result.separation >= 0.0
    assert 0.0 <= result.separation_normalized <= 1.0
    assert result.updq == pytest.approx(
        (result.purity + result.stability + result.separation_normalized) / 3.0
    )
    assert math.isfinite(result.updq)
    assert math.isfinite(result.structural_calibration)
    assert len(result.calibration_bins) == 10
    assert len(result.unknownness) == 12
    assert result.reference_groups == tuple(
        cluster for cluster in range(3) for _ in range(4)
    )


def test_open_set_analysis_records_single_cluster_degeneracy() -> None:
    rows = 12
    bundle = OpenSetBundle(
        states=np.zeros((rows, 15), dtype=np.float64),
        residuals=np.zeros((rows, 6), dtype=np.float64),
        motif_mismatch=np.zeros((rows, 6), dtype=np.float64),
        structural_confidence=np.full(rows, 0.5, dtype=np.float64),
        structural_validity=np.asarray([index % 2 for index in range(rows)]),
        reference_groups=np.asarray([index % 2 for index in range(rows)]),
    )

    result = run_open_set_analysis(bundle, bundle, run_config())

    assert result.degenerate_single_cluster is True
    assert result.selected_cluster_count == 1
    assert result.validation_candidate_counts == run_config().open_set.cluster_counts
    assert result.validation_mean_silhouettes == tuple(
        0.0 for _ in run_config().open_set.cluster_counts
    )
    assert set(result.seedwise_labels[0]) == {0}
    assert result.stability == 1.0
    assert result.separation == 0.0
    assert result.separation_normalized == 0.0
    assert math.isfinite(result.updq)
