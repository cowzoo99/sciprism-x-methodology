import math
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_paper_contract_is_explicit() -> None:
    from sciprism_x.config import load_run_config

    config = load_run_config(PROJECT_ROOT / "configs" / "track_b.yaml")

    assert config.paper.seeds == (42, 0, 1, 123, 1234)
    assert config.paper.hyperparameter_trials == 50
    assert config.paper.bootstrap_resamples == 1000
    assert config.paper.max_epochs == 100
    assert config.transport.dtype == "float64"
    assert math.isclose(
        sum(config.transport.node_cost_weights.model_dump().values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert config.transport.node_cost_weights.scibert_cosine == 1.0 / 6.0
    assert config.residuals.minimum_support_threshold == 0.5
    assert config.residuals.residual_consistency_beta == (-0.5,) * 6
    assert config.state_field.adamw_betas == (0.9, 0.999)
    assert config.state_field.adamw_eps == 1.0e-8
    assert config.state_field.class_weighting == "inverse_frequency_mean_one"
    assert config.boundary.mad_consistency_factor == 1.0
    assert config.open_set.cluster_selection == "silhouette"
    assert config.open_set.kmeans_init == "k-means++"
    assert config.open_set.kmeans_n_init == 20
    assert config.open_set.kmeans_max_iterations == 300
    assert config.open_set.kmeans_tolerance == 1.0e-4
    assert config.open_set.kmeans_algorithm == "lloyd"
    assert config.open_set.unknownness_weights == (1 / 3, 1 / 3, 1 / 3)
    assert config.open_set.novelty_normalization_quantile == 0.95
    assert config.open_set.calibration_minimum_bin_samples == 20
    assert config.open_set.calibration_minimum_bin_fraction == 0.01
    assert config.graph.short_fragment_token_threshold == 4
    assert config.graph.salience_weights.model_dump() == {
        "inverse_position": 0.25,
        "motif_confidence": 0.25,
        "variable_reuse": 0.25,
        "graph_centrality": 0.25,
    }
    assert config.validators.scibert.revision == (
        "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
    )
    assert config.validators.bart_mnli.revision == (
        "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
    )
    assert all(record.provenance is not None for record in config.parameter_records())


def test_track_a_rejects_published_targets() -> None:
    from sciprism_x.config import TrackIsolationError, load_run_config

    with pytest.raises(TrackIsolationError, match="published_targets"):
        load_run_config(PROJECT_ROOT / "tests" / "fixtures" / "invalid_track_a.yaml")
