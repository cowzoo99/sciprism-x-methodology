from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sciprism_x.config import RunConfig
from sciprism_x.open_set.calibration import CalibrationBin, compute_sc
from sciprism_x.open_set.grouping import (
    cluster_purity,
    cluster_separation,
    cluster_stability,
    cluster_unknowns,
    compute_novelty_from_validation,
    compute_unknownness,
    compute_updq,
)


@dataclass(frozen=True, slots=True)
class OpenSetBundle:
    states: np.ndarray
    residuals: np.ndarray
    motif_mismatch: np.ndarray
    structural_confidence: np.ndarray
    structural_validity: np.ndarray
    reference_groups: np.ndarray


@dataclass(frozen=True, slots=True)
class OpenSetMetricRow:
    selected_cluster_count: int
    validation_candidate_counts: tuple[int, ...]
    validation_mean_silhouettes: tuple[float, ...]
    seedwise_labels: tuple[tuple[int, ...], ...]
    seedwise_purity: tuple[float, ...]
    seedwise_separation_normalized: tuple[float, ...]
    seedwise_updq: tuple[float, ...]
    seedwise_structural_calibration: tuple[float, ...]
    reference_groups: tuple[int, ...]
    unknownness: tuple[float, ...]
    purity: float
    stability: float
    separation: float
    separation_normalized: float
    updq: float
    structural_calibration: float
    calibration_ece: float
    calibration_bins: tuple[CalibrationBin, ...]
    degenerate_single_cluster: bool


def _validate_bundle(bundle: OpenSetBundle, name: str) -> None:
    blocks = (bundle.states, bundle.residuals, bundle.motif_mismatch)
    if any(block.ndim != 2 or block.shape[0] == 0 for block in blocks):
        raise ValueError(f"{name} feature blocks must be non-empty matrices")
    if len({block.shape[0] for block in blocks}) != 1:
        raise ValueError(f"{name} feature blocks must have equal row counts")
    row_count = blocks[0].shape[0]
    if bundle.structural_confidence.shape != (row_count,):
        raise ValueError(f"{name} structural confidence has the wrong shape")
    if bundle.structural_validity.shape != (row_count,):
        raise ValueError(f"{name} structural validity has the wrong shape")
    if bundle.reference_groups.shape != (row_count,):
        raise ValueError(f"{name} reference groups have the wrong shape")
    if any(not np.isfinite(value).all() for value in (*blocks, bundle.structural_confidence)):
        raise ValueError(f"{name} open-set inputs must be finite")


def _cluster(
    bundle: OpenSetBundle,
    config: RunConfig,
    *,
    seed: int,
    cluster_counts: tuple[int, ...],
):
    return cluster_unknowns(
        state_vectors=bundle.states,
        residuals=bundle.residuals,
        motif_directions=bundle.motif_mismatch,
        cluster_counts=cluster_counts,
        init=config.open_set.kmeans_init,
        n_init=config.open_set.kmeans_n_init,
        max_iterations=config.open_set.kmeans_max_iterations,
        tolerance=config.open_set.kmeans_tolerance,
        algorithm=config.open_set.kmeans_algorithm,
        random_state=seed,
        mad_floor=config.open_set.separation_floor,
    )


def _features(bundle: OpenSetBundle) -> np.ndarray:
    return np.concatenate(
        (bundle.states, bundle.residuals, bundle.motif_mismatch),
        axis=1,
    )


def run_open_set_analysis(
    validation: OpenSetBundle,
    unknown: OpenSetBundle,
    config: RunConfig,
) -> OpenSetMetricRow:
    _validate_bundle(validation, "validation")
    _validate_bundle(unknown, "unknown")
    validation_features = _features(validation)
    unknown_features = _features(unknown)
    degenerate_single_cluster = np.unique(validation_features, axis=0).shape[0] == 1
    if degenerate_single_cluster:
        counts = config.open_set.cluster_counts
        mean_scores = tuple(0.0 for _ in counts)
        selected_count = 1
        validation_fixed = ()
        seedwise_labels = tuple(
            tuple(0 for _ in range(unknown_features.shape[0]))
            for _seed in config.paper.seeds
        )
    else:
        validation_searches = tuple(
            _cluster(
                validation,
                config,
                seed=seed,
                cluster_counts=config.open_set.cluster_counts,
            )
            for seed in config.paper.seeds
        )
        counts = validation_searches[0].candidate_counts
        if any(result.candidate_counts != counts for result in validation_searches):
            raise ValueError("validation cluster searches used inconsistent candidate counts")
        mean_scores = tuple(
            float(np.mean([result.candidate_scores[index] for result in validation_searches]))
            for index in range(len(counts))
        )
        selected_count = max(
            zip(mean_scores, counts, strict=True),
            key=lambda item: (item[0], -item[1]),
        )[1]
        validation_fixed = tuple(
            _cluster(validation, config, seed=seed, cluster_counts=(selected_count,))
            for seed in config.paper.seeds
        )
        unknown_fixed = tuple(
            _cluster(unknown, config, seed=seed, cluster_counts=(selected_count,))
            for seed in config.paper.seeds
        )
        seedwise_labels = tuple(result.labels for result in unknown_fixed)

    reference_groups = tuple(int(value) for value in unknown.reference_groups)
    seedwise_purity = tuple(
        float(cluster_purity(labels, reference_groups)) for labels in seedwise_labels
    )
    purity = float(np.mean(seedwise_purity, dtype=np.float64))
    stability = float(np.clip(cluster_stability(seedwise_labels), 0.0, 1.0))
    if degenerate_single_cluster:
        separation = 0.0
        separation_normalized = 0.0
        seedwise_separation_normalized = tuple(0.0 for _ in config.paper.seeds)
    else:
        validation_separations = np.asarray(
            [
                cluster_separation(
                    validation_features,
                    result.labels,
                    config.open_set.separation_floor,
                )
                for result in validation_fixed
            ],
            dtype=np.float64,
        )
        unknown_separations = np.asarray(
            [
                cluster_separation(
                    unknown_features,
                    labels,
                    config.open_set.separation_floor,
                )
                if len(set(labels)) > 1
                else 0.0
                for labels in seedwise_labels
            ],
            dtype=np.float64,
        )
        separation = float(unknown_separations.mean())
        separation_scale = max(
            float(
                np.quantile(
                    validation_separations,
                    config.open_set.novelty_normalization_quantile,
                )
            ),
            config.open_set.separation_floor,
        )
        seedwise_separation_normalized = tuple(
            float(np.clip(value / separation_scale, 0.0, 1.0))
            for value in unknown_separations
        )
        separation_normalized = float(
            np.mean(seedwise_separation_normalized, dtype=np.float64)
        )
    novelty = compute_novelty_from_validation(
        candidate_states=unknown.states,
        candidate_residuals=unknown.residuals,
        candidate_motif_directions=unknown.motif_mismatch,
        validation_states=validation.states,
        validation_residuals=validation.residuals,
        validation_motif_directions=validation.motif_mismatch,
        mad_floor=config.open_set.separation_floor,
        normalization_quantile=config.open_set.novelty_normalization_quantile,
    )
    unknownness = compute_unknownness(
        support_distance=np.asarray(novelty.support_distance),
        residual_novelty=np.asarray(novelty.residual_novelty),
        motif_direction_novelty=np.asarray(novelty.motif_direction_novelty),
        weights=config.open_set.unknownness_weights,
    )
    seedwise_updq = tuple(
        compute_updq(
            purity=seed_purity,
            stability=stability,
            separation=seed_separation,
            weights=config.open_set.updq_weights,
        )
        for seed_purity, seed_separation in zip(
            seedwise_purity,
            seedwise_separation_normalized,
            strict=True,
        )
    )
    updq = float(np.mean(seedwise_updq, dtype=np.float64))
    calibration = compute_sc(
        confidence=unknown.structural_confidence,
        structural_validity=unknown.structural_validity,
        bins=config.open_set.calibration_bins,
        minimum_bin_samples=config.open_set.calibration_minimum_bin_samples,
        minimum_bin_fraction=config.open_set.calibration_minimum_bin_fraction,
    )
    return OpenSetMetricRow(
        selected_cluster_count=selected_count,
        validation_candidate_counts=counts,
        validation_mean_silhouettes=mean_scores,
        seedwise_labels=seedwise_labels,
        seedwise_purity=seedwise_purity,
        seedwise_separation_normalized=seedwise_separation_normalized,
        seedwise_updq=seedwise_updq,
        seedwise_structural_calibration=tuple(
            calibration.sc for _seed in config.paper.seeds
        ),
        reference_groups=reference_groups,
        unknownness=tuple(float(value) for value in unknownness),
        purity=purity,
        stability=stability,
        separation=separation,
        separation_normalized=separation_normalized,
        updq=updq,
        structural_calibration=calibration.sc,
        calibration_ece=calibration.ece,
        calibration_bins=calibration.bins,
        degenerate_single_cluster=degenerate_single_cluster,
    )
