from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass(frozen=True, slots=True)
class NoveltyComponents:
    support_distance: tuple[float, ...]
    residual_novelty: tuple[float, ...]
    motif_direction_novelty: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OpenSetResult:
    labels: tuple[int, ...]
    cluster_count: int
    silhouette: float
    candidate_counts: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    feature_median: tuple[float, ...]
    feature_mad: tuple[float, ...]


def _as_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _validate_weights(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(tuple(weights), dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("weights must contain three finite nonnegative values")
    if not np.isclose(values.sum(), 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("weights must sum to one")
    return values


def compute_unknownness(
    *,
    support_distance: np.ndarray,
    residual_novelty: np.ndarray,
    motif_direction_novelty: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    components = np.column_stack(
        [support_distance, residual_novelty, motif_direction_novelty]
    ).astype(np.float64, copy=False)
    if components.ndim != 2 or components.shape[1] != 3:
        raise ValueError("unknownness components must have matching one-dimensional shapes")
    if not np.isfinite(components).all() or np.any((components < 0.0) | (components > 1.0)):
        raise ValueError("unknownness components must be finite and normalized to [0, 1]")
    return components @ _validate_weights(weights)


def _normalized_nearest_distance(
    candidates: np.ndarray,
    reference: np.ndarray,
    mad_floor: float,
    normalization_quantile: float,
) -> np.ndarray:
    if candidates.shape[1] != reference.shape[1]:
        raise ValueError("candidate and validation feature dimensions must match")
    if reference.shape[0] < 2:
        raise ValueError("validation support requires at least two rows")
    median = np.median(reference, axis=0)
    mad = np.maximum(np.median(np.abs(reference - median), axis=0), mad_floor)
    scaled_reference = (reference - median) / mad
    scaled_candidates = (candidates - median) / mad

    reference_distances = np.linalg.norm(
        scaled_reference[:, None, :] - scaled_reference[None, :, :], axis=2
    )
    np.fill_diagonal(reference_distances, np.inf)
    leave_one_out = np.min(reference_distances, axis=1)
    distance_scale = max(
        float(np.quantile(leave_one_out, normalization_quantile)), mad_floor
    )
    candidate_distances = np.min(
        np.linalg.norm(
            scaled_candidates[:, None, :] - scaled_reference[None, :, :], axis=2
        ),
        axis=1,
    )
    return np.clip(candidate_distances / distance_scale, 0.0, 1.0)


def compute_novelty_from_validation(
    *,
    candidate_states: np.ndarray,
    candidate_residuals: np.ndarray,
    candidate_motif_directions: np.ndarray,
    validation_states: np.ndarray,
    validation_residuals: np.ndarray,
    validation_motif_directions: np.ndarray,
    mad_floor: float,
    normalization_quantile: float,
) -> NoveltyComponents:
    if not 0.0 < normalization_quantile <= 1.0:
        raise ValueError("normalization_quantile must be in (0, 1]")
    if not np.isfinite(mad_floor) or mad_floor <= 0.0:
        raise ValueError("mad_floor must be finite and positive")
    candidate_blocks = tuple(
        _as_matrix(value, name)
        for value, name in (
            (candidate_states, "candidate_states"),
            (candidate_residuals, "candidate_residuals"),
            (candidate_motif_directions, "candidate_motif_directions"),
        )
    )
    validation_blocks = tuple(
        _as_matrix(value, name)
        for value, name in (
            (validation_states, "validation_states"),
            (validation_residuals, "validation_residuals"),
            (validation_motif_directions, "validation_motif_directions"),
        )
    )
    if len({block.shape[0] for block in candidate_blocks}) != 1:
        raise ValueError("candidate blocks must have equal row counts")
    if len({block.shape[0] for block in validation_blocks}) != 1:
        raise ValueError("validation blocks must have equal row counts")
    values = tuple(
        _normalized_nearest_distance(
            candidate,
            reference,
            mad_floor,
            normalization_quantile,
        )
        for candidate, reference in zip(candidate_blocks, validation_blocks, strict=True)
    )
    return NoveltyComponents(*(tuple(float(item) for item in value) for value in values))


def cluster_unknowns(
    *,
    state_vectors: np.ndarray,
    residuals: np.ndarray,
    motif_directions: np.ndarray,
    cluster_counts: Sequence[int],
    init: str,
    n_init: int,
    max_iterations: int,
    tolerance: float,
    algorithm: str,
    random_state: int,
    mad_floor: float,
) -> OpenSetResult:
    blocks = tuple(
        _as_matrix(value, name)
        for value, name in (
            (state_vectors, "state_vectors"),
            (residuals, "residuals"),
            (motif_directions, "motif_directions"),
        )
    )
    if len({block.shape[0] for block in blocks}) != 1:
        raise ValueError("joint open-set feature blocks must have equal row counts")
    features = np.concatenate(blocks, axis=1)
    if not np.isfinite(mad_floor) or mad_floor <= 0.0:
        raise ValueError("mad_floor must be finite and positive")
    median = np.median(features, axis=0)
    mad = np.maximum(np.median(np.abs(features - median), axis=0), mad_floor)
    standardized = (features - median) / mad
    counts = tuple(
        count
        for count in dict.fromkeys(int(value) for value in cluster_counts)
        if 2 <= count < features.shape[0]
    )
    if not counts:
        raise ValueError("cluster search requires at least one count in [2, n_samples)")
    if n_init <= 0 or max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("K-means controls must be positive")

    candidates: list[tuple[float, int, np.ndarray]] = []
    for count in counts:
        estimator = KMeans(
            n_clusters=count,
            init=init,
            n_init=n_init,
            max_iter=max_iterations,
            tol=tolerance,
            verbose=0,
            random_state=random_state,
            copy_x=True,
            algorithm=algorithm,
        )
        labels = estimator.fit_predict(standardized)
        distinct_labels = np.unique(labels).size
        score = (
            float(silhouette_score(standardized, labels, metric="euclidean"))
            if 1 < distinct_labels < standardized.shape[0]
            else -1.0
        )
        candidates.append((score, count, labels))
    best_score, best_count, best_labels = max(candidates, key=lambda item: (item[0], -item[1]))
    return OpenSetResult(
        labels=tuple(int(label) for label in best_labels),
        cluster_count=best_count,
        silhouette=best_score,
        candidate_counts=counts,
        candidate_scores=tuple(score for score, _count, _labels in candidates),
        feature_median=tuple(float(value) for value in median),
        feature_mad=tuple(float(value) for value in mad),
    )


def cluster_purity(labels: Sequence[int], reference_groups: Sequence[int]) -> float:
    clusters = np.asarray(tuple(labels))
    groups = np.asarray(tuple(reference_groups))
    if clusters.shape != groups.shape or clusters.ndim != 1 or clusters.size == 0:
        raise ValueError("labels and reference_groups must be non-empty equal 1-D shapes")
    correct = 0
    for cluster in np.unique(clusters):
        _values, counts = np.unique(groups[clusters == cluster], return_counts=True)
        correct += int(np.max(counts))
    return correct / clusters.size


def cluster_stability(seedwise_labels: Sequence[Sequence[int]]) -> float:
    labelings = tuple(np.asarray(tuple(labels)) for labels in seedwise_labels)
    if len(labelings) < 2 or any(labels.ndim != 1 for labels in labelings):
        raise ValueError("stability requires at least two one-dimensional labelings")
    if len({labels.shape for labels in labelings}) != 1:
        raise ValueError("seed-wise labelings must have equal shapes")
    scores = [
        adjusted_rand_score(labelings[left], labelings[right])
        for left in range(len(labelings))
        for right in range(left + 1, len(labelings))
    ]
    return float(np.mean(scores, dtype=np.float64))


def cluster_separation(features: np.ndarray, labels: Sequence[int], floor: float) -> float:
    matrix = _as_matrix(features, "features")
    assignments = np.asarray(tuple(labels))
    if assignments.shape != (matrix.shape[0],):
        raise ValueError("labels must match the number of feature rows")
    clusters = np.unique(assignments)
    if clusters.size < 2:
        raise ValueError("separation requires at least two clusters")
    centroids = np.vstack([matrix[assignments == cluster].mean(axis=0) for cluster in clusters])
    inter = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    upper = inter[np.triu_indices(clusters.size, k=1)]
    within = np.mean(
        [
            np.linalg.norm(matrix[index] - centroids[np.where(clusters == label)[0][0]])
            for index, label in enumerate(assignments)
        ],
        dtype=np.float64,
    )
    return float(np.mean(upper, dtype=np.float64) / max(float(within), floor))


def compute_updq(
    *, purity: float, stability: float, separation: float, weights: Sequence[float]
) -> float:
    values = np.asarray([purity, stability, separation], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("UPDQ components must be finite")
    return float(values @ _validate_weights(weights))
