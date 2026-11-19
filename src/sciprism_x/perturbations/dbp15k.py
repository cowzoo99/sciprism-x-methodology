from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from sciprism_x.reproducibility import derive_seed


Edge = tuple[str, str, str]
Alignment = tuple[str, str]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EdgeChange:
    graph: str
    edge: Edge
    action: str
    replacement_relation: str | None = None


@dataclass(frozen=True, slots=True)
class SeedAlignmentChange:
    alignment: Alignment
    action: str


@dataclass(frozen=True, slots=True)
class DBP15KPerturbation:
    pair_id: str
    family: str
    strength: float
    changed_edges: tuple[EdgeChange, ...]
    changed_seed_alignments: tuple[SeedAlignmentChange, ...]


def _ordered(values: Sequence[T], seed: int, namespace: str) -> tuple[T, ...]:
    unique = tuple(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"{namespace} requires non-empty values")
    rng = np.random.default_rng(derive_seed(seed, namespace))
    return tuple(unique[index] for index in rng.permutation(len(unique)))


def _removal_count(length: int, ratio: float) -> int:
    return min(length - 1, max(1, int(np.ceil(length * ratio))))


def generate_dbp15k_perturbations(
    *,
    pair_id: str,
    source_edges: Sequence[Edge],
    target_edges: Sequence[Edge],
    seed_alignments: Sequence[Alignment],
    removal_ratios: Sequence[float],
    seed_fractions: Sequence[float],
    original_seed_fraction: float,
    seed: int,
) -> tuple[DBP15KPerturbation, ...]:
    """Create the four Appendix C.2 DBP15K structural perturbation families."""
    if not pair_id:
        raise ValueError("pair_id must not be empty")
    ratios = tuple(float(value) for value in removal_ratios)
    fractions = tuple(float(value) for value in seed_fractions)
    if not ratios or any(value <= 0.0 or value >= 1.0 for value in ratios):
        raise ValueError("removal ratios must be in (0, 1)")
    if not 0.0 < original_seed_fraction <= 1.0:
        raise ValueError("original_seed_fraction must be in (0, 1]")
    if not fractions or any(value < 0.0 or value >= original_seed_fraction for value in fractions):
        raise ValueError("seed fractions must be nonnegative and below the original fraction")

    source = _ordered(source_edges, seed, f"{pair_id}:source-edges")
    target = _ordered(target_edges, seed, f"{pair_id}:target-edges")
    alignments = _ordered(seed_alignments, seed, f"{pair_id}:seed-alignments")
    records: list[DBP15KPerturbation] = []

    for ratio in ratios:
        source_count = _removal_count(len(source), ratio)
        target_count = _removal_count(len(target), ratio)
        records.append(
            DBP15KPerturbation(
                pair_id,
                "neighborhood_degradation",
                ratio,
                tuple(
                    [EdgeChange("source", edge, "remove") for edge in source[:source_count]]
                    + [EdgeChange("target", edge, "remove") for edge in target[:target_count]]
                ),
                (),
            )
        )

    for ratio in ratios:
        count = max(1, int(np.ceil(ratio * min(len(source), len(target)))))
        changed = tuple(
            EdgeChange("source", edge, "replace_relation", "ambiguous_relation")
            for edge in source[:count]
        ) + tuple(
            EdgeChange("target", edge, "replace_relation", "ambiguous_relation")
            for edge in target[:count]
        )
        records.append(
            DBP15KPerturbation(pair_id, "relation_ambiguity_injection", ratio, changed, ())
        )

    for fraction in fractions:
        retained = int(round(len(alignments) * fraction / original_seed_fraction))
        removed = alignments[max(0, min(retained, len(alignments))) :]
        records.append(
            DBP15KPerturbation(
                pair_id,
                "seed_alignment_reduction",
                fraction,
                (),
                tuple(SeedAlignmentChange(value, "remove") for value in removed),
            )
        )

    for ratio in ratios:
        source_count = _removal_count(len(source), ratio)
        target_count = _removal_count(len(target), ratio)
        changed = tuple(
            [EdgeChange("source", edge, "weaken") for edge in source[:source_count]]
            + [EdgeChange("target", edge, "weaken") for edge in target[:target_count]]
        )
        records.append(
            DBP15KPerturbation(
                pair_id,
                "cross_graph_evidence_weakening",
                ratio,
                changed,
                (),
            )
        )
    return tuple(records)
