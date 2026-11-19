from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sciprism_x.reproducibility import derive_seed


@dataclass(frozen=True, slots=True)
class UnitChange:
    unit_id: str
    action: str
    level: float


@dataclass(frozen=True, slots=True)
class SciCUPerturbation:
    task_id: str
    family: str
    strength: float
    changed_units: tuple[UnitChange, ...]
    changed_edges: tuple[tuple[str, str, str], ...]


def _ordered_units(unit_ids: Sequence[str], seed: int, namespace: str) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(unit_ids))
    if not unique or any(not unit_id for unit_id in unique):
        raise ValueError(f"{namespace} requires at least one non-empty unit ID")
    rng = np.random.default_rng(derive_seed(seed, namespace))
    return tuple(unique[index] for index in rng.permutation(len(unique)))


def _changes(
    ordered_ids: tuple[str, ...],
    count: int,
    action: str,
    level: float,
) -> tuple[UnitChange, ...]:
    return tuple(UnitChange(unit_id, action, level) for unit_id in ordered_ids[:count])


def generate_scicu_perturbations(
    *,
    task_id: str,
    evidence_unit_ids: Sequence[str],
    context_unit_ids: Sequence[str],
    modality_unit_ids: Sequence[str],
    symbolic_unit_ids: Sequence[str],
    boundary_unit_ids: Sequence[str],
    radii: Sequence[float],
    depth_steps: Sequence[int],
    context_steps: Sequence[int],
    seed: int,
) -> tuple[SciCUPerturbation, ...]:
    """Create the five Appendix C.2 SciCUEval perturbation families.

    Every record carries the exact affected scientific units. Evidence removal
    retains at least one evidence unit so the generated task remains answerable.
    Materialization of changed scientific content is deliberately left to the
    domain-specific task adapter, which must validate admissibility.
    """
    if not task_id:
        raise ValueError("task_id must not be empty")
    radius_values = tuple(float(value) for value in radii)
    depth_values = tuple(int(value) for value in depth_steps)
    context_values = tuple(int(value) for value in context_steps)
    if not radius_values or any(value <= 0.0 or value > 1.0 for value in radius_values):
        raise ValueError("radii must be in (0, 1]")
    if len(depth_values) != len(radius_values) or len(context_values) != len(radius_values):
        raise ValueError("radii, depth_steps, and context_steps must have equal lengths")
    if any(value <= 0 for value in depth_values + context_values):
        raise ValueError("step counts must be positive")

    evidence = _ordered_units(evidence_unit_ids, seed, f"{task_id}:evidence")
    context = _ordered_units(context_unit_ids, seed, f"{task_id}:context")
    modality = _ordered_units(modality_unit_ids, seed, f"{task_id}:modality")
    symbolic = _ordered_units(symbolic_unit_ids, seed, f"{task_id}:symbolic")
    boundary = _ordered_units(boundary_unit_ids, seed, f"{task_id}:boundary")
    records: list[SciCUPerturbation] = []

    for radius in radius_values:
        removal_count = min(
            len(evidence) - 1,
            max(1, int(np.ceil(radius * len(evidence)))),
        )
        if removal_count <= 0:
            raise ValueError("evidence removal requires at least two evidence units")
        records.append(
            SciCUPerturbation(
                task_id,
                "evidence_removal",
                radius,
                _changes(evidence, removal_count, "remove", radius),
                (),
            )
        )

    for radius, count in zip(radius_values, context_values, strict=True):
        records.append(
            SciCUPerturbation(
                task_id,
                "context_composition_shift",
                float(count),
                _changes(context, min(count, len(context)), "recompose", radius),
                (),
            )
        )

    for radius in radius_values:
        records.append(
            SciCUPerturbation(
                task_id,
                "modality_conflict",
                radius,
                _changes(modality, len(modality), "inject_conflict", radius),
                (),
            )
        )

    for radius, count in zip(radius_values, depth_values, strict=True):
        records.append(
            SciCUPerturbation(
                task_id,
                "symbolic_depth_increase",
                float(count),
                _changes(symbolic, len(symbolic), "add_transformations", float(count)),
                (),
            )
        )

    for radius in radius_values:
        records.append(
            SciCUPerturbation(
                task_id,
                "boundary_condition_change",
                radius,
                _changes(boundary, len(boundary), "shift_condition", radius),
                (),
            )
        )
    return tuple(records)
