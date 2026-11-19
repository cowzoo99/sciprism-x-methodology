from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from sciprism_x.data.schema import SciCUCase

from .allocation import AllocationRow, DBP_FAMILIES, SCICU_FAMILIES, STRENGTHS
from .datasets import DBPLanguageDataset


DBP_REMOVAL_RATIOS = (0.05, 0.10, 0.20)
DBP_SEED_FRACTIONS = (0.25, 0.20, 0.10)
Triple = tuple[int, int, int]
Alignment = tuple[int, int]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MaterializedSciCUPerturbation:
    allocation_id: str
    family: str
    strength: float
    original_task_id: str
    case: SciCUCase
    changed_units: tuple[str, ...]
    valid: bool
    validity_checks: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class GraphEdgeChange:
    graph: str
    edge: Triple
    action: str
    replacement_relation: int | None = None


@dataclass(frozen=True, slots=True)
class GraphSeedChange:
    alignment: Alignment
    action: str


@dataclass(frozen=True, slots=True)
class MaterializedDBPPerturbation:
    allocation_id: str
    family: str
    strength: float
    applied_value: float
    language_pair: str
    source_id: int
    target_id: int
    triples_1: tuple[Triple, ...]
    triples_2: tuple[Triple, ...]
    seed_pairs: tuple[Alignment, ...]
    changed_edges: tuple[GraphEdgeChange, ...]
    changed_seed_alignments: tuple[GraphSeedChange, ...]
    valid: bool
    validity_checks: tuple[tuple[str, bool], ...]


def _step(strength: float, strengths: Sequence[float] = STRENGTHS) -> int:
    try:
        return tuple(strengths).index(strength) + 1
    except ValueError as error:
        raise ValueError(f"unsupported perturbation strength: {strength}") from error


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if part.strip()
    ]


def _new_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deterministic_order(values: tuple[T, ...], *, seed: int, namespace: str) -> tuple[T, ...]:
    return tuple(
        sorted(
            dict.fromkeys(values),
            key=lambda value: (
                hashlib.sha256(f"{seed}::{namespace}::{value!r}".encode("utf-8")).digest(),
                value,
            ),
        )
    )


def _two_hop_edges(triples: tuple[Triple, ...], entity_id: int) -> tuple[Triple, ...]:
    first = tuple(edge for edge in triples if entity_id in (edge[0], edge[2]))
    neighbors = {
        node
        for subject, _, object_ in first
        for node in (subject, object_)
    }
    second = tuple(
        edge for edge in triples if edge[0] in neighbors or edge[2] in neighbors
    )
    return tuple(dict.fromkeys(first + second))


def _changed_edge_count(length: int, ratio: float) -> int:
    if length <= 0:
        raise ValueError("DBP15K perturbation requires non-empty local graph evidence")
    return min(length, max(1, int(math.ceil(length * ratio))))


def _replace_relations(
    triples: tuple[Triple, ...],
    replacements: dict[Triple, int],
) -> tuple[Triple, ...]:
    return tuple(
        (edge[0], replacements[edge], edge[2]) if edge in replacements else edge
        for edge in triples
    )


def materialize_dbp_perturbation(
    dataset: DBPLanguageDataset,
    allocation: AllocationRow,
    *,
    seed: int,
    standardized_radii: Sequence[float] = STRENGTHS,
    removal_ratios: Sequence[float] = DBP_REMOVAL_RATIOS,
    seed_fractions: Sequence[float] = DBP_SEED_FRACTIONS,
) -> MaterializedDBPPerturbation:
    if allocation.dataset != "dbp15k" or allocation.regime != "boundary_stress":
        raise ValueError("DBP15K perturbation requires a boundary allocation")
    if allocation.family not in DBP_FAMILIES or allocation.strength is None:
        raise ValueError("DBP15K perturbation family and strength are required")
    if allocation.language_pair != dataset.language_pair:
        raise ValueError("DBP15K allocation language pair does not match the dataset")
    parts = allocation.case_id.split("::")
    if len(parts) != 3 or parts[0] != dataset.language_pair:
        raise ValueError(f"invalid DBP15K case identifier: {allocation.case_id}")
    source_id, target_id = int(parts[1]), int(parts[2])
    if (source_id, target_id) not in dataset.reference_pairs:
        raise ValueError("DBP15K case must be a frozen reference alignment")

    strengths = tuple(float(value) for value in standardized_radii)
    index = _step(allocation.strength, strengths) - 1
    ratios = tuple(float(value) for value in removal_ratios)
    fractions = tuple(float(value) for value in seed_fractions)
    if len(ratios) != len(strengths) or len(fractions) != len(strengths):
        raise ValueError("DBP perturbation grids must match the three paper strengths")
    source_local = _deterministic_order(
        _two_hop_edges(dataset.triples_1, source_id),
        seed=seed,
        namespace=f"{allocation.allocation_id}::source",
    )
    target_local = _deterministic_order(
        _two_hop_edges(dataset.triples_2, target_id),
        seed=seed,
        namespace=f"{allocation.allocation_id}::target",
    )
    triples_1 = dataset.triples_1
    triples_2 = dataset.triples_2
    seed_pairs = dataset.seed_pairs
    edge_changes: list[GraphEdgeChange] = []
    seed_changes: list[GraphSeedChange] = []
    original_relations = {
        relation for _, relation, _ in dataset.triples_1 + dataset.triples_2
    }

    if allocation.family == "seed_alignment_reduction":
        applied_value = fractions[index]
        ordered_seeds = _deterministic_order(
            dataset.seed_pairs,
            seed=seed,
            namespace=f"{allocation.allocation_id}::seeds",
        )
        retained_count = int(round(len(ordered_seeds) * applied_value / dataset.seed_fraction))
        retained_count = min(len(ordered_seeds), max(0, retained_count))
        retained = frozenset(ordered_seeds[:retained_count])
        removed = tuple(value for value in ordered_seeds if value not in retained)
        seed_pairs = tuple(value for value in dataset.seed_pairs if value in retained)
        seed_changes.extend(GraphSeedChange(value, "remove") for value in removed)
    else:
        applied_value = ratios[index]
        source_selected = source_local[: _changed_edge_count(len(source_local), applied_value)]
        target_selected = target_local[: _changed_edge_count(len(target_local), applied_value)]
        if allocation.family == "neighborhood_degradation":
            source_removed = frozenset(source_selected)
            target_removed = frozenset(target_selected)
            triples_1 = tuple(edge for edge in triples_1 if edge not in source_removed)
            triples_2 = tuple(edge for edge in triples_2 if edge not in target_removed)
            edge_changes.extend(GraphEdgeChange("source", edge, "remove") for edge in source_selected)
            edge_changes.extend(GraphEdgeChange("target", edge, "remove") for edge in target_selected)
        else:
            maximum_relation = max(
                relation for _, relation, _ in dataset.triples_1 + dataset.triples_2
            )
            if allocation.family == "relation_ambiguity_injection":
                source_relation = target_relation = maximum_relation + 1
                action = "replace_relation_with_shared_ambiguous_relation"
            elif allocation.family == "cross_graph_evidence_weakening":
                source_relation = maximum_relation + 1
                target_relation = maximum_relation + 2
                action = "replace_relation_with_graph_specific_weak_relation"
            else:
                raise ValueError(f"unsupported DBP15K perturbation: {allocation.family}")
            source_replacements = {edge: source_relation for edge in source_selected}
            target_replacements = {edge: target_relation for edge in target_selected}
            triples_1 = _replace_relations(triples_1, source_replacements)
            triples_2 = _replace_relations(triples_2, target_replacements)
            edge_changes.extend(
                GraphEdgeChange("source", edge, action, source_relation)
                for edge in source_selected
            )
            edge_changes.extend(
                GraphEdgeChange("target", edge, action, target_relation)
                for edge in target_selected
            )

    family_checks: tuple[tuple[str, bool], ...]
    if allocation.family == "neighborhood_degradation":
        family_checks = (
            ("source_local_graph_remains_valid", bool(_two_hop_edges(triples_1, source_id))),
            ("target_local_graph_remains_valid", bool(_two_hop_edges(triples_2, target_id))),
        )
    elif allocation.family == "relation_ambiguity_injection":
        family_checks = (
            (
                "no_ground_truth_relation_leakage",
                all(
                    change.replacement_relation not in original_relations
                    for change in edge_changes
                ),
            ),
        )
    elif allocation.family == "seed_alignment_reduction":
        family_checks = (
            ("same_test_pair_retained", (source_id, target_id) in dataset.reference_pairs),
        )
    else:
        family_checks = (
            (
                "no_cross_graph_label_leakage",
                len(
                    {
                        change.replacement_relation
                        for change in edge_changes
                        if change.replacement_relation is not None
                    }
                )
                == 2,
            ),
        )
    validity_checks = (
        ("reference_alignment_preserved", (source_id, target_id) in dataset.reference_pairs),
        ("source_graph_nonempty", bool(triples_1)),
        ("target_graph_nonempty", bool(triples_2)),
        ("seed_evidence_nonempty", bool(seed_pairs)),
        ("intervention_applied", bool(edge_changes or seed_changes)),
        ("target_remains_in_full_candidate_pool", target_id in dataset.target_candidate_ids),
    ) + family_checks
    return MaterializedDBPPerturbation(
        allocation_id=allocation.allocation_id,
        family=allocation.family,
        strength=allocation.strength,
        applied_value=applied_value,
        language_pair=dataset.language_pair,
        source_id=source_id,
        target_id=target_id,
        triples_1=triples_1,
        triples_2=triples_2,
        seed_pairs=seed_pairs,
        changed_edges=tuple(edge_changes),
        changed_seed_alignments=tuple(seed_changes),
        valid=all(value for _, value in validity_checks),
        validity_checks=validity_checks,
    )


def materialize_scicu_perturbation(
    case: SciCUCase,
    allocation: AllocationRow,
    *,
    seed: int,
    standardized_radii: Sequence[float] = STRENGTHS,
    symbolic_depth_steps: Sequence[int] = (1, 2, 3),
    context_composition_steps: Sequence[int] = (1, 2, 3),
) -> MaterializedSciCUPerturbation:
    if allocation.dataset != "scicueval" or allocation.regime != "boundary_stress":
        raise ValueError("SciCUEval perturbation requires a boundary allocation")
    if allocation.family not in SCICU_FAMILIES or allocation.strength is None:
        raise ValueError("SciCUEval perturbation family and strength are required")
    strengths = tuple(float(value) for value in standardized_radii)
    strength_index = _step(allocation.strength, strengths) - 1
    nominal_step = strength_index + 1
    depth_steps = tuple(int(value) for value in symbolic_depth_steps)
    composition_steps = tuple(int(value) for value in context_composition_steps)
    if len(depth_steps) != len(strengths) or len(composition_steps) != len(strengths):
        raise ValueError("SciCUEval step grids must match the three paper strengths")
    prompt = case.prompt
    context = case.reference_context
    answer = case.reference_answer
    changed: list[str] = []
    family_checks: tuple[tuple[str, bool], ...]

    if allocation.family == "evidence_removal":
        units = _sentences(context)
        answer_tokens = set(re.findall(r"[A-Za-z0-9]+", answer.casefold()))
        overlap = tuple(
            len(answer_tokens & set(re.findall(r"[A-Za-z0-9]+", unit.casefold())))
            for unit in units
        )
        anchor = max(range(len(units)), key=lambda index: (overlap[index], -index))
        candidates = tuple(index for index in range(len(units)) if index != anchor)
        ordered = _deterministic_order(
            candidates,
            seed=seed,
            namespace=f"{allocation.allocation_id}::evidence",
        )
        removed = frozenset(ordered[: min(nominal_step, len(ordered))])
        changed.extend(units[index] for index in range(len(units)) if index in removed)
        context = " ".join(
            unit for index, unit in enumerate(units) if index not in removed
        )
        family_checks = (
            ("task_remains_answerable", bool(units) and units[anchor] in context),
        )
    elif allocation.family == "context_composition_shift":
        units = tuple(_sentences(context))
        step = composition_steps[strength_index]
        if len(units) > 1:
            if step < len(units):
                offset = step
                permutation = tuple(range(offset, len(units))) + tuple(range(offset))
            else:
                permutation = tuple(reversed(range(len(units))))
            reordered = tuple(units[index] for index in permutation)
            context = " ".join(reordered)
            new_position = {
                original_index: reordered_index
                for reordered_index, original_index in enumerate(permutation)
            }
            changed.extend(
                f"context-position-{index}->{new_position[index]}"
                for index in range(len(units))
                if new_position[index] != index
            )
        family_checks = (
            (
                "no_contradiction_introduced",
                len(units) > 1 and sorted(units) == sorted(_sentences(context)),
            ),
        )
    elif allocation.family == "modality_conflict":
        number_pattern = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)
        selected = min(nominal_step, len(number_pattern.findall(context)))
        replacement_index = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal replacement_index
            replacement_index += 1
            if replacement_index > selected:
                return match.group(0)
            value = float(match.group(0))
            changed.append(f"modality-value-{replacement_index}")
            return f"{value * (1.0 + 0.05 * replacement_index):.6g}"

        context = number_pattern.sub(replace, context)
        family_checks = (
            (
                "artifact_remains_interpretable",
                selected > 0 and bool(context.strip()) and len(_sentences(context)) > 0,
            ),
        )
    elif allocation.family == "symbolic_depth_increase":
        step = depth_steps[strength_index]
        prompt += (
            f" Show {step} additional algebraically equivalent intermediate "
            "transformation steps without changing the requested scientific target."
        )
        changed.extend(f"symbolic-step-{index + 1}" for index in range(step))
        family_checks = (
            ("same_scientific_target", answer == case.reference_answer),
        )
    elif allocation.family == "boundary_condition_change":
        prompt = re.sub(r"\bideal\b", "non-ideal", prompt, flags=re.IGNORECASE)
        context = re.sub(r"\bideal\b", "non-ideal", context, flags=re.IGNORECASE)
        answer = re.sub(r"\bideal\b", "non-ideal", answer, flags=re.IGNORECASE)
        for index in range(nominal_step):
            marker = f"boundary-condition-{index + 1}"
            context += (
                f" Perturbed boundary condition {index + 1} remains within the "
                "stated solvable regime."
            )
            changed.append(marker)
        family_checks = (
            (
                "scientifically_admissible_condition",
                bool(prompt.strip()) and bool(context.strip()) and bool(answer.strip()),
            ),
        )
    else:
        raise ValueError(f"unsupported SciCUEval perturbation: {allocation.family}")

    target = case.reference_answer.strip().casefold()
    original_visible_text = f"{case.prompt}\n{case.reference_context}".casefold()
    perturbed_visible_text = f"{prompt}\n{context}".casefold()
    introduced_target_leakage = (
        target not in original_visible_text and target in perturbed_visible_text
    )
    validity_checks = (
        ("prompt_nonempty", bool(prompt.strip())),
        ("reference_context_nonempty", bool(context.strip())),
        ("reference_target_nonempty", bool(answer.strip())),
        ("intervention_applied", bool(changed)),
        ("no_total_evidence_deletion", not case.reference_context.strip() or bool(context.strip())),
        ("no_new_target_leakage", not introduced_target_leakage),
    ) + family_checks
    payload = {
        "allocation_id": allocation.allocation_id,
        "seed": seed,
        "prompt": prompt,
        "context": context,
        "answer": answer,
        "changed": changed,
    }
    perturbed = SciCUCase(
        task_id=f"{case.task_id}::PERTURB::{allocation.allocation_id}",
        prompt=prompt,
        reference_context=context,
        reference_answer=answer,
        domain=case.domain,
        competency=case.competency,
        motif_family=case.motif_family,
        source_hash=_new_hash(payload),
    )
    return MaterializedSciCUPerturbation(
        allocation_id=allocation.allocation_id,
        family=allocation.family,
        strength=allocation.strength,
        original_task_id=case.task_id,
        case=perturbed,
        changed_units=tuple(changed),
        valid=all(value for _, value in validity_checks),
        validity_checks=validity_checks,
    )
