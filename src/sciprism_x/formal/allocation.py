from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sciprism_x.data.schema import SciCUCase

from .datasets import DBPLanguageDataset, FormalDatasets
from .schema import ScaleContract


SCICU_FAMILIES = (
    "evidence_removal",
    "context_composition_shift",
    "modality_conflict",
    "symbolic_depth_increase",
    "boundary_condition_change",
)
DBP_FAMILIES = (
    "neighborhood_degradation",
    "relation_ambiguity_injection",
    "seed_alignment_reduction",
    "cross_graph_evidence_weakening",
)
STRENGTHS = (0.25, 0.50, 1.00)
NONSTRESS_REGIMES = ("iid", "motif_or_structure", "domain_transfer")


def largest_remainder(total: int, capacities: Mapping[str, int]) -> dict[str, int]:
    if total < 0 or any(value < 0 for value in capacities.values()):
        raise ValueError("allocation counts must be nonnegative")
    capacity_sum = sum(capacities.values())
    if total > capacity_sum:
        raise ValueError(f"cannot allocate {total} rows into capacity {capacity_sum}")
    if total == 0:
        return {key: 0 for key in capacities}
    raw = {key: total * value / capacity_sum for key, value in capacities.items()}
    result = {key: min(capacities[key], int(np.floor(raw[key]))) for key in capacities}
    while sum(result.values()) < total:
        candidates = tuple(
            key for key in capacities if result[key] < capacities[key]
        )
        if not candidates:
            raise ValueError("allocation cannot satisfy requested total")
        key = min(candidates, key=lambda item: (-(raw[item] - result[item]), item))
        result[key] += 1
    return result


def _rank(seed: int, namespace: str, identifier: str) -> bytes:
    return hashlib.sha256(f"{seed}::{namespace}::{identifier}".encode("utf-8")).digest()


def _balanced_scicu_sample(
    cases: Sequence[SciCUCase],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[SciCUCase, ...]:
    groups: dict[str, list[SciCUCase]] = defaultdict(list)
    for case in cases:
        groups[f"{case.domain}::{case.competency}"].append(case)
    quotas = largest_remainder(count, {key: len(rows) for key, rows in groups.items()})
    selected: list[SciCUCase] = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda case: (_rank(seed, namespace, case.task_id), case.task_id),
        )
        selected.extend(ranked[: quotas[key]])
    return tuple(sorted(selected, key=lambda case: case.task_id))


@dataclass(frozen=True, slots=True)
class AllocationRow:
    dataset: Literal["scicueval", "dbp15k"]
    allocation_id: str
    case_id: str
    regime: Literal["iid", "motif_or_structure", "domain_transfer", "boundary_stress"]
    family: str | None
    strength: float | None
    language_pair: str | None
    bsi_regime: str | None = None


@dataclass(frozen=True, slots=True)
class AllocationTable:
    rows: tuple[AllocationRow, ...]

    def count_regime(self, regime: str) -> int:
        return sum(row.regime == regime for row in self.rows)

    def family_strength_counts(self) -> dict[tuple[str, float], int]:
        return dict(
            Counter(
                (row.family, row.strength)
                for row in self.rows
                if row.regime == "boundary_stress"
                and row.family is not None
                and row.strength is not None
            )
        )

    def family_strength_language_counts(self) -> dict[tuple[str, str, float], int]:
        return dict(
            Counter(
                (row.language_pair, row.family, row.strength)
                for row in self.rows
                if row.regime == "boundary_stress"
                and row.language_pair is not None
                and row.family is not None
                and row.strength is not None
            )
        )


@dataclass(frozen=True, slots=True)
class FormalAllocations:
    scicu: AllocationTable
    dbp: AllocationTable


def _scicu_allocations(
    datasets: FormalDatasets,
    contract: ScaleContract,
    seed: int,
    strengths: tuple[float, ...],
) -> AllocationTable:
    rows: list[AllocationRow] = []
    cases = tuple(datasets.scicueval)
    domains = sorted(
        {case.domain for case in cases},
        key=lambda value: (_rank(seed, "scicu::domain-holdout", value), value),
    )
    heldout_domains: list[str] = []
    heldout_capacity = 0
    for domain in domains:
        heldout_domains.append(domain)
        heldout_capacity += sum(case.domain == domain for case in cases)
        if heldout_capacity >= contract.scicu_nonstress_per_regime:
            break
    domain_pool = tuple(case for case in cases if case.domain in heldout_domains)
    remaining_domains = tuple(case for case in cases if case.domain not in heldout_domains)

    combinations: dict[str, list[SciCUCase]] = defaultdict(list)
    for case in remaining_domains:
        combinations[f"{case.domain}::{case.motif_family}"].append(case)
    ranked_combinations = sorted(
        combinations,
        key=lambda value: (_rank(seed, "scicu::motif-holdout", value), value),
    )
    heldout_combinations: list[str] = []
    heldout_motif_capacity = 0
    for combination in ranked_combinations:
        heldout_combinations.append(combination)
        heldout_motif_capacity += len(combinations[combination])
        if heldout_motif_capacity >= contract.scicu_nonstress_per_regime:
            break
    motif_pool = tuple(
        case
        for key in heldout_combinations
        for case in combinations[key]
    )
    heldout_combination_set = set(heldout_combinations)
    iid_pool = tuple(
        case
        for case in remaining_domains
        if f"{case.domain}::{case.motif_family}" not in heldout_combination_set
    )
    regime_pools = {
        "iid": iid_pool,
        "motif_or_structure": motif_pool,
        "domain_transfer": domain_pool,
    }
    selected_by_regime: dict[str, tuple[SciCUCase, ...]] = {}
    for regime in NONSTRESS_REGIMES:
        selected = _balanced_scicu_sample(
            regime_pools[regime],
            contract.scicu_nonstress_per_regime,
            seed=seed,
            namespace=f"scicu::{regime}",
        )
        selected_by_regime[regime] = selected
        rows.extend(
            AllocationRow(
                dataset="scicueval",
                allocation_id=f"scicueval::{regime}::{index:05d}",
                case_id=case.task_id,
                regime=regime,
                family=None,
                strength=None,
                language_pair=None,
            )
            for index, case in enumerate(selected)
        )

    bsi_regimes = (
        NONSTRESS_REGIMES + ("boundary_stress",)
        if contract.scale == "paper"
        else ("pooled_smoke",)
    )
    pooled_base = tuple(
        case
        for regime in NONSTRESS_REGIMES
        for case in selected_by_regime[regime]
    )
    boundary_pools = selected_by_regime | {
        "boundary_stress": pooled_base,
        "pooled_smoke": pooled_base,
    }
    cell_count = len(bsi_regimes) * len(SCICU_FAMILIES) * len(strengths)
    per_cell = contract.scicu_boundary // cell_count
    if per_cell * cell_count != contract.scicu_boundary:
        raise ValueError(
            "SciCUEval boundary count must divide across BSI-regime, family, "
            "and strength cells"
        )
    for bsi_regime in bsi_regimes:
        for family in SCICU_FAMILIES:
            selected = _balanced_scicu_sample(
                boundary_pools[bsi_regime],
                per_cell,
                seed=seed,
                namespace=f"scicu::boundary::{bsi_regime}::{family}",
            )
            for strength in strengths:
                rows.extend(
                    AllocationRow(
                        dataset="scicueval",
                        allocation_id=(
                            f"scicueval::boundary::{family}::{strength:.2f}::"
                            f"{bsi_regime}::{index:05d}"
                        ),
                        case_id=case.task_id,
                        regime="boundary_stress",
                        family=family,
                        strength=strength,
                        language_pair=None,
                        bsi_regime=bsi_regime,
                    )
                    for index, case in enumerate(selected)
                )
    return AllocationTable(tuple(rows))


def _dbp_case_id(language_pair: str, pair: tuple[int, int]) -> str:
    return f"{language_pair}::{pair[0]}::{pair[1]}"


def _ranked_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    count: int,
    seed: int,
    namespace: str,
    language_pair: str,
) -> tuple[tuple[int, int], ...]:
    ranked = sorted(
        pairs,
        key=lambda pair: (
            _rank(seed, namespace, _dbp_case_id(language_pair, pair)),
            pair,
        ),
    )
    return tuple(ranked[:count])


def _dbp_structure_scores(
    dataset: DBPLanguageDataset,
) -> dict[tuple[int, int], float]:
    degree_1: Counter[int] = Counter()
    degree_2: Counter[int] = Counter()
    relations_1: dict[int, set[int]] = defaultdict(set)
    relations_2: dict[int, set[int]] = defaultdict(set)
    for source, relation, target in dataset.triples_1:
        degree_1[source] += 1
        degree_1[target] += 1
        relations_1[source].add(relation)
        relations_1[target].add(relation)
    for source, relation, target in dataset.triples_2:
        degree_2[source] += 1
        degree_2[target] += 1
        relations_2[source].add(relation)
        relations_2[target].add(relation)
    scores = {}
    for source_id, target_id in dataset.reference_pairs:
        left_degree = degree_1[source_id]
        right_degree = degree_2[target_id]
        degree_shift = abs(np.log1p(left_degree) - np.log1p(right_degree))
        left_relations = len(relations_1[source_id])
        right_relations = len(relations_2[target_id])
        relation_shift = abs(left_relations - right_relations) / max(
            1, left_relations, right_relations
        )
        scores[(source_id, target_id)] = float(degree_shift + relation_shift)
    return scores


def _dbp_allocations(
    datasets: FormalDatasets,
    contract: ScaleContract,
    seed: int,
    strengths: tuple[float, ...],
) -> AllocationTable:
    rows: list[AllocationRow] = []
    language_pairs = tuple(sorted(datasets.dbp15k.language_pairs))
    capacities = {
        language: len(datasets.dbp15k.language_pairs[language].reference_pairs)
        for language in language_pairs
    }
    quotas = largest_remainder(contract.dbp_nonstress_per_regime, capacities)
    selected_by_regime: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
        regime: {} for regime in NONSTRESS_REGIMES
    }
    for language in language_pairs:
        dataset = datasets.dbp15k.language_pairs[language]
        remaining = tuple(dataset.reference_pairs)
        domain_transfer = _ranked_pairs(
            remaining,
            count=quotas[language],
            seed=seed,
            namespace="dbp::domain-transfer-language-holdout",
            language_pair=language,
        )
        domain_set = set(domain_transfer)
        remaining = tuple(pair for pair in remaining if pair not in domain_set)
        structure_scores = _dbp_structure_scores(dataset)
        structure = tuple(
            sorted(
                remaining,
                key=lambda pair: (
                    -structure_scores[pair],
                    _rank(
                        seed,
                        "dbp::structure-holdout",
                        _dbp_case_id(language, pair),
                    ),
                    pair,
                ),
            )[: quotas[language]]
        )
        structure_set = set(structure)
        remaining = tuple(pair for pair in remaining if pair not in structure_set)
        iid = _ranked_pairs(
            remaining,
            count=quotas[language],
            seed=seed,
            namespace="dbp::iid",
            language_pair=language,
        )
        selected_by_regime["iid"][language] = iid
        selected_by_regime["motif_or_structure"][language] = structure
        selected_by_regime["domain_transfer"][language] = domain_transfer

    selected_nonstress: dict[str, list[tuple[int, int]]] = {
        language: [] for language in language_pairs
    }
    for regime in NONSTRESS_REGIMES:
        for language in language_pairs:
            selected = selected_by_regime[regime][language]
            rows.extend(
                AllocationRow(
                    dataset="dbp15k",
                    allocation_id=f"dbp15k::{regime}::{language}::{index:05d}",
                    case_id=_dbp_case_id(language, pair),
                    regime=regime,
                    family=None,
                    strength=None,
                    language_pair=language,
                )
                for index, pair in enumerate(selected)
            )
            selected_nonstress[language].extend(selected)

    if contract.dbp_boundary % len(strengths) != 0:
        raise ValueError("DBP boundary count must divide across perturbation strengths")
    cells = tuple(
        (language, family)
        for language in language_pairs
        for family in DBP_FAMILIES
    )
    cell_capacities = {
        f"{language}::{family}": len(selected_nonstress[language])
        for language, family in cells
    }
    quotas = largest_remainder(
        contract.dbp_boundary // len(strengths),
        cell_capacities,
    )
    for language, family in cells:
        key = f"{language}::{family}"
        selected = _ranked_pairs(
            tuple(selected_nonstress[language]),
            count=quotas[key],
            seed=seed,
            namespace=f"dbp::boundary::{family}",
            language_pair=language,
        )
        for strength in strengths:
            rows.extend(
                AllocationRow(
                    dataset="dbp15k",
                    allocation_id=f"dbp15k::boundary::{language}::{family}::{strength:.2f}::{index:05d}",
                    case_id=_dbp_case_id(language, pair),
                    regime="boundary_stress",
                    family=family,
                    strength=strength,
                    language_pair=language,
                )
                for index, pair in enumerate(selected)
            )
    return AllocationTable(tuple(rows))


def build_formal_allocations(
    datasets: FormalDatasets,
    contract: ScaleContract,
    *,
    seed: int,
    strengths: Sequence[float] = STRENGTHS,
) -> FormalAllocations:
    resolved_strengths = tuple(float(value) for value in strengths)
    if not resolved_strengths or any(value <= 0.0 for value in resolved_strengths):
        raise ValueError("formal perturbation strengths must be positive")
    return FormalAllocations(
        scicu=_scicu_allocations(datasets, contract, seed, resolved_strengths),
        dbp=_dbp_allocations(datasets, contract, seed, resolved_strengths),
    )
