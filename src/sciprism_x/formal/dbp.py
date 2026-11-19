from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math

import numpy as np

from sciprism_x.baselines.schema import EvaluatedCase, GraphEdge, GraphView
from sciprism_x.config import RunConfig
from sciprism_x.metrics.structural import compute_scs, compute_tp
from sciprism_x.transport.solver import solve_transport

from .allocation import AllocationTable, build_formal_allocations
from .datasets import DBPLanguageDataset, FormalDatasets
from .evaluator import MethodParameters, fixture_method_parameters
from .perturbed_cases import MaterializedDBPPerturbation
from .schema import ScaleContract


BASE_FEATURE_COUNT = 7


@dataclass(frozen=True, slots=True)
class DBPContext:
    dataset: DBPLanguageDataset
    config: RunConfig
    parameters: MethodParameters
    source_signatures: dict[int, np.ndarray] = field(compare=False, repr=False)
    target_signatures: dict[int, np.ndarray] = field(compare=False, repr=False)
    source_relation_presence: dict[int, np.ndarray] = field(compare=False, repr=False)
    target_relation_presence: dict[int, np.ndarray] = field(compare=False, repr=False)
    target_candidate_ids: tuple[int, ...]
    target_candidate_matrix: np.ndarray = field(compare=False, repr=False)
    admissible_pairs: frozenset[tuple[int, int]]
    reference_pairs: frozenset[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class DBPCaseResult:
    language_pair: str
    source_id: int
    target_id: int
    candidate_count: int
    candidate_pruning: bool
    correct_rank: int
    scs: float
    transport_purity: float
    hits_at_1: float
    mrr: float
    allocation_id: str = ""
    regime: str = "direct"
    family: str | None = None
    strength: float | None = None


@dataclass(frozen=True, slots=True)
class DBPManifest:
    nonstress_counts: dict[str, int]
    boundary_count: int
    language_pairs: tuple[str, ...]
    allocation_count: int


def _graph_maps(
    entities: tuple[tuple[int, str], ...],
    triples: tuple[tuple[int, int, int], ...],
):
    entity_ids = tuple(identifier for identifier, _ in entities)
    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    undirected: dict[int, set[int]] = defaultdict(set)
    for subject, relation, object_ in triples:
        outgoing[subject].append((relation, object_))
        incoming[object_].append((relation, subject))
        undirected[subject].add(object_)
        undirected[object_].add(subject)
    return entity_ids, outgoing, incoming, undirected


def _neighborhood_count(
    entity: int,
    undirected: dict[int, set[int]],
    hops: int,
) -> int:
    seen = {entity}
    frontier = {entity}
    for _ in range(hops):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in undirected.get(node, set())
            if neighbor not in seen
        }
        seen.update(frontier)
    return len(seen) - 1


def _raw_signatures(
    entities: tuple[tuple[int, str], ...],
    triples: tuple[tuple[int, int, int], ...],
    relation_index: dict[tuple[str, int], int],
    neighborhood_hops: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    entity_ids, outgoing, incoming, undirected = _graph_maps(entities, triples)
    signatures: dict[int, np.ndarray] = {}
    presence: dict[int, np.ndarray] = {}
    for entity in entity_ids:
        out = outgoing.get(entity, ())
        inc = incoming.get(entity, ())
        relations = {relation for relation, _ in out} | {relation for relation, _ in inc}
        vector = np.zeros(BASE_FEATURE_COUNT + len(relation_index), dtype=np.float64)
        vector[:BASE_FEATURE_COUNT] = (
            len(out),
            len(inc),
            len(out) + len(inc),
            len(relations),
            len(undirected.get(entity, set())),
            _neighborhood_count(entity, undirected, neighborhood_hops),
            float(not out and not inc),
        )
        relation_presence = np.zeros(len(relation_index), dtype=np.bool_)
        for relation, _ in out:
            index = relation_index[("out", relation)]
            vector[BASE_FEATURE_COUNT + index] += 1.0
            relation_presence[index] = True
        for relation, _ in inc:
            index = relation_index[("in", relation)]
            vector[BASE_FEATURE_COUNT + index] += 1.0
            relation_presence[index] = True
        signatures[entity] = vector
        presence[entity] = relation_presence
    return signatures, presence


def _normalize_signatures(
    source: dict[int, np.ndarray],
    target: dict[int, np.ndarray],
) -> None:
    matrix = np.vstack((*source.values(), *target.values()))
    scale = np.maximum(matrix[:, :BASE_FEATURE_COUNT].max(axis=0), 1.0)
    for vector in (*source.values(), *target.values()):
        vector[:BASE_FEATURE_COUNT] /= scale
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm


def build_dbp_context(
    dataset: DBPLanguageDataset,
    config: RunConfig,
    parameters: MethodParameters | None = None,
) -> DBPContext:
    relations = sorted(
        {relation for _, relation, _ in dataset.triples_1 + dataset.triples_2}
    )
    relation_index = {
        (direction, relation): index
        for index, (direction, relation) in enumerate(
            (pair for relation in relations for pair in (("out", relation), ("in", relation)))
        )
    }
    source, source_presence = _raw_signatures(
        dataset.entity_ids_1,
        dataset.triples_1,
        relation_index,
        config.dbp_reconstruction.signature_neighborhood_hops,
    )
    target, target_presence = _raw_signatures(
        dataset.entity_ids_2,
        dataset.triples_2,
        relation_index,
        config.dbp_reconstruction.signature_neighborhood_hops,
    )
    _normalize_signatures(source, target)
    candidates = dataset.target_candidate_ids
    missing = set(candidates).difference(target)
    if missing:
        raise ValueError(f"DBP target candidates are absent from entity IDs: {len(missing)}")
    return DBPContext(
        dataset=dataset,
        config=config,
        parameters=parameters or fixture_method_parameters(config),
        source_signatures=source,
        target_signatures=target,
        source_relation_presence=source_presence,
        target_relation_presence=target_presence,
        target_candidate_ids=candidates,
        target_candidate_matrix=np.vstack([target[identifier] for identifier in candidates]),
        # Appendix D.4: the 30% seed alignments are the only correspondence
        # evidence available to the method.  Frozen test/reference pairs are
        # retained separately and are used only by SCS/TP evaluation.
        admissible_pairs=frozenset(dataset.seed_pairs),
        reference_pairs=frozenset(dataset.reference_pairs),
    )


def _relation_jaccard(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    intersections = targets @ source.astype(np.int64)
    unions = targets.sum(axis=1) + source.sum() - intersections
    return np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=np.float64),
        where=unions > 0,
    )


def ranking_metrics(scores: np.ndarray, correct_index: int) -> tuple[float, float, int]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    rank = int(np.flatnonzero(order == correct_index)[0]) + 1
    return float(rank == 1), 1.0 / rank, rank


def _local_nodes(
    triples: tuple[tuple[int, int, int], ...],
    center: int,
    hops: int,
) -> tuple[int, ...]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for subject, _, object_ in triples:
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)
    seen = {center}
    queue = deque(((center, 0),))
    while queue:
        node, depth = queue.popleft()
        if depth == hops:
            continue
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))
    return tuple(sorted(seen))


def _adjacency(
    nodes: tuple[int, ...],
    triples: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    index = {node: position for position, node in enumerate(nodes)}
    matrix = np.zeros((1, len(nodes), len(nodes)), dtype=np.float64)
    for subject, _, object_ in triples:
        if subject in index and object_ in index:
            matrix[0, index[subject], index[object_]] = 1.0
    return matrix


def evaluate_dbp_case(source_id: int, target_id: int, context: DBPContext) -> DBPCaseResult:
    if source_id not in context.source_signatures:
        raise ValueError("DBP source entity is absent")
    try:
        correct_index = context.target_candidate_ids.index(target_id)
    except ValueError as error:
        raise ValueError("DBP target entity is absent from the full candidate pool") from error
    source_signature = context.source_signatures[source_id]
    cosine = np.clip(context.target_candidate_matrix @ source_signature, 0.0, 1.0)
    target_presence = np.vstack(
        [context.target_relation_presence[value] for value in context.target_candidate_ids]
    )
    relation_overlap = _relation_jaccard(
        context.source_relation_presence[source_id],
        target_presence,
    )
    reconstruction = context.config.dbp_reconstruction
    scores = (
        reconstruction.signature_cosine_weight * cosine
        + reconstruction.relation_overlap_weight * relation_overlap
    )
    hits, mrr, rank = ranking_metrics(scores, correct_index)

    dataset = context.dataset
    local_hops = context.config.dbp_reconstruction.local_transport_hops
    source_nodes = _local_nodes(dataset.triples_1, source_id, local_hops)
    target_nodes = _local_nodes(dataset.triples_2, target_id, local_hops)
    source_matrix = np.vstack([context.source_signatures[node] for node in source_nodes])
    target_matrix = np.vstack([context.target_signatures[node] for node in target_nodes])
    node_cost = np.clip(1.0 - source_matrix @ target_matrix.T, 0.0, 1.0)
    reference = np.zeros(node_cost.shape, dtype=np.bool_)
    soft = np.ones(node_cost.shape, dtype=np.float64)
    source_index = {node: index for index, node in enumerate(source_nodes)}
    target_index = {node: index for index, node in enumerate(target_nodes)}
    for left, right in context.admissible_pairs:
        if left in source_index and right in target_index:
            soft[source_index[left], target_index[right]] = 0.0
    for left, right in context.reference_pairs:
        if left in source_index and right in target_index:
            reference[source_index[left], target_index[right]] = True
    config = context.config.transport
    transport = solve_transport(
        node_cost=node_cost,
        source_mass=np.full(len(source_nodes), 1.0 / len(source_nodes)),
        target_mass=np.full(len(target_nodes), 1.0 / len(target_nodes)),
        hard_mask=np.ones(node_cost.shape, dtype=np.bool_),
        epsilon=context.parameters.epsilon,
        max_iterations=config.max_iterations,
        marginal_tolerance=config.marginal_tolerance,
        objective_tolerance=config.objective_tolerance,
        stable_iterations=config.stable_iterations,
        source_adjacency=_adjacency(source_nodes, dataset.triples_1),
        target_adjacency=_adjacency(target_nodes, dataset.triples_2),
        soft_constraint_cost=soft,
        lambda_structure=context.parameters.lambda_structure,
        lambda_constraint=context.parameters.lambda_constraint,
    )
    if transport.objective is None or not transport.diagnostics.feasible:
        raise RuntimeError("DBP local transport unexpectedly became infeasible")
    # The paper treats Hits@1/MRR as auxiliary entity-ranking diagnostics and
    # SCS as the primary constrained-correspondence measure.  Exclude the
    # entropic regularizer from the reported structural score so that SCS is a
    # bounded transform of node, relational, and admissibility costs rather
    # than of the unrelated full-pool ranking heuristic above.
    correspondence_cost = (
        transport.objective.node
        + context.parameters.lambda_structure * transport.objective.structural
        + context.parameters.lambda_constraint * transport.objective.soft_constraint
    )
    scs = math.exp(-max(float(correspondence_cost), 0.0))
    return DBPCaseResult(
        language_pair=dataset.language_pair,
        source_id=source_id,
        target_id=target_id,
        candidate_count=len(context.target_candidate_ids),
        candidate_pruning=dataset.candidate_pruning,
        correct_rank=rank,
        scs=compute_scs((scs,)),
        transport_purity=compute_tp(transport.plan, reference),
        hits_at_1=hits,
        mrr=mrr,
    )


def _baseline_graph(
    nodes: tuple[int, ...],
    triples: tuple[tuple[int, int, int], ...],
) -> GraphView:
    node_set = set(nodes)
    return GraphView(
        nodes=tuple(str(node) for node in nodes),
        edges=tuple(
            GraphEdge(str(subject), str(relation), str(object_))
            for subject, relation, object_ in triples
            if subject in node_set and object_ in node_set
        ),
    )


def build_dbp_baseline_cases(
    datasets: FormalDatasets,
    allocations: AllocationTable,
    config: RunConfig,
    *,
    perturbations: Mapping[str, MaterializedDBPPerturbation] | None = None,
    sinkhorn_epsilon: float,
) -> tuple[EvaluatedCase, ...]:
    """Build comparator inputs from DBP15K evidence, never SciPRISM-X outputs."""

    base_contexts = {
        language: build_dbp_context(dataset, config)
        for language, dataset in datasets.dbp15k.language_pairs.items()
    }
    output: list[EvaluatedCase] = []
    for allocation in allocations.rows:
        if allocation.language_pair is None:
            raise ValueError("DBP allocation is missing its language pair")
        _, source_text, target_text = allocation.case_id.split("::")
        source_id, target_id = int(source_text), int(target_text)
        context = base_contexts[allocation.language_pair]
        if allocation.regime == "boundary_stress":
            if perturbations is None or allocation.allocation_id not in perturbations:
                raise ValueError(
                    f"DBP boundary allocation lacks a materialized perturbation: "
                    f"{allocation.allocation_id}"
                )
            perturbation = perturbations[allocation.allocation_id]
            original = datasets.dbp15k.language_pairs[allocation.language_pair]
            context = build_dbp_context(
                replace(
                    original,
                    triples_1=perturbation.triples_1,
                    triples_2=perturbation.triples_2,
                    seed_pairs=perturbation.seed_pairs,
                ),
                config,
            )

        correct_index = context.target_candidate_ids.index(target_id)
        source_signature = context.source_signatures[source_id]
        cosine = np.clip(context.target_candidate_matrix @ source_signature, 0.0, 1.0)
        target_presence = np.vstack(
            [
                context.target_relation_presence[value]
                for value in context.target_candidate_ids
            ]
        )
        relation_overlap = _relation_jaccard(
            context.source_relation_presence[source_id],
            target_presence,
        )
        reconstruction = config.dbp_reconstruction
        candidate_scores = (
            reconstruction.signature_cosine_weight * cosine
            + reconstruction.relation_overlap_weight * relation_overlap
        )
        predicted_index = int(np.argmax(candidate_scores))
        predicted_target = context.target_candidate_ids[predicted_index]

        local_hops = config.dbp_reconstruction.local_transport_hops
        source_nodes = _local_nodes(context.dataset.triples_1, source_id, local_hops)
        target_nodes = _local_nodes(context.dataset.triples_2, target_id, local_hops)
        source_matrix = np.vstack(
            [context.source_signatures[node] for node in source_nodes]
        )
        target_matrix = np.vstack(
            [context.target_signatures[node] for node in target_nodes]
        )
        pairwise_cost = np.clip(1.0 - source_matrix @ target_matrix.T, 0.0, 1.0)
        source_mass = np.full(len(source_nodes), 1.0 / len(source_nodes))
        target_mass = np.full(len(target_nodes), 1.0 / len(target_nodes))
        baseline_transport = solve_transport(
            node_cost=pairwise_cost,
            source_mass=source_mass,
            target_mass=target_mass,
            hard_mask=np.ones(pairwise_cost.shape, dtype=np.bool_),
            epsilon=sinkhorn_epsilon,
            max_iterations=config.transport.max_iterations,
            marginal_tolerance=config.transport.marginal_tolerance,
            objective_tolerance=config.transport.objective_tolerance,
            stable_iterations=config.transport.stable_iterations,
            source_adjacency=None,
            target_adjacency=None,
            soft_constraint_cost=np.zeros(pairwise_cost.shape, dtype=np.float64),
            lambda_structure=0.0,
            lambda_constraint=0.0,
        )
        if baseline_transport.objective is None:
            raise RuntimeError("DBP Sinkhorn baseline transport is infeasible")
        reference = np.zeros(pairwise_cost.shape, dtype=np.bool_)
        source_index = {node: index for index, node in enumerate(source_nodes)}
        target_index = {node: index for index, node in enumerate(target_nodes)}
        for left, right in context.reference_pairs:
            if left in source_index and right in target_index:
                reference[source_index[left], target_index[right]] = True
        output.append(
            EvaluatedCase(
                case_id=(
                    f"{allocation.allocation_id}::{allocation.language_pair}::"
                    f"{source_id}::{target_id}"
                ),
                dataset="dbp15k",
                task=f"Align entity {source_id}",
                response=str(predicted_target),
                reference_context=allocation.language_pair,
                predicted_target=str(predicted_target),
                reference_target=str(target_id),
                response_segments=(str(predicted_target),),
                evidence_spans=(allocation.language_pair,),
                task_metadata=(("language_pair", allocation.language_pair),),
                confidence=float(np.clip(candidate_scores[predicted_index], 0.0, 1.0)),
                entropy=float(1.0 - np.clip(candidate_scores[predicted_index], 0.0, 1.0)),
                representation=np.array(source_signature, dtype=np.float64, copy=True),
                validation_support=np.vstack((source_signature, source_signature)),
                source_graph=_baseline_graph(source_nodes, context.dataset.triples_1),
                target_graph=_baseline_graph(target_nodes, context.dataset.triples_2),
                pairwise_cost=pairwise_cost,
                source_mass=source_mass,
                target_mass=target_mass,
                source_adjacency=_adjacency(source_nodes, context.dataset.triples_1),
                target_adjacency=_adjacency(target_nodes, context.dataset.triples_2),
                candidate_scores=np.array(candidate_scores, dtype=np.float64, copy=True),
                correct_candidate_index=correct_index,
                residuals=np.zeros(6, dtype=np.float64),
                transport_plan=baseline_transport.plan,
                generator_identity="withheld-generator",
                motif_graph="withheld-motif-graph",
                scientific_constraint_mask=reference,
                boundary_label="withheld-boundary",
                open_set_category="withheld-open-set",
            )
        )
    return tuple(output)


def build_dbp_manifest(
    datasets: FormalDatasets,
    contract: ScaleContract,
    *,
    seed: int,
) -> DBPManifest:
    allocations = build_formal_allocations(datasets, contract, seed=seed).dbp
    return DBPManifest(
        nonstress_counts={
            regime: allocations.count_regime(regime)
            for regime in ("iid", "motif_or_structure", "domain_transfer")
        },
        boundary_count=allocations.count_regime("boundary_stress"),
        language_pairs=tuple(sorted(datasets.dbp15k.language_pairs)),
        allocation_count=len(allocations.rows),
    )


def run_dbp_experiment(
    datasets: FormalDatasets,
    allocations: AllocationTable,
    config: RunConfig,
    *,
    perturbations: Mapping[str, MaterializedDBPPerturbation] | None = None,
    parameters: MethodParameters | None = None,
) -> tuple[DBPCaseResult, ...]:
    contexts = {
        language: build_dbp_context(dataset, config, parameters)
        for language, dataset in datasets.dbp15k.language_pairs.items()
    }
    results = []
    for row in allocations.rows:
        if row.language_pair is None:
            raise ValueError("DBP allocation is missing its language pair")
        _, source, target = row.case_id.split("::")
        context = contexts[row.language_pair]
        if row.regime == "boundary_stress":
            if perturbations is None or row.allocation_id not in perturbations:
                raise ValueError(
                    f"DBP boundary allocation lacks a materialized perturbation: {row.allocation_id}"
                )
            perturbation = perturbations[row.allocation_id]
            original = datasets.dbp15k.language_pairs[row.language_pair]
            perturbed = replace(
                original,
                triples_1=perturbation.triples_1,
                triples_2=perturbation.triples_2,
                seed_pairs=perturbation.seed_pairs,
            )
            context = build_dbp_context(perturbed, config, parameters)
        results.append(
            replace(
                evaluate_dbp_case(int(source), int(target), context),
                allocation_id=row.allocation_id,
                regime=row.regime,
                family=row.family,
                strength=row.strength,
            )
        )
    return tuple(results)
