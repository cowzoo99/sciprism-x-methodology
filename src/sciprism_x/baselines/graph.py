from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar, Literal

import networkx as nx
import numpy as np

from sciprism_x.transport.solver import solve_transport

from .schema import (
    BaselineKind,
    BaselineResult,
    BaselineStatus,
    GraphBaselineInput,
    RankingInput,
)


def _result(
    kind: BaselineKind,
    metric: str,
    score: float,
    details: tuple[tuple[str, str], ...] = (),
) -> BaselineResult:
    return BaselineResult(kind, BaselineStatus.AVAILABLE, metric, float(score), details)


def _transport_score(
    *,
    kind: BaselineKind,
    value: GraphBaselineInput,
    epsilon: float,
    max_iterations: int,
    marginal_tolerance: float,
    objective_tolerance: float,
    stable_iterations: int,
    score_transform: Literal["exp_negative_cost"],
    node_cost: np.ndarray,
    source_adjacency: np.ndarray | None,
    target_adjacency: np.ndarray | None,
    lambda_structure: float,
) -> BaselineResult:
    if score_transform != "exp_negative_cost":
        raise ValueError("unsupported score transform")
    result = solve_transport(
        node_cost=node_cost,
        source_mass=value.source_mass,
        target_mass=value.target_mass,
        hard_mask=np.ones(node_cost.shape, dtype=np.bool_),
        epsilon=epsilon,
        max_iterations=max_iterations,
        marginal_tolerance=marginal_tolerance,
        objective_tolerance=objective_tolerance,
        stable_iterations=stable_iterations,
        source_adjacency=source_adjacency,
        target_adjacency=target_adjacency,
        soft_constraint_cost=np.zeros_like(node_cost),
        lambda_structure=lambda_structure,
        lambda_constraint=0.0,
    )
    if result.objective is None:
        return BaselineResult(
            kind,
            BaselineStatus.NOT_APPLICABLE,
            "alignment_similarity",
            None,
            (("reason", result.diagnostics.reason),),
        )
    raw_cost = result.objective.node + lambda_structure * result.objective.structural
    return _result(
        kind,
        "alignment_similarity",
        math.exp(-raw_cost),
        (
            ("converged", str(result.diagnostics.converged).lower()),
            ("iterations", str(result.diagnostics.iterations)),
        ),
    )


@dataclass(frozen=True, slots=True)
class GraphEditAdapter:
    node_insertion_deletion_cost: float
    relation_insertion_deletion_cost: float
    relation_substitution_cost: float
    timeout_seconds: float

    kind: ClassVar[BaselineKind] = BaselineKind.GRAPH_EDIT
    input_type: ClassVar[type[GraphBaselineInput]] = GraphBaselineInput

    def evaluate(self, value: GraphBaselineInput) -> BaselineResult:
        if not isinstance(value, GraphBaselineInput):
            raise TypeError("GraphEditAdapter requires GraphBaselineInput")
        costs = (
            self.node_insertion_deletion_cost,
            self.relation_insertion_deletion_cost,
            self.relation_substitution_cost,
        )
        if any(cost < 0.0 for cost in costs) or self.timeout_seconds <= 0.0:
            raise ValueError("graph edit costs must be nonnegative")
        source = nx.MultiDiGraph()
        target = nx.MultiDiGraph()
        source.add_nodes_from(value.source_graph.nodes)
        target.add_nodes_from(value.target_graph.nodes)
        source.add_edges_from(
            (edge.source, edge.target, {"relation": edge.relation})
            for edge in value.source_graph.edges
        )
        target.add_edges_from(
            (edge.source, edge.target, {"relation": edge.relation})
            for edge in value.target_graph.edges
        )
        distance = nx.graph_edit_distance(
            source,
            target,
            node_subst_cost=lambda _left, _right: 0.0,
            node_del_cost=lambda _node: self.node_insertion_deletion_cost,
            node_ins_cost=lambda _node: self.node_insertion_deletion_cost,
            edge_subst_cost=lambda left, right: (
                0.0
                if left["relation"] == right["relation"]
                else self.relation_substitution_cost
            ),
            edge_del_cost=lambda _edge: self.relation_insertion_deletion_cost,
            edge_ins_cost=lambda _edge: self.relation_insertion_deletion_cost,
            roots=None,
            upper_bound=None,
            timeout=self.timeout_seconds,
        )
        if distance is None:
            return BaselineResult(
                self.kind,
                BaselineStatus.NOT_APPLICABLE,
                "graph_edit_similarity",
                None,
                (("reason", "graph_edit_timeout"),),
            )
        maximum = (
            self.node_insertion_deletion_cost
            * (source.number_of_nodes() + target.number_of_nodes())
            + self.relation_insertion_deletion_cost
            * (source.number_of_edges() + target.number_of_edges())
        )
        score = 1.0 if maximum == 0.0 else 1.0 - distance / maximum
        return _result(self.kind, "graph_edit_similarity", score)


@dataclass(frozen=True, slots=True)
class UnconstrainedSinkhornAdapter:
    epsilon: float
    max_iterations: int
    marginal_tolerance: float
    objective_tolerance: float
    stable_iterations: int
    score_transform: Literal["exp_negative_cost"]

    kind: ClassVar[BaselineKind] = BaselineKind.SINKHORN
    input_type: ClassVar[type[GraphBaselineInput]] = GraphBaselineInput

    def evaluate(self, value: GraphBaselineInput) -> BaselineResult:
        if not isinstance(value, GraphBaselineInput):
            raise TypeError("UnconstrainedSinkhornAdapter requires GraphBaselineInput")
        return _transport_score(
            kind=self.kind,
            value=value,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations,
            marginal_tolerance=self.marginal_tolerance,
            objective_tolerance=self.objective_tolerance,
            stable_iterations=self.stable_iterations,
            score_transform=self.score_transform,
            node_cost=np.asarray(value.pairwise_cost, dtype=np.float64),
            source_adjacency=None,
            target_adjacency=None,
            lambda_structure=0.0,
        )


@dataclass(frozen=True, slots=True)
class GromovTransportAdapter:
    epsilon: float
    lambda_structure: float
    max_iterations: int
    marginal_tolerance: float
    objective_tolerance: float
    stable_iterations: int
    score_transform: Literal["exp_negative_cost"]

    kind: ClassVar[BaselineKind] = BaselineKind.GROMOV
    input_type: ClassVar[type[GraphBaselineInput]] = GraphBaselineInput

    def evaluate(self, value: GraphBaselineInput) -> BaselineResult:
        if not isinstance(value, GraphBaselineInput):
            raise TypeError("GromovTransportAdapter requires GraphBaselineInput")
        return _transport_score(
            kind=self.kind,
            value=value,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations,
            marginal_tolerance=self.marginal_tolerance,
            objective_tolerance=self.objective_tolerance,
            stable_iterations=self.stable_iterations,
            score_transform=self.score_transform,
            node_cost=np.zeros_like(value.pairwise_cost, dtype=np.float64),
            source_adjacency=value.source_adjacency,
            target_adjacency=value.target_adjacency,
            lambda_structure=self.lambda_structure,
        )


@dataclass(frozen=True, slots=True)
class StructureAwareAdapter:
    epsilon: float
    lambda_structure: float
    max_iterations: int
    marginal_tolerance: float
    objective_tolerance: float
    stable_iterations: int
    score_transform: Literal["exp_negative_cost"]

    kind: ClassVar[BaselineKind] = BaselineKind.STRUCTURE_AWARE
    input_type: ClassVar[type[GraphBaselineInput]] = GraphBaselineInput

    def evaluate(self, value: GraphBaselineInput) -> BaselineResult:
        if not isinstance(value, GraphBaselineInput):
            raise TypeError("StructureAwareAdapter requires GraphBaselineInput")
        return _transport_score(
            kind=self.kind,
            value=value,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations,
            marginal_tolerance=self.marginal_tolerance,
            objective_tolerance=self.objective_tolerance,
            stable_iterations=self.stable_iterations,
            score_transform=self.score_transform,
            node_cost=np.asarray(value.pairwise_cost, dtype=np.float64),
            source_adjacency=value.source_adjacency,
            target_adjacency=value.target_adjacency,
            lambda_structure=self.lambda_structure,
        )


@dataclass(frozen=True, slots=True)
class RelationAwareAdapter:
    multiset: bool

    kind: ClassVar[BaselineKind] = BaselineKind.RELATION_AWARE
    input_type: ClassVar[type[GraphBaselineInput]] = GraphBaselineInput

    def evaluate(self, value: GraphBaselineInput) -> BaselineResult:
        if not isinstance(value, GraphBaselineInput):
            raise TypeError("RelationAwareAdapter requires GraphBaselineInput")
        source = Counter(edge.relation for edge in value.source_graph.edges)
        target = Counter(edge.relation for edge in value.target_graph.edges)
        if self.multiset:
            intersection = sum((source & target).values())
            union = sum((source | target).values())
        else:
            intersection = len(set(source) & set(target))
            union = len(set(source) | set(target))
        score = 1.0 if union == 0 else intersection / union
        return _result(self.kind, "relation_overlap", score)


def _rank(value: RankingInput, descending: bool) -> int:
    scores = np.asarray(value.candidate_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("candidate_scores must be a non-empty finite vector")
    if not 0 <= value.correct_candidate_index < scores.size:
        raise ValueError("correct_candidate_index is out of range")
    order = np.argsort(-scores if descending else scores, kind="stable")
    return int(np.where(order == value.correct_candidate_index)[0][0]) + 1


@dataclass(frozen=True, slots=True)
class HitsAt1Adapter:
    descending: bool

    kind: ClassVar[BaselineKind] = BaselineKind.HITS_AT_1
    input_type: ClassVar[type[RankingInput]] = RankingInput

    def evaluate(self, value: RankingInput) -> BaselineResult:
        if not isinstance(value, RankingInput):
            raise TypeError("HitsAt1Adapter requires RankingInput")
        return _result(self.kind, "hits_at_1", float(_rank(value, self.descending) == 1))


@dataclass(frozen=True, slots=True)
class MRRAdapter:
    descending: bool

    kind: ClassVar[BaselineKind] = BaselineKind.MRR
    input_type: ClassVar[type[RankingInput]] = RankingInput

    def evaluate(self, value: RankingInput) -> BaselineResult:
        if not isinstance(value, RankingInput):
            raise TypeError("MRRAdapter requires RankingInput")
        return _result(self.kind, "reciprocal_rank", 1.0 / _rank(value, self.descending))
