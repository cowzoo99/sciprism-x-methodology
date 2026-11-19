from __future__ import annotations

import math

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import MotifNode, TypedEdge


class SalienceWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inverse_position: float = Field(ge=0.0)
    motif_confidence: float = Field(ge=0.0)
    variable_reuse: float = Field(ge=0.0)
    graph_centrality: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_unit_sum(self) -> "SalienceWeights":
        if not math.isclose(
            sum(self.model_dump().values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("salience weights must sum to one")
        return self


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1.0e-12):
        constant = 0.0 if math.isclose(upper, 0.0, abs_tol=1.0e-12) else 1.0
        return [constant] * len(values)
    scale = upper - lower
    return [(value - lower) / scale for value in values]


def apply_salience(
    nodes: tuple[MotifNode, ...],
    edges: tuple[TypedEdge, ...],
    weights: SalienceWeights,
) -> tuple[MotifNode, ...]:
    if not nodes:
        return ()
    graph = nx.DiGraph()
    graph.add_nodes_from(node.node_id for node in nodes)
    graph.add_edges_from((edge.source, edge.target) for edge in edges)
    centrality = nx.betweenness_centrality(graph, normalized=True)

    variable_counts: dict[str, int] = {}
    for node in nodes:
        for variable in set(node.variables):
            variable_counts[variable] = variable_counts.get(variable, 0) + 1

    components = {
        "inverse_position": _normalize([1.0 / (index + 1) for index in range(len(nodes))]),
        "motif_confidence": _normalize([node.confidence for node in nodes]),
        "variable_reuse": _normalize(
            [
                float(sum(max(variable_counts[variable] - 1, 0) for variable in node.variables))
                for node in nodes
            ]
        ),
        "graph_centrality": _normalize(
            [float(centrality[node.node_id]) for node in nodes]
        ),
    }
    weight_values = weights.model_dump()
    return tuple(
        node.model_copy(
            update={
                "salience": sum(
                    weight_values[name] * components[name][index]
                    for name in components
                )
            }
        )
        for index, node in enumerate(nodes)
    )
