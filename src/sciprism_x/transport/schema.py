from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
BoolMatrix = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CostBundle:
    node_components: dict[str, FloatMatrix]
    node_cost: FloatMatrix
    source_adjacency: NDArray[np.float64]
    target_adjacency: NDArray[np.float64]
    soft_constraint_cost: FloatMatrix
    hard_mask: BoolMatrix
    source_mass: NDArray[np.float64]
    target_mass: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObjectiveComponents:
    node: float
    structural: float
    soft_constraint: float
    negative_entropy: float
    total: float


@dataclass(frozen=True, slots=True)
class TransportDiagnostics:
    feasible: bool
    converged: bool
    iterations: int
    marginal_error: float
    objective_change: float | None
    consecutive_stable_iterations: int
    reason: str


@dataclass(frozen=True, slots=True)
class TransportResult:
    plan: FloatMatrix
    objective: ObjectiveComponents | None
    diagnostics: TransportDiagnostics
