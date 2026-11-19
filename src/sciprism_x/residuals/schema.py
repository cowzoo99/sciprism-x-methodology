from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ResidualChannel(str, Enum):
    PHYSICAL = "physical"
    LOGICAL = "logical"
    UNIT_DIMENSIONAL = "unit_dimensional"
    CAUSAL = "causal"
    MODAL_EVIDENTIAL = "modal_evidential"
    UNCERTAINTY = "uncertainty"


@dataclass(frozen=True, slots=True)
class ResidualComponent:
    matched: float
    missing: float
    unsupported: float
    raw: float
    normalizer: float
    normalized: float


@dataclass(frozen=True, slots=True)
class ResidualVector:
    components: dict[ResidualChannel, ResidualComponent]

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.components[channel].normalized for channel in ResidualChannel],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ScoreBundle:
    tsr: float
    pcs: float
    residual_consistency: float
    node_loss: float
    structure_loss: float
    constraint_loss: float
    direction_warnings: tuple[str, ...]
