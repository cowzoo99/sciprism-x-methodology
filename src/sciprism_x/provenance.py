from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProvenanceClass(str, Enum):
    PAPER_EXPLICIT = "PAPER_EXPLICIT"
    OFFICIAL_SOURCE = "OFFICIAL_SOURCE"
    IMPLEMENTATION_SETTING = "IMPLEMENTATION_SETTING"
    UNAVAILABLE_EXTERNAL = "UNAVAILABLE_EXTERNAL"


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    path: str
    value: Any
    provenance: ProvenanceClass
