from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from .schema import (
    AttributionInput,
    BaselineKind,
    BaselineResult,
    BaselineStatus,
    EndpointInput,
    SupportInput,
    TaxonomyInput,
    UncertaintyInput,
)


def _available(kind: BaselineKind, metric: str, score: float, **details: str) -> BaselineResult:
    return BaselineResult(
        kind=kind,
        status=BaselineStatus.AVAILABLE,
        metric=metric,
        score=float(score),
        details=tuple(sorted(details.items())),
    )


@dataclass(frozen=True, slots=True)
class EndpointAdapter:
    case_sensitive: bool
    strip_whitespace: bool
    collapse_whitespace: bool

    kind: ClassVar[BaselineKind] = BaselineKind.ENDPOINT
    input_type: ClassVar[type[EndpointInput]] = EndpointInput

    def _normalize(self, value: str) -> str:
        result = value if self.case_sensitive else value.casefold()
        if self.strip_whitespace:
            result = result.strip()
        if self.collapse_whitespace:
            result = " ".join(result.split())
        return result

    def evaluate(self, value: EndpointInput) -> BaselineResult:
        if not isinstance(value, EndpointInput):
            raise TypeError("EndpointAdapter requires EndpointInput")
        score = float(
            self._normalize(value.predicted_target)
            == self._normalize(value.reference_target)
        )
        return _available(self.kind, "endpoint_correctness", score)


@dataclass(frozen=True, slots=True)
class AttributionAdapter:
    lowercase: bool
    token_pattern: str
    empty_overlap_score: float

    kind: ClassVar[BaselineKind] = BaselineKind.ATTRIBUTION
    input_type: ClassVar[type[AttributionInput]] = AttributionInput

    def _tokens(self, values: tuple[str, ...]) -> set[str]:
        text = " ".join(values)
        if self.lowercase:
            text = text.casefold()
        return set(re.findall(self.token_pattern, text))

    def evaluate(self, value: AttributionInput) -> BaselineResult:
        if not isinstance(value, AttributionInput):
            raise TypeError("AttributionAdapter requires AttributionInput")
        response = self._tokens(value.response_segments)
        evidence = self._tokens(value.evidence_spans)
        union = response | evidence
        score = self.empty_overlap_score if not union else len(response & evidence) / len(union)
        return _available(self.kind, "attribution_consistency", score)


@dataclass(frozen=True, slots=True)
class FixedTaxonomyAdapter:
    category_patterns: tuple[tuple[str, tuple[str, ...]], ...]
    no_failure_score: float
    failure_score: float
    lowercase: bool

    kind: ClassVar[BaselineKind] = BaselineKind.FIXED_TAXONOMY
    input_type: ClassVar[type[TaxonomyInput]] = TaxonomyInput

    def evaluate(self, value: TaxonomyInput) -> BaselineResult:
        if not isinstance(value, TaxonomyInput):
            raise TypeError("FixedTaxonomyAdapter requires TaxonomyInput")
        text = value.response.casefold() if self.lowercase else value.response
        for category, patterns in self.category_patterns:
            normalized = tuple(pattern.casefold() for pattern in patterns) if self.lowercase else patterns
            if any(pattern in text for pattern in normalized):
                return _available(
                    self.kind,
                    "fixed_failure_category",
                    self.failure_score,
                    category=category,
                )
        return _available(
            self.kind,
            "fixed_failure_category",
            self.no_failure_score,
            category="none",
        )


@dataclass(frozen=True, slots=True)
class UncertaintyAdapter:
    mode: Literal["confidence", "one_minus_entropy", "product"]
    maximum_entropy: float

    kind: ClassVar[BaselineKind] = BaselineKind.UNCERTAINTY
    input_type: ClassVar[type[UncertaintyInput]] = UncertaintyInput

    def evaluate(self, value: UncertaintyInput) -> BaselineResult:
        if not isinstance(value, UncertaintyInput):
            raise TypeError("UncertaintyAdapter requires UncertaintyInput")
        if self.maximum_entropy <= 0.0:
            raise ValueError("maximum_entropy must be positive")
        if not 0.0 <= value.confidence <= 1.0 or not 0.0 <= value.entropy <= self.maximum_entropy:
            raise ValueError("confidence or entropy is outside its declared range")
        entropy_score = 1.0 - value.entropy / self.maximum_entropy
        if self.mode == "confidence":
            score = value.confidence
        elif self.mode == "one_minus_entropy":
            score = entropy_score
        else:
            score = value.confidence * entropy_score
        return _available(self.kind, "calibrated_uncertainty", score, mode=self.mode)


@dataclass(frozen=True, slots=True)
class SupportDistanceAdapter:
    mode: Literal["knn", "centroid"]
    neighbors: int
    distance_scale: float
    clip_to_unit_interval: bool

    @property
    def kind(self) -> BaselineKind:
        return BaselineKind.KNN_SUPPORT if self.mode == "knn" else BaselineKind.CENTROID_SUPPORT

    input_type: ClassVar[type[SupportInput]] = SupportInput

    def evaluate(self, value: SupportInput) -> BaselineResult:
        if not isinstance(value, SupportInput):
            raise TypeError("SupportDistanceAdapter requires SupportInput")
        representation = np.asarray(value.representation, dtype=np.float64)
        support = np.asarray(value.validation_support, dtype=np.float64)
        if representation.ndim != 1 or support.ndim != 2 or support.shape[1] != representation.size:
            raise ValueError("representation and validation support shapes do not match")
        if not np.isfinite(representation).all() or not np.isfinite(support).all():
            raise ValueError("support features must be finite")
        if self.neighbors <= 0 or self.neighbors > support.shape[0] or self.distance_scale <= 0.0:
            raise ValueError("support-distance controls are invalid")
        if self.mode == "knn":
            distances = np.linalg.norm(support - representation[None, :], axis=1)
            distance = float(np.mean(np.partition(distances, self.neighbors - 1)[: self.neighbors]))
        else:
            distance = float(np.linalg.norm(representation - support.mean(axis=0)))
        score = distance / self.distance_scale
        if self.clip_to_unit_interval:
            score = float(np.clip(score, 0.0, 1.0))
        return _available(self.kind, "validation_support_novelty", score, mode=self.mode)
