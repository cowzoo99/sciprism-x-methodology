from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Protocol

import pint
import sympy

from .schema import ValidationPair, ValidationResult, pair_evidence


class Validator(Protocol):
    def validate(self, pair: ValidationPair) -> ValidationResult: ...


def _feature(node, name: str) -> tuple[str, ...]:
    return tuple(node.feature_evidence.get(name, ()))


def _equation_difference(equation: str):
    if "=" not in equation:
        raise ValueError("equation has no equality sign")
    left, right = equation.split("=", 1)
    names = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", equation))
    symbols = {name: sympy.Symbol(name) for name in names}
    return sympy.sympify(left.replace("^", "**"), locals=symbols) - sympy.sympify(
        right.replace("^", "**"), locals=symbols
    )


def _symbolically_equivalent(task_equation: str, response_equation: str) -> bool:
    task = _equation_difference(task_equation)
    response = _equation_difference(response_equation)
    return bool(
        sympy.simplify(task - response) == 0
        or sympy.simplify(task + response) == 0
    )


class SymbolicEquivalenceValidator:
    validator_id = "deterministic_symbolic_equivalence"

    def __init__(self, *, timeout_seconds: float) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def validate(self, pair: ValidationPair) -> ValidationResult:
        task_equations = _feature(pair.task_node, "equations")
        response_equations = _feature(pair.response_node, "equations")
        if not task_equations or not response_equations:
            return ValidationResult.unresolved(
                "symbolic_cues_incomplete",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            _symbolically_equivalent,
            task_equations[0],
            response_equations[0],
        )
        try:
            equivalent = future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            future.cancel()
            return ValidationResult.unresolved(
                "symbolic_timeout",
                confidence=0.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        except (TypeError, ValueError, sympy.SympifyError):
            return ValidationResult.unresolved(
                "symbolic_parse_failed",
                confidence=0.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        factory = ValidationResult.pass_ if equivalent else ValidationResult.fail
        return factory(
            "symbolically_equivalent" if equivalent else "symbolic_mismatch",
            confidence=1.0,
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )


class DimensionalCompatibilityValidator:
    validator_id = "deterministic_dimensional_compatibility"

    def __init__(self) -> None:
        self.registry = pint.UnitRegistry(
            autoconvert_offset_to_baseunit=False,
            case_sensitive=True,
        )

    def validate(self, pair: ValidationPair) -> ValidationResult:
        task_quantities = _feature(pair.task_node, "units")
        response_quantities = _feature(pair.response_node, "units")
        if not task_quantities or not response_quantities:
            return ValidationResult.unresolved(
                "unit_cues_incomplete",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        try:
            task_dimension = self.registry.parse_expression(task_quantities[0]).dimensionality
            response_dimension = self.registry.parse_expression(
                response_quantities[0]
            ).dimensionality
        except (pint.errors.PintError, TypeError, ValueError):
            return ValidationResult.unresolved(
                "unit_parse_failed",
                confidence=0.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        compatible = task_dimension == response_dimension
        factory = ValidationResult.pass_ if compatible else ValidationResult.fail
        return factory(
            "dimensions_compatible" if compatible else "unit_dimension_mismatch",
            confidence=1.0,
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )


_CAUSES = re.compile(
    r"^\s*(?P<cause>.+?)\s+causes?\s+(?P<effect>.+?)[.!?]?\s*$",
    re.IGNORECASE,
)
_BECAUSE = re.compile(
    r"^\s*(?P<effect>.+?)\s+because\s+(?P<cause>.+?)[.!?]?\s*$",
    re.IGNORECASE,
)


def _causal_pair(text: str) -> tuple[str, str] | None:
    match = _CAUSES.match(text) or _BECAUSE.match(text)
    if match is None:
        return None

    def normalize(value: str) -> str:
        value = re.sub(r"\b(?:the|a|an)\b", " ", value.lower())
        return " ".join(re.findall(r"[a-z0-9]+", value))

    return normalize(match.group("cause")), normalize(match.group("effect"))


class CausalDirectionValidator:
    validator_id = "deterministic_causal_direction"

    def validate(self, pair: ValidationPair) -> ValidationResult:
        task = _causal_pair(pair.task_node.text)
        response = _causal_pair(pair.response_node.text)
        if task is None or response is None:
            return ValidationResult.unresolved(
                "causal_cues_implicit",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        if task == response:
            return ValidationResult.pass_(
                "causal_direction_matches",
                confidence=1.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        if task == (response[1], response[0]):
            return ValidationResult.fail(
                "causal_direction_reversed",
                confidence=1.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        return ValidationResult.fail(
            "causal_relation_mismatch",
            confidence=0.9,
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )


_ANCHOR = re.compile(
    r"\b(?P<kind>table|figure|fig\.?|graph|image|record|passage|reference|source)"
    r"(?:\s+(?P<identifier>[A-Za-z0-9.-]+))?",
    re.IGNORECASE,
)


def _anchors(text: str) -> tuple[tuple[str, str], ...]:
    anchors: list[tuple[str, str]] = []
    for match in _ANCHOR.finditer(text):
        kind = match.group("kind").lower().rstrip(".")
        if kind == "fig":
            kind = "figure"
        identifier = (match.group("identifier") or "").lower().rstrip(".,;:)]}")
        anchors.append((kind, identifier))
    return tuple(anchors)


class EvidenceBindingValidator:
    validator_id = "deterministic_evidence_binding"

    def validate(self, pair: ValidationPair) -> ValidationResult:
        required = _anchors(pair.task_node.text)
        observed = _anchors(pair.response_node.text)
        if not required:
            return ValidationResult.unresolved(
                "evidence_anchor_not_explicit",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        if set(required).intersection(observed):
            return ValidationResult.pass_(
                "evidence_anchor_bound",
                confidence=1.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        return ValidationResult.fail(
            "required_evidence_anchor_missing",
            confidence=1.0,
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )


class ModalityCompatibilityValidator:
    validator_id = "deterministic_modality_compatibility"

    def validate(self, pair: ValidationPair) -> ValidationResult:
        required = {kind for kind, identifier in _anchors(pair.task_node.text)}
        observed = {kind for kind, identifier in _anchors(pair.response_node.text)}
        if not required:
            return ValidationResult.unresolved(
                "modality_not_explicit",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        compatible = bool(required.intersection(observed))
        factory = ValidationResult.pass_ if compatible else ValidationResult.fail
        return factory(
            "modality_matches" if compatible else "modality_anchor_mismatch",
            confidence=1.0,
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )


_UNCERTAINTY = re.compile(
    r"\b(?:approximately|about|may|might|could|assuming|assume|ideal|"
    r"uncertain|uncertainty|confidence|caveat|limitation)\b",
    re.IGNORECASE,
)


class UncertaintyQualifierValidator:
    validator_id = "deterministic_uncertainty_qualification"

    def validate(self, pair: ValidationPair) -> ValidationResult:
        required = bool(_UNCERTAINTY.search(pair.task_node.text))
        observed = bool(_UNCERTAINTY.search(pair.response_node.text))
        if not required:
            return ValidationResult.unresolved(
                "uncertainty_not_required_explicitly",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        factory = ValidationResult.pass_ if observed else ValidationResult.fail
        return factory(
            "uncertainty_qualified" if observed else "uncertainty_qualifier_missing",
            confidence=1.0,
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )


_UNSUPPORTED = re.compile(
    r"\b(?:without (?:evidence|support|premises)|unsupported|regardless of|"
    r"exact and unconditional)\b",
    re.IGNORECASE,
)


class UnsupportedClaimValidator:
    validator_id = "deterministic_unsupported_claim"

    def validate(self, pair: ValidationPair) -> ValidationResult:
        if _UNSUPPORTED.search(pair.response_node.text):
            return ValidationResult.fail(
                "unsupported_claim_explicit",
                confidence=1.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        return ValidationResult.unresolved(
            "unsupported_claim_not_rule_checkable",
            confidence=0.0,
            validator_id=self.validator_id,
        )


class BoundaryConditionValidator:
    validator_id = "deterministic_boundary_condition"
    _CONDITION = re.compile(r"\b(?:under|assuming)\s+([^,.;]+)", re.IGNORECASE)

    def validate(self, pair: ValidationPair) -> ValidationResult:
        required = self._CONDITION.search(pair.task_node.text)
        if required is None:
            return ValidationResult.unresolved(
                "boundary_condition_not_explicit",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        if re.search(r"\bregardless of\b", pair.response_node.text, re.IGNORECASE):
            return ValidationResult.fail(
                "boundary_condition_violated",
                confidence=1.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        required_text = required.group(1).lower().strip()
        if required_text in pair.response_node.text.lower():
            return ValidationResult.pass_(
                "boundary_condition_preserved",
                confidence=1.0,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        return ValidationResult.unresolved(
            "boundary_condition_implicit",
            confidence=0.0,
            validator_id=self.validator_id,
        )


def default_deterministic_validators(
    *,
    symbolic_timeout_seconds: float,
) -> tuple[Validator, ...]:
    return (
        SymbolicEquivalenceValidator(timeout_seconds=symbolic_timeout_seconds),
        DimensionalCompatibilityValidator(),
        CausalDirectionValidator(),
        EvidenceBindingValidator(),
        ModalityCompatibilityValidator(),
        UncertaintyQualifierValidator(),
        UnsupportedClaimValidator(),
        BoundaryConditionValidator(),
    )
