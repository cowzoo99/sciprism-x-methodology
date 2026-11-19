from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .schema import ValidationPair, ValidationResult, ValidationStatus


class Validator(Protocol):
    def validate(self, pair: ValidationPair) -> ValidationResult: ...


class ValidatorRegistry:
    def __init__(
        self,
        *,
        deterministic: Sequence[Validator],
        learned: Sequence[Validator],
    ) -> None:
        self.deterministic = tuple(deterministic)
        self.learned = tuple(learned)

    def validate(self, pair: ValidationPair) -> ValidationResult:
        deterministic_passes: list[ValidationResult] = []
        deterministic_unavailable: list[ValidationResult] = []
        for validator in self.deterministic:
            result = validator.validate(pair)
            if result.status is ValidationStatus.FAIL:
                return result
            if result.status is ValidationStatus.PASS:
                deterministic_passes.append(result)
            elif result.status is ValidationStatus.UNAVAILABLE_EXTERNAL:
                deterministic_unavailable.append(result)
        if deterministic_passes:
            return max(deterministic_passes, key=lambda item: item.confidence)

        learned_unavailable: list[ValidationResult] = []
        for validator in self.learned:
            result = validator.validate(pair)
            if result.status in {ValidationStatus.PASS, ValidationStatus.FAIL}:
                return result
            if result.status is ValidationStatus.UNAVAILABLE_EXTERNAL:
                learned_unavailable.append(result)
        unavailable = learned_unavailable or deterministic_unavailable
        if unavailable:
            return unavailable[0]
        return ValidationResult.unresolved(
            "all_validators_unresolved",
            confidence=0.0,
            validator_id="validator_registry",
        )


def validate_pair(
    pair: ValidationPair,
    registry: ValidatorRegistry,
) -> ValidationResult:
    return registry.validate(pair)
