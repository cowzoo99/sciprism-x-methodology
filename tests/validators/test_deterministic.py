from sciprism_x.motifs.schema import MotifFamily, MotifNode


def _node(node_id: str, text: str, family: MotifFamily, **evidence) -> MotifNode:
    return MotifNode(
        node_id=node_id,
        text=text,
        motif_family=family,
        confidence=1.0,
        salience=1.0,
        source_span=(0, len(text)),
        variables=tuple(evidence.pop("variables", ())),
        feature_evidence={key: tuple(value) for key, value in evidence.items()},
    )


def _pair(task: MotifNode, response: MotifNode, context: str = ""):
    from sciprism_x.validators.schema import ValidationPair

    return ValidationPair(
        task_node=task,
        response_node=response,
        reference_context=context,
    )


def test_symbolic_equivalence_passes_and_mismatch_fails() -> None:
    from sciprism_x.validators.deterministic import SymbolicEquivalenceValidator
    from sciprism_x.validators.schema import ValidationStatus

    validator = SymbolicEquivalenceValidator(timeout_seconds=1.0)
    task = _node(
        "T1",
        "v = E/B",
        MotifFamily.SYMBOLIC_TRANSFORMATION,
        equations=("v = E/B",),
    )
    same = _node(
        "R1",
        "v = E/B",
        MotifFamily.SYMBOLIC_TRANSFORMATION,
        equations=("v = E/B",),
    )
    wrong = _node(
        "R2",
        "v = E*B",
        MotifFamily.SYMBOLIC_TRANSFORMATION,
        equations=("v = E*B",),
    )

    assert validator.validate(_pair(task, same)).status is ValidationStatus.PASS
    assert validator.validate(_pair(task, wrong)).status is ValidationStatus.FAIL


def test_dimensional_compatibility_uses_explicit_quantities() -> None:
    from sciprism_x.validators.deterministic import DimensionalCompatibilityValidator
    from sciprism_x.validators.schema import ValidationStatus

    validator = DimensionalCompatibilityValidator()
    task = _node(
        "T1",
        "The result is 12 m/s.",
        MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY,
        units=("12 m/s",),
    )
    compatible = _node(
        "R1",
        "The result is 43.2 km/h.",
        MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY,
        units=("43.2 km/h",),
    )
    incompatible = _node(
        "R2",
        "The result is 12 kg.",
        MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY,
        units=("12 kg",),
    )

    assert validator.validate(_pair(task, compatible)).status is ValidationStatus.PASS
    assert validator.validate(_pair(task, incompatible)).status is ValidationStatus.FAIL


def test_causal_reversal_is_a_deterministic_failure() -> None:
    from sciprism_x.validators.deterministic import CausalDirectionValidator
    from sciprism_x.validators.schema import ValidationStatus

    task = _node(
        "T1",
        "Temperature causes pressure.",
        MotifFamily.CAUSAL_MECHANISTIC_RELATION,
    )
    response = _node(
        "R1",
        "Pressure causes temperature.",
        MotifFamily.CAUSAL_MECHANISTIC_RELATION,
    )

    result = CausalDirectionValidator().validate(_pair(task, response))

    assert result.status is ValidationStatus.FAIL
    assert result.reason_code == "causal_direction_reversed"


def test_required_evidence_and_modality_anchor_cannot_be_dropped() -> None:
    from sciprism_x.validators.deterministic import (
        EvidenceBindingValidator,
        ModalityCompatibilityValidator,
    )
    from sciprism_x.validators.schema import ValidationStatus

    task = _node(
        "T1",
        "According to Table 1, the value rises.",
        MotifFamily.MODALITY_GROUNDING,
    )
    response = _node(
        "R1",
        "The value rises without a cited source.",
        MotifFamily.MODALITY_GROUNDING,
    )
    pair = _pair(task, response, context="Table 1 contains the measurements.")

    assert EvidenceBindingValidator().validate(pair).status is ValidationStatus.FAIL
    assert ModalityCompatibilityValidator().validate(pair).status is ValidationStatus.FAIL


def test_evidence_binding_ignores_sentence_punctuation_after_identifier() -> None:
    from sciprism_x.validators.deterministic import EvidenceBindingValidator
    from sciprism_x.validators.schema import ValidationStatus

    pair = _pair(
        _node(
            "T1",
            "Use Table 1 to derive the result.",
            MotifFamily.EVIDENCE_BINDING,
        ),
        _node(
            "R1",
            "The result follows from Table 1.",
            MotifFamily.EVIDENCE_BINDING,
        ),
    )

    assert EvidenceBindingValidator().validate(pair).status is ValidationStatus.PASS


def test_missing_uncertainty_and_unsupported_claim_are_failures() -> None:
    from sciprism_x.validators.deterministic import (
        UncertaintyQualifierValidator,
        UnsupportedClaimValidator,
    )
    from sciprism_x.validators.schema import ValidationStatus

    uncertainty_pair = _pair(
        _node(
            "T1",
            "Under ideal conditions, the value is approximately 12 m/s.",
            MotifFamily.UNCERTAINTY_QUALIFICATION,
        ),
        _node(
            "R1",
            "The value is exactly 12 m/s in every condition.",
            MotifFamily.UNCERTAINTY_QUALIFICATION,
        ),
    )
    unsupported_pair = _pair(
        _node("T2", "Use the stated evidence.", MotifFamily.EVIDENCE_BINDING),
        _node(
            "R2",
            "The conclusion follows without evidence.",
            MotifFamily.EVIDENCE_BINDING,
        ),
    )

    assert (
        UncertaintyQualifierValidator().validate(uncertainty_pair).status
        is ValidationStatus.FAIL
    )
    assert UnsupportedClaimValidator().validate(unsupported_pair).status is ValidationStatus.FAIL
