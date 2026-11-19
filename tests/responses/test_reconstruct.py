import json


def _case(answer: str):
    from sciprism_x.data.schema import SciCUCase

    return SciCUCase(
        task_id="SCICU-0001",
        prompt="Calculate the speed and explain the condition.",
        reference_context="Table 1 gives the measured field values.",
        reference_answer=answer,
        domain="physics",
        competency="multi-source integration",
        motif_family="unit/dimensional consistency",
        source_hash="a" * 64,
    )


def _profile(**updates):
    from sciprism_x.responses.schema import ReconstructionProfile

    values = {
        "physical_error": 0.0,
        "logical_error": 0.0,
        "unit_error": 0.0,
        "causal_error": 0.0,
        "modal_evidence_error": 0.0,
        "uncertainty_error": 0.0,
        "endpoint_error": 0.0,
        "motif_omission": 0.0,
        "unsupported_content": 0.0,
        "length_ratio": 1.0,
        "evidence_synthesis": 1.0,
    }
    values.update(updates)
    return ReconstructionProfile(**values)


def test_unit_mutation_is_deterministic_and_observable() -> None:
    from sciprism_x.responses.reconstruct import reconstruct_response

    case = _case("Under ideal conditions, the result is 12 m/s.")
    profile = _profile(unit_error=1.0)

    first = reconstruct_response(case, profile, seed=42)
    second = reconstruct_response(case, profile, seed=42)

    assert first == second
    assert "12 m/s" not in first.response_text
    assert set(first.model_dump()) == {
        "task_id",
        "model_id",
        "replicate",
        "response_text",
    }


def test_all_structured_transformations_have_auditable_effects() -> None:
    from sciprism_x.responses.reconstruct import reconstruct_response_with_audit

    case = _case(
        "According to Table 1, under ideal conditions the force balances because "
        "the fields oppose, so the result is approximately 12 m/s."
    )
    result = reconstruct_response_with_audit(
        case,
        _profile(
            physical_error=1.0,
            logical_error=1.0,
            unit_error=1.0,
            causal_error=1.0,
            modal_evidence_error=1.0,
            uncertainty_error=1.0,
            endpoint_error=1.0,
            motif_omission=1.0,
            unsupported_content=1.0,
            length_ratio=1.5,
            evidence_synthesis=0.0,
        ),
        seed=42,
        model_id="gpt-4o-2024-08-06",
        replicate=2,
    )

    assert result.record.model_id == "gpt-4o-2024-08-06"
    assert result.record.replicate == 2
    assert set(result.audit.applied_transforms) == {
        "physical",
        "logical",
        "unit",
        "causal",
        "modal_evidence",
        "uncertainty",
        "endpoint",
        "motif_omission",
        "unsupported_content",
        "evidence_synthesis",
        "length_ratio",
    }
    assert result.record.response_text


def test_recovery_routes_are_ordered_and_manifest_only(tmp_path) -> None:
    from sciprism_x.responses.adapters import RecoveryRequest, recover_responses

    supplied = tmp_path / "responses.jsonl"
    supplied.write_text(
        json.dumps(
            {
                "task_id": "SCICU-0001",
                "model_id": "gpt-4o-2024-08-06",
                "replicate": 0,
                "response_text": "Supplied response.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = recover_responses(
        RecoveryRequest(
            cases=(_case("Reference answer."),),
            model_id="gpt-4o-2024-08-06",
            responses_per_task=1,
            seed=42,
            reconstruction_profile=_profile(),
            supplied_file=supplied,
            exact_model_loader=lambda case, replicate: "Exact response.",
            official_call_authorized=True,
            exact_endpoint="official:gpt-4o-2024-08-06",
        )
    )

    assert result.rows[0].response_text == "Supplied response."
    assert result.acquisition.route == "supplied_file"
    assert set(result.rows[0].model_dump()) == {
        "task_id",
        "model_id",
        "replicate",
        "response_text",
    }


def test_recovery_falls_back_to_deterministic_reconstruction() -> None:
    from sciprism_x.responses.adapters import RecoveryRequest, recover_responses

    request = RecoveryRequest(
        cases=(_case("The result is 12 m/s."),),
        model_id="gpt-4o-2024-08-06",
        responses_per_task=2,
        seed=42,
        reconstruction_profile=_profile(unit_error=1.0),
    )

    first = recover_responses(request)
    second = recover_responses(request)

    assert first == second
    assert len(first.rows) == 2
    assert first.acquisition.route == "deterministic_reconstruction"
    assert all(row.model_id == request.model_id for row in first.rows)


def test_exact_model_precedes_nearest_model() -> None:
    from sciprism_x.responses.adapters import RecoveryRequest, recover_responses

    result = recover_responses(
        RecoveryRequest(
            cases=(_case("Reference answer."),),
            model_id="gpt-4o-2024-08-06",
            responses_per_task=1,
            seed=42,
            reconstruction_profile=_profile(),
            exact_model_loader=lambda case, replicate: "Exact response.",
            exact_endpoint="official:gpt-4o-2024-08-06",
            nearest_model_id="gpt-4o-nearest",
            nearest_model_loader=lambda case, replicate: "Nearest response.",
            nearest_endpoint="official:gpt-4o-nearest",
            official_call_authorized=True,
        )
    )

    assert result.rows[0].response_text == "Exact response."
    assert result.acquisition.route == "exact_official_model"
    assert result.acquisition.resolved_model_id == "gpt-4o-2024-08-06"


def test_nearest_model_precedes_reconstruction() -> None:
    from sciprism_x.responses.adapters import RecoveryRequest, recover_responses

    result = recover_responses(
        RecoveryRequest(
            cases=(_case("Reference answer."),),
            model_id="Mistral-Large-2411",
            responses_per_task=1,
            seed=42,
            reconstruction_profile=_profile(),
            nearest_model_id="mistral-large-nearest",
            nearest_model_loader=lambda case, replicate: "Nearest response.",
            nearest_endpoint="official:mistral-large-nearest",
            official_call_authorized=True,
        )
    )

    assert result.rows[0].response_text == "Nearest response."
    assert result.acquisition.route == "nearest_official_model"
    assert result.acquisition.resolved_model_id == "mistral-large-nearest"


def test_model_calls_require_explicit_authorization() -> None:
    from sciprism_x.responses.adapters import RecoveryRequest, recover_responses

    result = recover_responses(
        RecoveryRequest(
            cases=(_case("Reference answer."),),
            model_id="gpt-4o-2024-08-06",
            responses_per_task=1,
            seed=42,
            reconstruction_profile=_profile(),
            exact_model_loader=lambda case, replicate: "Exact response.",
            exact_endpoint="official:gpt-4o-2024-08-06",
            official_call_authorized=False,
        )
    )

    assert result.acquisition.route == "deterministic_reconstruction"
