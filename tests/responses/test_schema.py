import pytest
from pydantic import ValidationError


def test_response_rows_use_paper_compatible_schema() -> None:
    from sciprism_x.responses.schema import ResponseRecord

    row = ResponseRecord(
        task_id="SCICU-0001",
        model_id="gpt-4o-2024-08-06",
        replicate=0,
        response_text="The pressure doubles because P is proportional to T.",
    )

    assert set(row.model_dump()) == {
        "task_id",
        "model_id",
        "replicate",
        "response_text",
    }


def test_response_rows_reject_internal_acquisition_fields() -> None:
    from sciprism_x.responses.schema import ResponseRecord

    with pytest.raises(ValidationError):
        ResponseRecord(
            task_id="SCICU-0001",
            model_id="gpt-4o-2024-08-06",
            replicate=0,
            response_text="Answer.",
            route="deterministic_reconstruction",
        )


def test_acquisition_record_is_a_separate_manifest_contract() -> None:
    from sciprism_x.responses.schema import ResponseAcquisitionRecord

    record = ResponseAcquisitionRecord(
        route="supplied_file",
        requested_model_id="gpt-4o-2024-08-06",
        resolved_model_id="gpt-4o-2024-08-06",
        endpoint=None,
        source_sha256="a" * 64,
        fit_sha256=None,
        settings={"path": "inputs/responses.jsonl"},
    )

    assert record.route == "supplied_file"
    assert "response_text" not in record.model_dump()
