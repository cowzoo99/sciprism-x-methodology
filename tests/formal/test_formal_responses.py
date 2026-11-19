import json

from sciprism_x.formal.responses import (
    acquire_formal_responses,
    load_pooled_profile_targets,
    response_seed,
    resolve_generator_profiles,
)
from sciprism_x.responses.reconstruct import fixture_cases

from .helpers import run_config, smoke_contract


def _smoke_cases():
    return fixture_cases(smoke_contract().scicu_task_count)


def _pooled_targets_only():
    targets, _ = load_pooled_profile_targets(run_config())
    return targets


def test_smoke_response_batch_has_every_model_and_replicate() -> None:
    config = run_config()
    contract = smoke_contract()
    batch = acquire_formal_responses(_smoke_cases(), config, contract)

    assert len(batch.rows) == contract.scicu_response_count == 594
    assert {row.model_id for row in batch.rows} == set(config.generators.model_ids)
    assert {row.replicate for row in batch.rows} == {0, 1, 2}
    assert all(row.response_text for row in batch.rows)
    assert len(batch.reference_labels) == len(batch.rows)
    assert all(label.source == "reconstruction_audit" for label in batch.reference_labels)
    assert all(
        label.structural_validity == (not label.applied_transforms)
        for label in batch.reference_labels
    )
    assert all(
        label.endpoint_success == ("endpoint" not in label.applied_transforms)
        for label in batch.reference_labels
    )
    assert all(
        label.failure_group
        == ("valid" if not label.applied_transforms else "|".join(label.applied_transforms))
        for label in batch.reference_labels
    )


def test_pooled_profile_is_maximum_entropy_when_model_cells_are_absent() -> None:
    config = run_config()
    profiles = resolve_generator_profiles(
        config,
        _pooled_targets_only(),
        seed=42,
        entropy_weight=0.01,
    )

    assert len(profiles) == 6
    assert len({profile.parameters for profile in profiles}) == 1
    assert all(profile.route == "maximum_entropy_pooled" for profile in profiles)
    assert {profile.model_id for profile in profiles} == set(config.generators.model_ids)


def test_computed_response_rows_do_not_contain_published_scores() -> None:
    batch = acquire_formal_responses(_smoke_cases(), run_config(), smoke_contract())
    serialized = json.dumps([row.model_dump() for row in batch.rows])

    assert "paper_published_target" not in serialized
    assert "PCS" not in serialized
    assert "BSI" not in serialized


def test_boundary_strengths_share_response_randomness() -> None:
    prefix = "case-1::PERTURB::scicueval::boundary::evidence_removal"
    identifiers = tuple(
        f"{prefix}::{strength:.2f}::iid::00000"
        for strength in (0.25, 0.50, 1.00)
    )

    assert len({response_seed(42, value, "model", 0) for value in identifiers}) == 1
    assert response_seed(42, identifiers[0], "model", 0) != response_seed(
        42,
        identifiers[0].replace("evidence_removal", "modality_conflict"),
        "model",
        0,
    )
    assert response_seed(42, identifiers[0], "model", 0) != response_seed(
        42,
        identifiers[0].replace("::iid::", "::domain_transfer::"),
        "model",
        0,
    )
