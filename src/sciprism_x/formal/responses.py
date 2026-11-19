from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from sciprism_x.config import RunConfig, TrackIsolationError
from sciprism_x.data.schema import SciCUCase
from sciprism_x.reproducibility import derive_seed
from sciprism_x.responses.adapters import RecoveryRequest, recover_responses
from sciprism_x.responses.fit import (
    PROFILE_BOUNDS,
    PROFILE_NAMES,
    OptimizerControls,
    ProfileFitRequest,
    ProfileFitResult,
    ProfileGroup,
    fit_profiles,
)
from sciprism_x.responses.reconstruct import reconstruct_response_with_audit
from sciprism_x.responses.schema import (
    ReconstructionProfile,
    ResponseAcquisitionRecord,
    ResponseRecord,
)

from .schema import ScaleContract


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ResolvedGeneratorProfile:
    model_id: str
    parameters: tuple[float, ...]
    profile: ReconstructionProfile
    route: Literal["maximum_entropy_pooled", "explicit_track_a_neutral"]
    fit: ProfileFitResult


@dataclass(frozen=True, slots=True)
class FormalResponseBatch:
    rows: tuple[ResponseRecord, ...]
    profiles: tuple[ResolvedGeneratorProfile, ...]
    acquisitions: tuple[ResponseAcquisitionRecord, ...]
    target_derivation: Mapping[str, str]
    reference_labels: tuple["ResponseReferenceLabel", ...]


@dataclass(frozen=True, slots=True)
class ResponseReferenceLabel:
    task_id: str
    model_id: str
    replicate: int
    endpoint_success: bool
    structural_validity: bool
    failure_group: str
    applied_transforms: tuple[str, ...]
    source: Literal["reconstruction_audit", "recovered_exact_match"]


def reference_label_from_audit(
    row: ResponseRecord,
    applied_transforms: Sequence[str],
) -> ResponseReferenceLabel:
    transforms = tuple(dict.fromkeys(str(value) for value in applied_transforms))
    return ResponseReferenceLabel(
        task_id=row.task_id,
        model_id=row.model_id,
        replicate=row.replicate,
        endpoint_success="endpoint" not in transforms,
        structural_validity=not transforms,
        failure_group="valid" if not transforms else "|".join(transforms),
        applied_transforms=transforms,
        source="reconstruction_audit",
    )


def _published_target_path(config: RunConfig) -> Path:
    if config.published_targets is None:
        raise TrackIsolationError("Track B fitted-profile routing requires published targets")
    configured = Path(config.published_targets)
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    if not path.is_file():
        raise FileNotFoundError(f"published target manifest is unavailable: {path}")
    return path.resolve()


def load_pooled_profile_targets(
    config: RunConfig,
) -> tuple[dict[str, float], dict[str, str]]:
    payload = json.loads(_published_target_path(config).read_text(encoding="utf-8"))
    worked = payload["worked_case"]
    residuals = worked["residuals"]
    secondary_gap = float(worked["secondary_diagnostic_gap"]["value"])
    targets = {
        "physical_error": float(residuals["physical"]),
        "logical_error": float(residuals["logical"]),
        "unit_error": float(residuals["unit_dimensional"]),
        "causal_error": float(residuals["causal"]),
        "modal_evidence_error": float(residuals["modal_evidential"]),
        "uncertainty_error": float(residuals["uncertainty"]),
        "endpoint_error": 1.0 - float(worked["endpoint_score"]),
        "motif_omission": 1.0 - float(worked["pcs"]),
        "unsupported_content": float(worked["bsi"]),
        "length_ratio": 1.0,
        "evidence_synthesis": 1.0 - secondary_gap,
    }
    derivation = {
        "six_channel_error_rates": "worked_case.residuals",
        "endpoint_error": "1 - worked_case.endpoint_score",
        "motif_omission": "1 - worked_case.pcs",
        "unsupported_content": "worked_case.bsi",
        "length_ratio": "neutral 1.0 because response length is unreported",
        "evidence_synthesis": "1 - worked_case.secondary_diagnostic_gap.value",
    }
    return targets, derivation


def load_track_a_profile_targets(
    config: RunConfig,
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve the explicit neutral profile without reading paper targets."""

    targets = {
        name: float(value)
        for name, value in config.response_reconstruction.fixture_profile.model_dump(
            mode="python"
        ).items()
    }
    derivation = {
        name: (
            f"response_reconstruction.fixture_profile.{name}; Track A neutral "
            "counterfactual profile for unobserved original boundary responses"
        )
        for name in targets
    }
    return targets, derivation


def resolve_generator_profiles(
    config: RunConfig,
    pooled_targets: Mapping[str, float],
    *,
    seed: int,
    entropy_weight: float = 0.01,
) -> tuple[ResolvedGeneratorProfile, ...]:
    unknown = set(pooled_targets).difference(PROFILE_NAMES)
    if unknown:
        raise ValueError(f"unknown pooled profile targets: {sorted(unknown)}")
    request = ProfileFitRequest(
        group=ProfileGroup(
            model_id="pooled_six_generators",
            regime="pooled",
            domain="pooled",
            competency="pooled",
        ),
        targets={name: float(value) for name, value in pooled_targets.items()},
        weights={name: 1.0 for name in pooled_targets},
    )
    fit = fit_profiles(
        (request,),
        seed=seed,
        entropy_weight=entropy_weight,
    )[0]
    parameters = tuple(float(getattr(fit.profile, name)) for name in PROFILE_NAMES)
    return tuple(
        ResolvedGeneratorProfile(
            model_id=model_id,
            parameters=parameters,
            profile=fit.profile,
            route="maximum_entropy_pooled",
            fit=fit,
        )
        for model_id in config.generators.model_ids
    )


def resolve_track_a_profiles(
    config: RunConfig,
    *,
    seed: int,
) -> tuple[ResolvedGeneratorProfile, ...]:
    profile = ReconstructionProfile.model_validate(
        config.response_reconstruction.fixture_profile.model_dump(mode="python")
    )
    target_vector = {
        name: float(value) for name, value in profile.model_dump(mode="python").items()
    }
    fit = ProfileFitResult(
        group=ProfileGroup(
            model_id="track_a_neutral_counterfactual",
            regime="boundary_original",
            domain="pooled",
            competency="pooled",
        ),
        profile=profile,
        target_vector=target_vector,
        weights={name: 1.0 for name in target_vector},
        bounds=PROFILE_BOUNDS,
        convergence_trace=(0.0,),
        loss=0.0,
        iterations=0,
        evaluations=0,
        seed=seed,
        entropy_weight=0.0,
        optimizer=OptimizerControls(),
    )
    parameters = tuple(float(getattr(profile, name)) for name in PROFILE_NAMES)
    return tuple(
        ResolvedGeneratorProfile(
            model_id=model_id,
            parameters=parameters,
            profile=profile,
            route="explicit_track_a_neutral",
            fit=fit,
        )
        for model_id in config.generators.model_ids
    )


def response_seed(paper_seed: int, task_id: str, model_id: str, replicate: int) -> int:
    paired_task_id = re.sub(
        r"(::PERTURB::scicueval::boundary::[^:]+)::(?:0\.25|0\.50|1\.00)((?:::[^:]+)?::\d+)$",
        r"\1::paired\2",
        task_id,
    )
    return derive_seed(
        paper_seed,
        f"formal-response::{paired_task_id}::{model_id}::{replicate}",
    )


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reconstructed_rows(
    cases: Sequence[SciCUCase],
    resolved: ResolvedGeneratorProfile,
    *,
    responses_per_task: int,
    paper_seed: int,
    temperature: float,
    max_output_tokens: int,
    evaluation_period: tuple[str, str],
) -> tuple[
    tuple[ResponseRecord, ...],
    tuple[ResponseReferenceLabel, ...],
    ResponseAcquisitionRecord,
]:
    reconstructed = tuple(
        reconstruct_response_with_audit(
            case,
            resolved.profile,
            seed=response_seed(paper_seed, case.task_id, resolved.model_id, replicate),
            model_id=resolved.model_id,
            replicate=replicate,
        )
        for case in cases
        for replicate in range(responses_per_task)
    )
    rows = tuple(result.record for result in reconstructed)
    labels = tuple(
        reference_label_from_audit(result.record, result.audit.applied_transforms)
        for result in reconstructed
    )
    profile_payload = resolved.profile.model_dump(mode="json")
    acquisition = ResponseAcquisitionRecord(
        route="deterministic_reconstruction",
        requested_model_id=resolved.model_id,
        resolved_model_id=resolved.model_id,
        endpoint=None,
        source_sha256=_canonical_hash(
            [row.model_dump(mode="json") for row in rows]
        ),
        fit_sha256=_canonical_hash(profile_payload),
        settings={
            "base_seed": paper_seed,
            "responses_per_task": responses_per_task,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "evaluation_period": evaluation_period,
            "profile": profile_payload,
            "profile_route": resolved.route,
        },
    )
    return rows, labels, acquisition


def _recovered_reference_labels(
    cases: Sequence[SciCUCase],
    rows: Sequence[ResponseRecord],
) -> tuple[ResponseReferenceLabel, ...]:
    references = {case.task_id: case.reference_answer.strip() for case in cases}
    return tuple(
        ResponseReferenceLabel(
            task_id=row.task_id,
            model_id=row.model_id,
            replicate=row.replicate,
            endpoint_success=row.response_text.strip() == references[row.task_id],
            structural_validity=row.response_text.strip() == references[row.task_id],
            failure_group=(
                "valid"
                if row.response_text.strip() == references[row.task_id]
                else "unadjudicated_response"
            ),
            applied_transforms=(),
            source="recovered_exact_match",
        )
        for row in rows
    )


def acquire_formal_responses(
    cases: Sequence[SciCUCase],
    config: RunConfig,
    contract: ScaleContract,
    *,
    entropy_weight: float = 0.01,
    recovery_requests: Mapping[str, RecoveryRequest] | None = None,
) -> FormalResponseBatch:
    case_rows = tuple(cases)
    if len(case_rows) != contract.scicu_task_count:
        raise ValueError(
            f"formal case count mismatch: expected {contract.scicu_task_count}, got {len(case_rows)}"
        )
    if len(config.generators.model_ids) != contract.generator_count:
        raise ValueError("formal generator count does not match the scale contract")
    if config.generators.responses_per_task != contract.responses_per_task:
        raise ValueError("formal replicate count does not match the scale contract")

    supplied_requests = recovery_requests or {}
    if config.track == "A":
        missing_models = set(config.generators.model_ids).difference(supplied_requests)
        if missing_models:
            raise TrackIsolationError(
                "Track A requires a recovery request for every generator; missing: "
                f"{sorted(missing_models)}"
            )
        targets, derivation = load_track_a_profile_targets(config)
        profiles = resolve_track_a_profiles(config, seed=config.paper.seeds[0])
    else:
        targets, derivation = load_pooled_profile_targets(config)
        profiles = resolve_generator_profiles(
            config,
            targets,
            seed=config.paper.seeds[0],
            entropy_weight=entropy_weight,
        )
    rows: list[ResponseRecord] = []
    reference_labels: list[ResponseReferenceLabel] = []
    acquisitions: list[ResponseAcquisitionRecord] = []
    for resolved in profiles:
        if resolved.model_id in supplied_requests:
            request = supplied_requests[resolved.model_id]
            if (
                request.cases != case_rows
                or request.model_id != resolved.model_id
                or request.responses_per_task != contract.responses_per_task
            ):
                raise ValueError("formal recovery request does not match the execution contract")
            recovered = recover_responses(request)
            model_rows = recovered.rows
            model_labels = _recovered_reference_labels(case_rows, model_rows)
            acquisition = recovered.acquisition.model_copy(
                update={
                    "settings": dict(recovered.acquisition.settings)
                    | {
                        "responses_per_task": config.generators.responses_per_task,
                        "temperature": config.generators.temperature,
                        "max_output_tokens": config.generators.max_output_tokens,
                        "evaluation_period": config.generators.evaluation_period,
                    }
                }
            )
        else:
            model_rows, model_labels, acquisition = _reconstructed_rows(
                case_rows,
                resolved,
                responses_per_task=contract.responses_per_task,
                paper_seed=config.paper.seeds[0],
                temperature=config.generators.temperature,
                max_output_tokens=config.generators.max_output_tokens,
                evaluation_period=config.generators.evaluation_period,
            )
        rows.extend(model_rows)
        reference_labels.extend(model_labels)
        acquisitions.append(acquisition)

    result_rows = tuple(rows)
    if len(result_rows) != contract.scicu_response_count:
        raise ValueError("formal response acquisition produced an incomplete matrix")
    return FormalResponseBatch(
        rows=result_rows,
        profiles=profiles,
        acquisitions=tuple(acquisitions),
        target_derivation=derivation,
        reference_labels=tuple(reference_labels),
    )
