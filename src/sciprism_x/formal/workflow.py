from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Literal
import uuid

import numpy as np
import pandas as pd
import torch

from sciprism_x.baselines.schema import BaselineKind, EvaluatedCase, GraphEdge, GraphView
from sciprism_x.baselines.llm_judge import (
    LLMJudgeRequest,
    PAPER_JUDGE_FIELDS,
    Requester,
    validate_paper_judgment,
)
from sciprism_x.config import RunConfig, TrackIsolationError, load_run_config
from sciprism_x.data.schema import SciCUCase
from sciprism_x.responses.adapters import RecoveryRequest
from sciprism_x.responses.reconstruct import reconstruct_response_with_audit
from sciprism_x.responses.schema import ReconstructionProfile, ResponseRecord
from sciprism_x.residuals.schema import ResidualChannel
from sciprism_x.results.schema import FormalRunReceipt
from sciprism_x.state_field.model import StateFieldModel
from sciprism_x.open_set.calibration import compute_sc
from sciprism_x.release.instance_outputs import serialize_evaluated_components
from sciprism_x.release.manifests import (
    build_effective_hyperparameters,
    build_learned_modules_manifest,
    build_sample_manifest_rows,
    build_solver_environment,
)
from sciprism_x.release.plotting import (
    load_plot_style,
    render_ablation_heatmap_svg,
    render_aggregate_metrics_svg,
    write_figure_manifest,
)
from sciprism_x.release.protocol import protocol_asset_text, protocol_manifest

from .ablations import (
    AblationControls,
    AblationKind,
    PerCaseMetric,
    resolve_ablation_policy,
    run_formal_ablations,
)
from .allocation import (
    AllocationRow,
    AllocationTable,
    FormalAllocations,
    build_formal_allocations,
    largest_remainder,
)
from .baselines import (
    build_formal_baseline_registry,
    run_formal_baselines,
    select_formal_baseline_parameters,
)
from .boundaries import BoundaryEvaluation, compute_boundary_rows
from .config import FormalExecutionConfig, load_formal_config, resolve_scale_contract
from .datasets import DBP15KDataset, FormalDatasets, load_formal_datasets
from .dbp import DBPCaseResult, build_dbp_baseline_cases, run_dbp_experiment
from .evaluator import (
    FormalEvaluationContext,
    FormalEvaluationRow,
    MethodParameters,
    build_formal_context,
    evaluate_scicu_response,
    evaluate_scicu_response_with_components,
)
from .method_selection import MethodSelectionExample, select_method_parameters
from .open_set import OpenSetBundle, run_open_set_analysis
from .perturbed_cases import (
    MaterializedDBPPerturbation,
    MaterializedSciCUPerturbation,
    materialize_dbp_perturbation,
    materialize_scicu_perturbation,
)
from .responses import (
    ResponseReferenceLabel,
    acquire_formal_responses,
    reference_label_from_audit,
    response_seed,
)
from .results import (
    FormalArtifactSpec,
    assemble_formal_results,
)
from .schema import ScaleContract
from .state_training import train_state_models
from .statistics import (
    ComputedPairRow,
    FormalStatisticsResult,
    SignificanceRow,
    ComputedComparisonRow,
    build_discrepancy_report,
    compute_formal_statistics,
)
from .stages import StageStore, canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_STAGE_ORDER = (
    "sources",
    "allocations",
    "perturbation_inputs",
    "responses",
    "method_selection",
    "evaluation",
    "state_training",
    "boundaries",
    "open_set",
    "dbp15k",
    "baselines",
    "ablations",
    "statistics",
    "paper_comparison",
    "plotting_inputs",
    "release_contract",
    "figures",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_response_file_manifest(
    path: Path | None,
    *,
    model_ids: tuple[str, ...],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, str]]:
    if path is None:
        return {}, {}, {}
    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("response manifest must contain a models mapping")
    unknown = set(models).difference(model_ids)
    if unknown:
        raise ValueError(f"response manifest contains unknown generators: {sorted(unknown)}")
    evaluation: dict[str, Path] = {}
    tuning: dict[str, Path] = {}
    hashes = {"manifest": _sha256(manifest)}
    for model_id, record in models.items():
        if not isinstance(record, dict):
            raise ValueError(f"response manifest entry must be a mapping: {model_id}")
        for field, destination in (
            ("evaluation_file", evaluation),
            ("tuning_file", tuning),
        ):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"response manifest lacks {field}: {model_id}")
            candidate = Path(value)
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (manifest.parent / candidate).resolve()
            )
            if not resolved.is_file():
                raise FileNotFoundError(f"response file is unavailable: {resolved}")
            destination[model_id] = resolved
            hashes[f"{model_id}:{field}"] = _sha256(resolved)
    return evaluation, tuning, hashes


def _supplied_recovery_requests(
    files: Mapping[str, Path],
    cases: tuple[SciCUCase, ...],
    config: RunConfig,
) -> dict[str, RecoveryRequest]:
    return {
        model_id: RecoveryRequest(
            cases=cases,
            model_id=model_id,
            responses_per_task=config.generators.responses_per_task,
            seed=config.paper.seeds[0],
            reconstruction_profile=None,
            supplied_file=path,
        )
        for model_id, path in files.items()
    }


def _load_judge_response_replay(
    path: Path | None,
) -> tuple[Requester | None, str | None]:
    if path is None:
        return None, None
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows: dict[str, dict[str, object]] = {}
        for item in payload:
            if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
                raise ValueError("judge response rows must contain a string case_id")
            case_id = str(item["case_id"])
            if case_id in rows:
                raise ValueError(f"duplicate judge response case_id: {case_id}")
            rows[case_id] = {key: value for key, value in item.items() if key != "case_id"}
    elif isinstance(payload, dict):
        rows = {
            str(case_id): dict(value)
            for case_id, value in payload.items()
            if isinstance(value, dict)
        }
        if len(rows) != len(payload):
            raise ValueError("judge response mapping values must be objects")
    else:
        raise ValueError("judge response file must contain a list or case-id mapping")
    expected = set(PAPER_JUDGE_FIELDS)
    for case_id, row in rows.items():
        if set(row) != expected:
            raise ValueError(
                f"judge response fields differ from the method contract: {case_id}"
            )
        validate_paper_judgment(row)
    source_hash = _sha256(source)

    def replay(request: LLMJudgeRequest, api_key: str) -> Mapping[str, object]:
        if api_key != source_hash:
            raise ValueError("judge replay source hash does not match the frozen input")
        try:
            return rows[request.case_id]
        except KeyError as error:
            raise ValueError(
                f"judge response file lacks case_id: {request.case_id}"
            ) from error

    return replay, source_hash


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "src" / "sciprism_x").rglob("*.py")):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash(config: object) -> str:
    payload = config.model_dump(mode="json")  # type: ignore[attr-defined]
    return canonical_sha256(payload)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty formal table: {path.name}")
    pd.DataFrame(rows).to_parquet(
        path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


def _allocation_payload(table: AllocationTable) -> list[dict[str, object]]:
    return [asdict(row) for row in table.rows]


def _load_allocations(path: Path) -> FormalAllocations:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FormalAllocations(
        scicu=AllocationTable(tuple(AllocationRow(**row) for row in payload["scicueval"])),
        dbp=AllocationTable(tuple(AllocationRow(**row) for row in payload["dbp15k"])),
    )


def _case_hash(case: SciCUCase, allocation_id: str) -> str:
    return canonical_sha256(
        {"source_hash": case.source_hash, "allocation_id": allocation_id}
    )


def _ranked_identifiers(
    identifiers: tuple[str, ...],
    *,
    requested: str,
    seed: int,
    namespace: str,
) -> tuple[str, ...]:
    remaining = tuple(value for value in identifiers if value != requested)
    ranked = tuple(
        sorted(
            remaining,
            key=lambda value: (
                hashlib.sha256(
                    f"{seed}::{namespace}::{value}".encode("utf-8")
                ).digest(),
                value,
            ),
        )
    )
    return (requested,) + ranked if requested in identifiers else ranked


def _resolve_scicu_boundary_trajectories(
    datasets: FormalDatasets,
    allocations: FormalAllocations,
    config: RunConfig,
    *,
    seed: int,
) -> dict[
    str,
    tuple[
        AllocationRow,
        MaterializedSciCUPerturbation,
        str,
        tuple[str, ...],
    ],
]:
    """Freeze complete valid radius trajectories before response generation.

    Appendix C.3 excludes invalid perturbations before final evaluation.  A
    deterministic replacement is therefore selected from the same frozen base
    evaluation pool, and all three radii for a trajectory stay on one case.
    """

    cases = {case.task_id: case for case in datasets.scicueval}
    selected_by_regime: dict[str, list[str]] = {}
    for row in allocations.scicu.rows:
        if row.regime != "boundary_stress":
            selected_by_regime.setdefault(row.regime, []).append(row.case_id)
    pooled = tuple(
        dict.fromkeys(
            case_id
            for regime in ("iid", "motif_or_structure", "domain_transfer")
            for case_id in selected_by_regime.get(regime, ())
        )
    )
    pools = {
        regime: tuple(dict.fromkeys(case_ids))
        for regime, case_ids in selected_by_regime.items()
    } | {"boundary_stress": pooled, "pooled_smoke": pooled}

    groups: dict[tuple[str, str, str], list[AllocationRow]] = {}
    for row in allocations.scicu.rows:
        if row.regime != "boundary_stress":
            continue
        if row.bsi_regime is None or row.family is None or row.strength is None:
            raise ValueError("SciCUEval boundary allocation is incomplete")
        trajectory_index = row.allocation_id.rsplit("::", 1)[-1]
        groups.setdefault(
            (row.bsi_regime, row.family, trajectory_index), []
        ).append(row)

    expected_radii = tuple(float(value) for value in config.boundary.standardized_radii)
    used: dict[tuple[str, str], set[str]] = {}
    resolved: dict[
        str,
        tuple[
            AllocationRow,
            MaterializedSciCUPerturbation,
            str,
            tuple[str, ...],
        ],
    ] = {}
    for (bsi_regime, family, trajectory_index), trajectory in sorted(groups.items()):
        ordered_trajectory = tuple(sorted(trajectory, key=lambda row: float(row.strength)))
        if tuple(float(row.strength) for row in ordered_trajectory) != expected_radii:
            raise ValueError("SciCUEval boundary trajectory does not cover the frozen radii")
        requested = ordered_trajectory[0].case_id
        if any(row.case_id != requested for row in ordered_trajectory):
            raise ValueError("SciCUEval boundary radii do not share one base case")
        pool = pools.get(bsi_regime, ())
        if not pool:
            raise ValueError(f"SciCUEval BSI regime has no frozen base pool: {bsi_regime}")
        used_key = (bsi_regime, family)
        already_used = used.setdefault(used_key, set())
        rejected: list[str] = []
        for candidate_id in _ranked_identifiers(
            pool,
            requested=requested,
            seed=seed,
            namespace=f"scicu::valid-boundary::{bsi_regime}::{family}::{trajectory_index}",
        ):
            if candidate_id in already_used:
                continue
            candidate = cases[candidate_id]
            candidate_rows = tuple(
                (
                    replace(row, case_id=candidate_id),
                    materialize_scicu_perturbation(
                        candidate,
                        replace(row, case_id=candidate_id),
                        seed=seed,
                        standardized_radii=config.boundary.standardized_radii,
                        symbolic_depth_steps=config.boundary.symbolic_depth_steps,
                        context_composition_steps=config.boundary.context_composition_steps,
                    ),
                )
                for row in ordered_trajectory
            )
            if all(materialized.valid for _, materialized in candidate_rows):
                already_used.add(candidate_id)
                for resolved_allocation, materialized in candidate_rows:
                    resolved[resolved_allocation.allocation_id] = (
                        resolved_allocation,
                        materialized,
                        requested,
                        tuple(rejected),
                    )
                break
            rejected.append(candidate_id)
        else:
            raise ValueError(
                "no validity-preserving SciCUEval replacement exists in the frozen "
                f"base pool for {bsi_regime}/{family}/{trajectory_index}"
            )
    return resolved


def _resolve_dbp_boundary_trajectories(
    datasets: FormalDatasets,
    allocations: FormalAllocations,
    config: RunConfig,
    *,
    seed: int,
) -> tuple[
    AllocationTable,
    dict[str, MaterializedDBPPerturbation],
    dict[str, dict[str, object]],
]:
    base_pool: dict[str, list[str]] = {}
    for row in allocations.dbp.rows:
        if row.regime != "boundary_stress" and row.language_pair is not None:
            base_pool.setdefault(row.language_pair, []).append(row.case_id)
    groups: dict[tuple[str, str, str], list[AllocationRow]] = {}
    for row in allocations.dbp.rows:
        if row.regime != "boundary_stress":
            continue
        if row.language_pair is None or row.family is None or row.strength is None:
            raise ValueError("DBP15K boundary allocation is incomplete")
        trajectory_index = row.allocation_id.rsplit("::", 1)[-1]
        groups.setdefault(
            (row.language_pair, row.family, trajectory_index), []
        ).append(row)

    expected_radii = tuple(float(value) for value in config.boundary.standardized_radii)
    used: dict[tuple[str, str], set[str]] = {}
    resolved_rows: dict[str, AllocationRow] = {}
    materialized_rows: dict[str, MaterializedDBPPerturbation] = {}
    audit: dict[str, dict[str, object]] = {}
    for (language_pair, family, trajectory_index), trajectory in sorted(groups.items()):
        ordered_trajectory = tuple(sorted(trajectory, key=lambda row: float(row.strength)))
        if tuple(float(row.strength) for row in ordered_trajectory) != expected_radii:
            raise ValueError("DBP15K boundary trajectory does not cover the frozen radii")
        requested = ordered_trajectory[0].case_id
        if any(row.case_id != requested for row in ordered_trajectory):
            raise ValueError("DBP15K boundary radii do not share one reference pair")
        dataset = datasets.dbp15k.language_pairs[language_pair]
        pool = tuple(dict.fromkeys(base_pool.get(language_pair, ())))
        if not pool:
            raise ValueError(
                f"DBP15K language pair has no frozen base pool: {language_pair}"
            )
        used_key = (language_pair, family)
        already_used = used.setdefault(used_key, set())
        rejected: list[str] = []
        for candidate_id in _ranked_identifiers(
            pool,
            requested=requested,
            seed=seed,
            namespace=f"dbp::valid-boundary::{language_pair}::{family}::{trajectory_index}",
        ):
            if candidate_id in already_used:
                continue
            candidate_rows = tuple(
                (
                    replace(row, case_id=candidate_id),
                    materialize_dbp_perturbation(
                        dataset,
                        replace(row, case_id=candidate_id),
                        seed=seed,
                        standardized_radii=config.boundary.standardized_radii,
                        removal_ratios=config.boundary.dbp_removal_ratios,
                        seed_fractions=config.boundary.dbp_seed_fractions,
                    ),
                )
                for row in ordered_trajectory
            )
            if all(value.valid for _, value in candidate_rows):
                already_used.add(candidate_id)
                for resolved_allocation, value in candidate_rows:
                    resolved_rows[resolved_allocation.allocation_id] = resolved_allocation
                    materialized_rows[resolved_allocation.allocation_id] = value
                    audit[resolved_allocation.allocation_id] = {
                        "requested_case_id": requested,
                        "selection_status": (
                            "requested_valid"
                            if candidate_id == requested
                            else "deterministic_valid_replacement"
                        ),
                        "rejected_candidate_ids": tuple(rejected),
                    }
                break
            rejected.append(candidate_id)
        else:
            raise ValueError(
                "no validity-preserving DBP15K replacement exists in the frozen "
                f"reference pool for {language_pair}/{family}/{trajectory_index}"
            )

    table = AllocationTable(
        tuple(resolved_rows.get(row.allocation_id, row) for row in allocations.dbp.rows)
    )
    return table, materialized_rows, audit


def _execution_cases(
    datasets: FormalDatasets,
    allocations: FormalAllocations,
    config: RunConfig,
    *,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    AllocationTable,
    dict[str, MaterializedDBPPerturbation],
    dict[str, dict[str, object]],
]:
    cases = {case.task_id: case for case in datasets.scicueval}
    resolved_boundary = _resolve_scicu_boundary_trajectories(
        datasets,
        allocations,
        config,
        seed=seed,
    )
    rows: list[dict[str, object]] = []
    for allocation in allocations.scicu.rows:
        requested_case_id = allocation.case_id
        if allocation.regime == "boundary_stress":
            (
                allocation,
                materialized,
                requested_case_id,
                rejected_candidate_ids,
            ) = resolved_boundary[allocation.allocation_id]
            original = cases[allocation.case_id]
            execution = materialized.case
            changed_units = materialized.changed_units
            valid = materialized.valid
            validity_checks = materialized.validity_checks
            selection_status = (
                "requested_valid"
                if allocation.case_id == requested_case_id
                else "deterministic_valid_replacement"
            )
        else:
            original = cases[allocation.case_id]
            execution = original.model_copy(
                update={
                    "task_id": allocation.allocation_id,
                    "source_hash": _case_hash(original, allocation.allocation_id),
                }
            )
            changed_units = ()
            valid = True
            validity_checks = ()
            rejected_candidate_ids = ()
            selection_status = "not_applicable"
        rows.append(
            {
                "allocation": asdict(allocation),
                "original_task_id": original.task_id,
                "requested_case_id": requested_case_id,
                "selection_status": selection_status,
                "rejected_candidate_ids": list(rejected_candidate_ids),
                "execution_case": execution.model_dump(mode="json"),
                "changed_units": list(changed_units),
                "valid": valid,
                "validity_checks": list(validity_checks),
            }
        )

    resolved_dbp, dbp, dbp_audit = _resolve_dbp_boundary_trajectories(
        datasets,
        allocations,
        config,
        seed=seed,
    )
    return rows, resolved_dbp, dbp, dbp_audit


def _load_execution_cases(path: Path) -> tuple[tuple[SciCUCase, ...], dict[str, dict[str, object]]]:
    records = _read_jsonl(path)
    cases = tuple(SciCUCase.model_validate(row["execution_case"]) for row in records)
    metadata = {
        case.task_id: {
            "allocation": row["allocation"],
            "original_task_id": row["original_task_id"],
            "changed_units": row["changed_units"],
            "valid": row["valid"],
            "validity_checks": row["validity_checks"],
            "requested_case_id": row["requested_case_id"],
            "selection_status": row["selection_status"],
            "rejected_candidate_ids": row["rejected_candidate_ids"],
        }
        for case, row in zip(cases, records, strict=True)
    }
    return cases, metadata


def _training_texts(
    datasets: FormalDatasets,
    metadata: dict[str, dict[str, object]],
    split_fractions: tuple[float, float, float],
) -> tuple[str, ...]:
    withheld = {str(row["original_task_id"]) for row in metadata.values()}
    training = tuple(
        case
        for case in datasets.scicueval
        if case.task_id not in withheld
        and _split(case.task_id, split_fractions) == "train"
    )
    if not training:
        raise ValueError("formal text proxy has no leakage-free training cases")
    return tuple(
        text
        for case in training
        for text in (case.prompt, case.reference_context, case.reference_answer)
        if text.strip()
    )


def _balanced_tuning_sample(
    cases: tuple[SciCUCase, ...],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> tuple[SciCUCase, ...]:
    groups: dict[str, list[SciCUCase]] = {}
    for case in cases:
        groups.setdefault(f"{case.domain}::{case.competency}", []).append(case)
    quotas = largest_remainder(count, {key: len(value) for key, value in groups.items()})
    selected = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda case: (
                hashlib.sha256(
                    f"{seed}::{namespace}::{case.task_id}".encode("utf-8")
                ).digest(),
                case.task_id,
            ),
        )
        selected.extend(ranked[: quotas[key]])
    return tuple(sorted(selected, key=lambda case: case.task_id))


def _tuning_cases(
    datasets: FormalDatasets,
    metadata: dict[str, dict[str, object]],
    contract: ScaleContract,
    split_fractions: tuple[float, float, float],
    *,
    seed: int,
) -> tuple[tuple[SciCUCase, ...], dict[str, str]]:
    withheld = {str(row["original_task_id"]) for row in metadata.values()}
    available = tuple(case for case in datasets.scicueval if case.task_id not in withheld)
    per_split = contract.scicu_nonstress_per_regime
    selected: list[SciCUCase] = []
    split_by_id: dict[str, str] = {}
    for split in ("train", "validation"):
        pool = tuple(
            case
            for case in available
            if _split(case.task_id, split_fractions) == split
        )
        sample = _balanced_tuning_sample(
            pool,
            per_split,
            seed=seed,
            namespace=f"formal-tuning::{split}",
        )
        if len(sample) != per_split:
            raise ValueError(f"formal {split} tuning pool is too small")
        selected.extend(sample)
        split_by_id.update({case.task_id: split for case in sample})
    identifiers = [case.task_id for case in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("formal train and validation tuning pools overlap")
    return tuple(selected), split_by_id


def _evaluation_payload(
    row: FormalEvaluationRow,
    metadata: dict[str, object],
    ccs: float,
    reference_label: ResponseReferenceLabel | None = None,
) -> dict[str, object]:
    allocation = dict(metadata["allocation"])
    endpoint_success = (
        reference_label.endpoint_success
        if reference_label is not None
        else row.endpoint_success
    )
    return {
        "task_id": row.task_id,
        "original_task_id": metadata["original_task_id"],
        "allocation_id": allocation["allocation_id"],
        "regime": allocation["regime"],
        "bsi_regime": allocation.get("bsi_regime"),
        "family": allocation["family"],
        "strength": allocation["strength"],
        "model_id": row.model_id,
        "replicate": row.replicate,
        "endpoint_success": endpoint_success,
        "tsr": float(endpoint_success),
        "pcs": row.pcs,
        "ccs": float(ccs),
        "residual_consistency": row.residual_consistency,
        "node_loss": row.node_loss,
        "structure_loss": row.structure_loss,
        "constraint_loss": row.constraint_loss,
        "residuals": row.residuals.tolist(),
        "hard_feasible": row.hard_feasible,
        "transport_converged": row.transport_converged,
        "state_vector": row.state_vector.tolist(),
        "diagnostic_reason": row.diagnostic_reason,
        "reference_structural_validity": (
            int(reference_label.structural_validity)
            if reference_label is not None
            else int(row.endpoint_success)
        ),
        "reference_group": (
            reference_label.failure_group
            if reference_label is not None
            else ("valid" if row.endpoint_success else "unadjudicated_response")
        ),
        "reference_label_source": (
            reference_label.source if reference_label is not None else "endpoint_fallback"
        ),
        "perturbation_valid": bool(metadata.get("valid", True)),
        "source": "computed",
    }


def _formal_row(record: dict[str, object]) -> FormalEvaluationRow:
    return FormalEvaluationRow(
        task_id=str(record["task_id"]),
        model_id=str(record["model_id"]),
        replicate=int(record["replicate"]),
        endpoint_success=bool(record["endpoint_success"]),
        tsr=float(record["tsr"]),
        pcs=float(record["pcs"]),
        residual_consistency=float(record["residual_consistency"]),
        node_loss=float(record["node_loss"]),
        structure_loss=float(record["structure_loss"]),
        constraint_loss=float(record["constraint_loss"]),
        residuals=np.asarray(record["residuals"], dtype=np.float64),
        hard_feasible=bool(record["hard_feasible"]),
        transport_converged=bool(record["transport_converged"]),
        state_vector=np.asarray(record["state_vector"], dtype=np.float64),
        diagnostic_reason=str(record["diagnostic_reason"]),
    )


def _split(
    task_id: str,
    fractions: tuple[float, float, float],
) -> str:
    if len(fractions) != 3 or any(value <= 0.0 for value in fractions):
        raise ValueError("train/validation/test split fractions must be positive triples")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-12, rel_tol=0.0):
        raise ValueError("train/validation/test split fractions must sum to one")
    # Percent buckets preserve the existing task-keyed split for the paper's
    # 70/15/15 fractions while making the configured cut points authoritative.
    bucket = (
        int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big")
        % 100
    ) / 100.0
    train_edge = fractions[0]
    validation_edge = train_edge + fractions[1]
    return "train" if bucket < train_edge else "validation" if bucket < validation_edge else "test"


def _infer_models(
    checkpoints,
    rows: tuple[FormalEvaluationRow, ...],
    config: RunConfig,
) -> tuple[list[list[int]], list[float]]:
    states = torch.as_tensor(
        np.stack([row.state_vector for row in rows]),
        dtype=torch.float64,
    )
    seedwise: list[list[int]] = []
    energies = []
    for checkpoint in checkpoints:
        model = StateFieldModel(
            dimensions=config.state_field.dimensions,
            energy_dimensions=config.state_field.energy_dimensions,
            activation=config.state_field.activation,
            gelu_approximate=config.state_field.gelu_approximate,
            linear_bias=config.state_field.linear_bias,
            initialization=config.state_field.initialization,
            dropout=config.state_field.dropout,
            dropout_inplace=config.state_field.dropout_inplace,
            dtype=torch.float64,
        )
        model.load_state_dict(checkpoint.state_dict)
        model.eval()
        with torch.inference_mode():
            logits, energy = model(states)
        seedwise.append(logits.argmax(dim=1).cpu().tolist())
        energies.append(energy.reshape(-1).cpu().numpy())
    mean_energy = np.mean(np.stack(energies), axis=0)
    return seedwise, [float(value) for value in mean_energy]


def _select_open_set_frames(
    frame: pd.DataFrame,
    split_fractions: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame.copy()
    selected["split"] = selected["original_task_id"].map(
        lambda task_id: _split(str(task_id), split_fractions)
    )
    validation = selected[
        selected["regime"].isin(("iid", "motif_or_structure"))
        & (selected["split"] == "validation")
    ].copy()
    # The paper reports UPDQ and SC in every evaluation regime.  The known
    # support is still frozen from validation IID/motif rows, while each
    # regime's held-out test rows are diagnosed separately below.
    unknown = selected[selected["split"] == "test"].copy()
    if "reference_group" in selected:
        names = sorted(set(validation["reference_group"].astype(str)))
        mapping = {name: index for index, name in enumerate(names)}
        validation["reference_group_code"] = (
            validation["reference_group"].astype(str).map(mapping)
        )
        unknown["reference_group_code"] = unknown["reference_group"].astype(str).map(
            lambda value: mapping.get(value, len(mapping))
        )
    return validation, unknown


def _encode_open_set_groups(
    validation: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = validation.copy()
    evaluation = candidates.copy()
    names = sorted(set(support["reference_group"].astype(str)))
    if not names:
        raise ValueError("open-set validation support has no reference groups")
    mapping = {name: index for index, name in enumerate(names)}
    support["reference_group_code"] = (
        support["reference_group"].astype(str).map(mapping)
    )
    evaluation["reference_group_code"] = evaluation["reference_group"].astype(
        str
    ).map(lambda value: mapping.get(value, len(mapping)))
    return support, evaluation


def _open_bundle(
    frame: pd.DataFrame,
    *,
    motif_center: np.ndarray | None = None,
) -> OpenSetBundle:
    states = np.stack(frame["state_vector"].map(lambda value: np.asarray(value, dtype=np.float64)))
    residuals = np.stack(frame["residuals"].map(lambda value: np.asarray(value, dtype=np.float64)))
    center = (
        np.median(residuals, axis=0)
        if motif_center is None
        else np.asarray(motif_center, dtype=np.float64)
    )
    if center.shape != (residuals.shape[1],) or not np.isfinite(center).all():
        raise ValueError("open-set motif center must be a finite validation-derived vector")
    motif = residuals - center[None, :]
    return OpenSetBundle(
        states=states,
        residuals=residuals,
        motif_mismatch=motif,
        structural_confidence=np.clip(frame["pcs"].to_numpy(dtype=np.float64), 0.0, 1.0),
        structural_validity=frame["reference_structural_validity"].to_numpy(dtype=np.int64),
        reference_groups=frame["reference_group_code"].to_numpy(dtype=np.int64),
    )


def _baseline_open_bundle(
    frame: pd.DataFrame,
    *,
    motif_center: np.ndarray,
) -> OpenSetBundle:
    score = np.clip(frame["baseline_score"].to_numpy(dtype=np.float64), 0.0, 1.0)
    residuals = (1.0 - score)[:, None]
    center = np.asarray(motif_center, dtype=np.float64)
    if center.shape != (1,) or not np.isfinite(center).all():
        raise ValueError("baseline open-set center must be validation-derived")
    return OpenSetBundle(
        states=np.column_stack((score, 1.0 - score)),
        residuals=residuals,
        motif_mismatch=residuals - center[None, :],
        structural_confidence=score,
        structural_validity=frame["reference_structural_validity"].to_numpy(
            dtype=np.int64
        ),
        reference_groups=frame["reference_group_code"].to_numpy(dtype=np.int64),
    )


def _baseline_score_frame(
    metric_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    baseline: BaselineKind,
) -> pd.DataFrame:
    metrics = metric_frame.copy()
    metrics["case_id"] = (
        metrics["task_id"].astype(str)
        + "::"
        + metrics["model_id"].astype(str)
        + "::"
        + metrics["replicate"].astype(str)
    )
    scores = baseline_frame[
        (baseline_frame["dataset"] == "scicueval")
        & (baseline_frame["baseline"] == baseline.value)
        & ~baseline_frame["case_id"].astype(str).str.startswith("baseline-original\t")
    ][["case_id", "score"]].rename(columns={"score": "baseline_score"})
    return metrics.merge(scores, on="case_id", how="inner", validate="one_to_one")


def _build_baseline_open_set_rows(
    metric_frame: pd.DataFrame,
    tuning_metric_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    config: RunConfig,
    *,
    scale: Literal["smoke", "paper"],
) -> list[dict[str, object]]:
    validation = _baseline_score_frame(
        tuning_metric_frame[tuning_metric_frame["split"] == "validation"],
        baseline_frame,
        BaselineKind.STRUCTURE_AWARE,
    )
    candidates = _baseline_score_frame(
        metric_frame,
        baseline_frame,
        BaselineKind.STRUCTURE_AWARE,
    )
    validation, candidates = _encode_open_set_groups(
        validation,
        candidates,
    )
    if len(validation) < 2 or len(candidates) < 3:
        raise ValueError("baseline open-set split is too small")
    center = np.median(
        (1.0 - validation["baseline_score"].to_numpy(dtype=np.float64))[:, None],
        axis=0,
    )
    validation_bundle = _baseline_open_bundle(validation, motif_center=center)
    uncertainty = _baseline_score_frame(
        metric_frame,
        baseline_frame,
        BaselineKind.UNCERTAINTY,
    )[["case_id", "baseline_score"]].rename(
        columns={"baseline_score": "uncertainty_score"}
    )
    candidates = candidates.merge(
        uncertainty,
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    groups = (
        tuple(
            (regime, candidates[candidates["regime"] == regime].copy())
            for regime in config.regimes.names
        )
        if scale == "paper"
        else (("pooled_smoke", candidates.copy()),)
    )
    rows: list[dict[str, object]] = []
    for regime, group in groups:
        if len(group) < 3:
            raise ValueError(
                f"baseline open-set test split is too small for regime {regime}: "
                f"{len(group)}"
            )
        discovery = run_open_set_analysis(
            validation_bundle,
            _baseline_open_bundle(group, motif_center=center),
            config,
        )
        calibration_method = (
            BaselineKind.STRUCTURE_AWARE
            if regime in ("boundary_stress", "pooled_smoke")
            else BaselineKind.UNCERTAINTY
        )
        confidence = (
            group["baseline_score"].to_numpy(dtype=np.float64)
            if calibration_method is BaselineKind.STRUCTURE_AWARE
            else group["uncertainty_score"].to_numpy(dtype=np.float64)
        )
        if not np.isfinite(confidence).all():
            raise ValueError("baseline calibration scores are incomplete")
        calibration = compute_sc(
            confidence=np.clip(confidence, 0.0, 1.0),
            structural_validity=group["reference_structural_validity"].to_numpy(
                dtype=np.int64
            ),
            bins=config.open_set.calibration_bins,
            minimum_bin_samples=config.open_set.calibration_minimum_bin_samples,
            minimum_bin_fraction=config.open_set.calibration_minimum_bin_fraction,
        )
        rows.append(
            {
                "regime": regime,
                "updq": discovery.updq,
                "seedwise_updq": discovery.seedwise_updq,
                "structural_calibration": calibration.sc,
                "seedwise_structural_calibration": tuple(
                    calibration.sc for _seed in config.paper.seeds
                ),
                "calibration_ece": calibration.ece,
                "updq_baseline": BaselineKind.STRUCTURE_AWARE.value,
                "sc_baseline": calibration_method.value,
                "candidate_count": len(group),
                "support_split": "independent_validation",
                "candidate_split": "formal_final_test",
                "source": "computed",
            }
        )
    return rows


def _baseline_regime(score: float, low: float, high: float) -> int:
    return 0 if score <= low else 2 if score >= high else 1


def _build_baseline_boundary_rows(
    metric_frame: pd.DataFrame,
    tuning_metric_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    config: RunConfig,
    formal_config: FormalExecutionConfig,
) -> list[dict[str, object]]:
    scored = _baseline_score_frame(
        metric_frame,
        baseline_frame,
        BaselineKind.STRUCTURE_AWARE,
    )
    validation = _baseline_score_frame(
        tuning_metric_frame[tuning_metric_frame["split"] == "validation"],
        baseline_frame,
        BaselineKind.STRUCTURE_AWARE,
    )
    if validation.empty:
        raise ValueError("baseline boundary thresholds require validation scores")
    low = float(
        np.quantile(
            validation["baseline_score"].to_numpy(dtype=np.float64),
            formal_config.invalid_pcs_quantile,
        )
    )
    high = float(
        np.quantile(
            validation["baseline_score"].to_numpy(dtype=np.float64),
            formal_config.stable_pcs_quantile,
        )
    )
    if low > high:
        raise ValueError("baseline validation thresholds are inconsistent")

    originals = baseline_frame[
        (baseline_frame["dataset"] == "scicueval")
        & (baseline_frame["baseline"] == BaselineKind.STRUCTURE_AWARE.value)
        & baseline_frame["case_id"].astype(str).str.startswith("baseline-original\t")
    ]
    original_states: dict[tuple[str, str, str, str, int], int] = {}
    for row in originals.itertuples(index=False):
        marker, original_id, bsi_regime, family, model_id, replicate = str(
            row.case_id
        ).split("\t")
        if marker != "baseline-original":
            raise ValueError("unexpected baseline original marker")
        original_states[
            (original_id, bsi_regime, family, model_id, int(replicate))
        ] = _baseline_regime(float(row.score), low, high)

    evaluations_by_regime: dict[str, list[BoundaryEvaluation]] = {}
    boundary = scored[scored["regime"] == "boundary_stress"]
    for row in boundary.itertuples(index=False):
        bsi_regime = str(row.bsi_regime)
        key = (
            str(row.original_task_id),
            bsi_regime,
            str(row.family),
            str(row.model_id),
            int(row.replicate),
        )
        evaluations_by_regime.setdefault(bsi_regime, []).append(
            BoundaryEvaluation(
                case_id=(
                    f"{row.original_task_id}\t{row.model_id}\t{int(row.replicate)}"
                ),
                family=str(row.family),
                seed=config.paper.seeds[0],
                radius=float(row.strength),
                original_regime=original_states[key],
                perturbed_regime=_baseline_regime(float(row.baseline_score), low, high),
                valid=bool(row.perturbation_valid),
            )
        )
    return [
        asdict(row)
        | {
            "regime": bsi_regime,
            "baseline": BaselineKind.STRUCTURE_AWARE.value,
            "threshold_low": low,
            "threshold_high": high,
            "threshold_split": "validation",
            "source": "computed",
        }
        for bsi_regime, evaluations in sorted(evaluations_by_regime.items())
        for row in compute_boundary_rows(
            tuple(evaluations),
            rho_max=max(config.boundary.standardized_radii),
        )
    ]


def _aggregate_metrics(
    metric_frame: pd.DataFrame,
    boundary_frame: pd.DataFrame,
    open_frame: pd.DataFrame,
    dbp_frame: pd.DataFrame,
) -> dict[str, object]:
    return {
        "tsr": float(metric_frame["tsr"].mean()),
        "pcs": float(metric_frame["pcs"].mean()),
        "ccs": float(metric_frame["ccs"].mean()),
        "BSI": float(boundary_frame["bsi_raw"].mean()),
        "UPDQ": float(open_frame["updq"].mean()),
        "SC": float(open_frame["structural_calibration"].mean()),
        "SCS": float(dbp_frame["scs"].mean()),
        "TP": float(dbp_frame["transport_purity"].mean()),
        "source": "computed",
    }


def _build_main_statistic_pairs(
    metric_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    boundary_frame: pd.DataFrame,
    baseline_boundary_frame: pd.DataFrame,
    open_frame: pd.DataFrame,
    baseline_open_frame: pd.DataFrame,
    dbp_frame: pd.DataFrame,
    *,
    baseline_task_success_threshold: float,
) -> tuple[ComputedPairRow, ...]:
    pairs: list[ComputedPairRow] = []
    scicu = metric_frame.copy()
    scicu["case_id"] = (
        scicu["task_id"].astype(str)
        + "::"
        + scicu["model_id"].astype(str)
        + "::"
        + scicu["replicate"].astype(str)
    )
    scicu_baselines = baseline_frame[
        (baseline_frame["dataset"] == "scicueval")
        & (baseline_frame["baseline"] == "structure_aware")
    ]
    scicu_joined = scicu.merge(
        scicu_baselines[["case_id", "baseline", "score"]],
        on="case_id",
        how="inner",
        validate="many_to_many",
    )
    group_columns = ["original_task_id", "model_id", "replicate"]
    baseline_anchor = scicu_joined.groupby(group_columns)["score"].transform("mean")
    scicu_joined["baseline_ccs"] = np.clip(
        1.0 - np.abs(scicu_joined["score"] - baseline_anchor),
        0.0,
        1.0,
    )
    for row in scicu_joined.itertuples(index=False):
        regime = str(row.regime)
        reference_regime = "motif_split" if regime == "motif_or_structure" else regime
        for metric, column in (
            ("TSR", "tsr"),
            ("PCS", "pcs"),
            ("CCS", "ccs"),
        ):
            if metric == "TSR":
                comparison_value = float(
                    row.score >= baseline_task_success_threshold
                )
            elif metric == "CCS":
                comparison_value = float(row.baseline_ccs)
            else:
                comparison_value = float(row.score)
            reference_path = ("scicueval", metric, reference_regime, "delta")
            pairs.append(
                ComputedPairRow(
                    pair_id=str(row.original_task_id),
                    dataset="scicueval",
                    regime=regime,
                    metric=metric,
                    proposed_method="sciprism_x",
                    comparison_method=str(row.baseline),
                    proposed_value=float(getattr(row, column)),
                    comparison_value=comparison_value,
                    direction="higher",
                    family="manuscript_scicueval_main",
                    reference_path=reference_path,
                )
            )

    proposed_boundary = (
        boundary_frame.groupby(["regime", "case_id", "family"], as_index=False)["bsi_raw"]
        .mean()
        .rename(columns={"bsi_raw": "proposed_bsi"})
    )
    comparison_boundary = baseline_boundary_frame[
        baseline_boundary_frame["baseline"] == BaselineKind.STRUCTURE_AWARE.value
    ][["regime", "case_id", "family", "bsi_raw"]].rename(
        columns={"bsi_raw": "comparison_bsi"}
    )
    boundary_joined = proposed_boundary.merge(
        comparison_boundary,
        on=["regime", "case_id", "family"],
        how="inner",
        validate="one_to_one",
    )
    for row in boundary_joined.itertuples(index=False):
        regime = str(row.regime)
        if regime == "pooled_smoke":
            continue
        reference_regime = "motif_split" if regime == "motif_or_structure" else regime
        pairs.append(
            ComputedPairRow(
                pair_id=str(row.case_id).split("\t", 1)[0],
                dataset="scicueval",
                regime=regime,
                metric="BSI",
                proposed_method="sciprism_x",
                comparison_method=BaselineKind.STRUCTURE_AWARE.value,
                proposed_value=float(row.proposed_bsi),
                comparison_value=float(row.comparison_bsi),
                direction="lower",
                family="manuscript_scicueval_main",
                reference_path=("scicueval", "BSI", reference_regime, "delta"),
            )
        )

    open_joined = open_frame.merge(
        baseline_open_frame,
        on="regime",
        how="inner",
        suffixes=("_proposed", "_comparison"),
        validate="one_to_one",
    )
    for row in open_joined.itertuples(index=False):
        regime = str(row.regime)
        if regime == "pooled_smoke":
            continue
        reference_regime = "motif_split" if regime == "motif_or_structure" else regime
        for metric, proposed_column, comparison_column, method_column in (
            (
                "UPDQ",
                "seedwise_updq_proposed",
                "seedwise_updq_comparison",
                "updq_baseline",
            ),
            (
                "SC",
                "seedwise_structural_calibration_proposed",
                "seedwise_structural_calibration_comparison",
                "sc_baseline",
            ),
        ):
            proposed_values = tuple(float(value) for value in getattr(row, proposed_column))
            comparison_values = tuple(
                float(value) for value in getattr(row, comparison_column)
            )
            if len(proposed_values) != len(comparison_values) or not proposed_values:
                raise ValueError("open-set seed-wise baseline comparison is incomplete")
            for seed_index, (proposed_value, comparison_value) in enumerate(
                zip(proposed_values, comparison_values, strict=True)
            ):
                pairs.append(
                    ComputedPairRow(
                        pair_id=f"open-set::{regime}::seed::{seed_index}",
                        dataset="scicueval",
                        regime=regime,
                        metric=metric,
                        proposed_method="sciprism_x",
                        comparison_method=str(getattr(row, method_column)),
                        proposed_value=proposed_value,
                        comparison_value=comparison_value,
                        direction="higher",
                        family="manuscript_scicueval_main",
                        reference_path=(
                            "scicueval",
                            metric,
                            reference_regime,
                            "delta",
                        ),
                    )
                )

    dbp = dbp_frame.copy()
    dbp["case_id"] = (
        dbp["allocation_id"].astype(str)
        + "::"
        + dbp["language_pair"].astype(str)
        + "::"
        + dbp["source_id"].astype(str)
        + "::"
        + dbp["target_id"].astype(str)
    )
    dbp_baselines = baseline_frame[
        (baseline_frame["dataset"] == "dbp15k")
        & (baseline_frame["baseline"] == "sinkhorn")
    ].copy()
    dbp_baselines.loc[
        dbp_baselines["metric"] == "alignment_similarity", "metric"
    ] = "SCS"
    dbp_baselines = dbp_baselines[dbp_baselines["metric"].isin(("SCS", "TP"))]
    dbp_joined = dbp.merge(
        dbp_baselines[["case_id", "baseline", "metric", "score"]],
        on="case_id",
        how="inner",
        validate="many_to_many",
    )
    for row in dbp_joined.itertuples(index=False):
        regime = str(row.regime)
        reference_regime = "structure_split" if regime == "motif_or_structure" else regime
        metric = str(row.metric)
        column = "scs" if metric == "SCS" else "transport_purity"
        reference_path = ("dbp15k", metric, reference_regime, "delta")
        pairs.append(
            ComputedPairRow(
                pair_id=(
                    f"{row.allocation_id}::{row.language_pair}::"
                    f"{row.source_id}::{row.target_id}"
                ),
                dataset="dbp15k",
                regime=regime,
                metric=metric,
                proposed_method="sciprism_x",
                comparison_method=str(row.baseline),
                proposed_value=float(getattr(row, column)),
                comparison_value=float(row.score),
                direction="higher",
                family="manuscript_dbp15k_main",
                reference_path=reference_path,
            )
        )
    if not pairs:
        raise ValueError("main statistics require matched computed baseline rows")
    return tuple(pairs)


_BASELINE_TOKEN = re.compile(r"[A-Za-z0-9]+")
_HEDGE = re.compile(
    r"\b(?:approximately|about|may|might|could|possibly|likely|unlikely|"
    r"assuming|suggests?|appears?)\b",
    re.IGNORECASE,
)


def _baseline_units(text: str) -> tuple[str, ...]:
    units = tuple(
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    )
    return units or (text.strip() or "empty",)


def _baseline_tokens(text: str) -> set[str]:
    return {value.casefold() for value in _BASELINE_TOKEN.findall(text)}


def _baseline_representation(
    case: SciCUCase,
    response: ResponseRecord,
    dimensions: int,
) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float64)
    text = f"{case.prompt}\n{case.reference_context}\n{response.response_text}"
    for token in _BASELINE_TOKEN.findall(text.casefold()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % vector.size
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def _baseline_graph(
    prefix: str,
    units: tuple[str, ...],
) -> tuple[GraphView, np.ndarray]:
    nodes = tuple(f"{prefix}-{index}" for index in range(len(units)))
    edges = tuple(
        GraphEdge(nodes[index], "sequence", nodes[index + 1])
        for index in range(len(nodes) - 1)
    )
    adjacency = np.zeros((1, len(nodes), len(nodes)), dtype=np.float64)
    for index in range(len(nodes) - 1):
        adjacency[0, index, index + 1] = 1.0
    return GraphView(nodes, edges), adjacency


def _baseline_pairwise_cost(
    source_units: tuple[str, ...],
    target_units: tuple[str, ...],
) -> np.ndarray:
    matrix = np.empty((len(source_units), len(target_units)), dtype=np.float64)
    for row, source in enumerate(source_units):
        left = _baseline_tokens(source)
        for column, target in enumerate(target_units):
            right = _baseline_tokens(target)
            union = left | right
            similarity = 1.0 if not union else len(left & right) / len(union)
            matrix[row, column] = 1.0 - similarity
    return matrix


def _matched_contexts(case: SciCUCase, max_rotations: int) -> tuple[str, ...]:
    units = _baseline_units(case.reference_context)
    if len(units) == 1:
        return (case.reference_context,)
    count = min(max_rotations, len(units) - 1)
    return tuple(
        " ".join(units[offset:] + units[:offset])
        for offset in range(1, count + 1)
    )


def _cross_context_consistency(
    *,
    case: SciCUCase,
    response: ResponseRecord,
    context: FormalEvaluationContext,
    reference_pcs: float,
    controls: AblationControls | None = None,
) -> float:
    matched_scores = tuple(
        evaluate_scicu_response(
            case=case.model_copy(update={"reference_context": matched_context}),
            response=response,
            context=context,
            controls=controls,
        ).pcs
        for matched_context in _matched_contexts(
            case,
            context.config.baseline_reconstruction.matched_context_rotations,
        )
    )
    return float(
        np.clip(
            1.0
            - np.mean(
                [abs(reference_pcs - matched_score) for matched_score in matched_scores]
            ),
            0.0,
            1.0,
        )
    )


def _baseline_case_scicu(
    record: dict[str, object],
    case: SciCUCase,
    response: ResponseRecord,
    validation_support: np.ndarray,
    *,
    representation_dimensions: int,
    case_id: str | None = None,
) -> EvaluatedCase:
    source_units = _baseline_units(f"{case.prompt}\n{case.reference_context}")
    target_units = _baseline_units(response.response_text)
    source, source_adjacency = _baseline_graph("task", source_units)
    target, target_adjacency = _baseline_graph("response", target_units)
    pairwise_cost = _baseline_pairwise_cost(source_units, target_units)
    representation = _baseline_representation(
        case,
        response,
        representation_dimensions,
    )
    hedge_count = len(_HEDGE.findall(response.response_text))
    confidence = 1.0 - min(1.0, hedge_count / max(1, len(target_units)))
    endpoint_success = bool(record["endpoint_success"])
    return EvaluatedCase(
        case_id=case_id or f"{case.task_id}::{response.model_id}::{response.replicate}",
        dataset="scicueval",
        task=case.prompt,
        response=response.response_text,
        reference_context=case.reference_context,
        predicted_target=(
            case.reference_answer if endpoint_success else response.response_text
        ),
        reference_target=case.reference_answer,
        response_segments=tuple(part for part in re.split(r"(?<=[.!?])\s+", response.response_text) if part),
        evidence_spans=tuple(part for part in case.reference_context.splitlines() if part) or (case.reference_context,),
        task_metadata=(("domain", case.domain), ("competency", case.competency)),
        confidence=confidence,
        entropy=1.0 - confidence,
        representation=representation,
        validation_support=validation_support,
        source_graph=source,
        target_graph=target,
        pairwise_cost=pairwise_cost,
        source_mass=np.full(len(source_units), 1.0 / len(source_units)),
        target_mass=np.full(len(target_units), 1.0 / len(target_units)),
        source_adjacency=source_adjacency,
        target_adjacency=target_adjacency,
        candidate_scores=np.array([confidence], dtype=np.float64),
        correct_candidate_index=0,
        residuals=np.asarray(record["residuals"], dtype=np.float64),
        transport_plan=np.zeros(pairwise_cost.shape, dtype=np.float64),
        generator_identity="withheld-generator",
        motif_graph="withheld-motif-graph",
        scientific_constraint_mask=np.ones(pairwise_cost.shape, dtype=np.bool_),
        boundary_label="withheld-boundary",
        open_set_category="withheld-open-set",
    )


def _dbp_baseline_validation_cases(
    datasets: FormalDatasets,
    config: RunConfig,
    *,
    seed: int,
    validation_fraction: float,
) -> tuple[EvaluatedCase, ...]:
    """Hold out one shared validation subset from the official 30% seed pool."""

    validation_datasets = {}
    rows: list[AllocationRow] = []
    for language_pair, dataset in sorted(datasets.dbp15k.language_pairs.items()):
        ranked = tuple(
            sorted(
                dataset.seed_pairs,
                key=lambda pair: (
                    hashlib.sha256(
                        f"{seed}::dbp-baseline-validation::{language_pair}::{pair}".encode(
                            "utf-8"
                        )
                    ).digest(),
                    pair,
                ),
            )
        )
        if len(ranked) < 2:
            raise ValueError("DBP15K baseline tuning requires at least two seed pairs")
        validation_count = min(
            len(ranked) - 1,
            max(1, int(round(len(ranked) * validation_fraction))),
        )
        validation_pairs = ranked[:validation_count]
        training_pairs = ranked[validation_count:]
        validation_datasets[language_pair] = replace(
            dataset,
            seed_pairs=training_pairs,
            reference_pairs=validation_pairs,
        )
        rows.extend(
            AllocationRow(
                dataset="dbp15k",
                allocation_id=f"dbp15k::validation::{language_pair}::{index:05d}",
                case_id=f"{language_pair}::{source_id}::{target_id}",
                regime="iid",
                family=None,
                strength=None,
                language_pair=language_pair,
            )
            for index, (source_id, target_id) in enumerate(validation_pairs)
        )
    validation_bundle = FormalDatasets(
        scicueval=datasets.scicueval,
        dbp15k=DBP15KDataset(validation_datasets),
    )
    return build_dbp_baseline_cases(
        validation_bundle,
        AllocationTable(tuple(rows)),
        config,
        sinkhorn_epsilon=(
            config.baseline_reconstruction.dbp_validation_transport_epsilon
        ),
    )


@dataclass(frozen=True, slots=True)
class _AblationBundle:
    scicu_cases: tuple[SciCUCase, ...]
    responses: tuple[ResponseRecord, ...]
    context: FormalEvaluationContext
    main_rows: tuple[dict[str, object], ...]
    dbp_rows: tuple[DBPCaseResult, ...]
    bsi_by_task: dict[str, float]
    updq: float
    sc: float


class _AblationEvaluator:
    def __init__(self, controls: AblationControls) -> None:
        self.resolved_controls = controls

    def evaluate(self, bundle: object) -> tuple[PerCaseMetric, ...]:
        if not isinstance(bundle, _AblationBundle):
            raise TypeError("workflow ablation evaluator requires _AblationBundle")
        case_by_id = {case.task_id: case for case in bundle.scicu_cases}
        response_by_key = {
            (row.task_id, row.model_id, row.replicate): row for row in bundle.responses
        }
        main_by_key = {
            (str(row["task_id"]), str(row["model_id"]), int(row["replicate"])): row
            for row in bundle.main_rows
        }
        rows: list[PerCaseMetric] = []
        for key, main in main_by_key.items():
            case = case_by_id[key[0]]
            response = response_by_key[key]
            if self.resolved_controls == AblationControls():
                evaluated = _formal_row(main)
                ccs = float(main["ccs"])
            elif (
                not self.resolved_controls.boundary_estimation
                or not self.resolved_controls.open_set
            ) and all(
                (
                    self.resolved_controls.motif_structure,
                    self.resolved_controls.hard_constraints,
                    self.resolved_controls.soft_penalties,
                    self.resolved_controls.residual_channels,
                    self.resolved_controls.state_field,
                    self.resolved_controls.transport_mode == "optimal",
                )
            ):
                evaluated = _formal_row(main)
                ccs = float(main["ccs"])
            else:
                evaluated = evaluate_scicu_response(
                    case=case,
                    response=response,
                    context=bundle.context,
                    controls=self.resolved_controls,
                )
                ccs = _cross_context_consistency(
                    case=case,
                    response=response,
                    context=bundle.context,
                    reference_pcs=evaluated.pcs,
                    controls=self.resolved_controls,
                )
            case_id = f"{key[0]}\t{key[1]}\t{key[2]}"
            residual_quality = float(np.clip(1.0 - np.mean(evaluated.residuals), 0.0, 1.0))
            bsi = (
                float(evaluated.hard_feasible)
                if not self.resolved_controls.boundary_estimation
                else bundle.bsi_by_task.get(key[0], float(evaluated.hard_feasible))
            )
            updq = residual_quality if not self.resolved_controls.open_set else bundle.updq
            sc = float(evaluated.hard_feasible) if not self.resolved_controls.open_set else bundle.sc
            for metric, value in (
                ("PCS", evaluated.pcs),
                ("CCS", ccs),
                ("BSI", bsi),
                ("UPDQ", updq),
                ("SC", sc),
            ):
                rows.append(PerCaseMetric(case_id, "scicueval", metric, float(value)))
        for result in bundle.dbp_rows:
            case_id = (
                f"{result.allocation_id}\t{result.language_pair}\t"
                f"{result.source_id}\t{result.target_id}"
            )
            scs = result.scs
            tp = result.transport_purity
            if not self.resolved_controls.motif_structure:
                scs = result.hits_at_1
            if not self.resolved_controls.soft_penalties:
                tp = result.scs
            if self.resolved_controls.transport_mode == "greedy":
                tp = result.hits_at_1
            rows.append(PerCaseMetric(case_id, "dbp15k", "SCS", float(scs)))
            rows.append(PerCaseMetric(case_id, "dbp15k", "TP", float(tp)))
        return tuple(rows)


def _statistics_result(bootstrap: pd.DataFrame, significance: pd.DataFrame) -> FormalStatisticsResult:
    comparisons = tuple(
        ComputedComparisonRow(
            comparison_id=row.comparison_id,
            family=row.family,
            dataset=row.dataset,
            regime=row.regime,
            metric=row.metric,
            proposed_method=row.proposed_method,
            comparison_method=row.comparison_method,
            observed_difference=float(row.observed_difference),
            ci_lower=float(row.ci_lower),
            ci_upper=float(row.ci_upper),
            p_value=float(row.p_value),
            resamples=int(row.resamples),
            confidence_level=float(row.confidence_level),
            seed=int(row.seed),
            direction=row.direction,
            pair_ids=tuple(row.pair_ids),
            reference_path=tuple(row.reference_path),
        )
        for row in bootstrap.itertuples(index=False)
    )
    significance_rows = tuple(
        SignificanceRow(
            comparison_id=row.comparison_id,
            family=row.family,
            raw_p_value=float(row.raw_p_value),
            adjusted_p_value=float(row.adjusted_p_value),
            reject=bool(row.reject),
            family_members=tuple(row.family_members),
        )
        for row in significance.itertuples(index=False)
    )
    return FormalStatisticsResult(comparisons, significance_rows)


def run_formal_experiment(
    config: RunConfig,
    formal_config: FormalExecutionConfig,
    *,
    scale: Literal["smoke", "paper"],
    output_root: Path,
    resume: bool = True,
    response_manifest_path: Path | None = None,
    paper_reference_path: Path | None = None,
    judge_response_path: Path | None = None,
) -> FormalRunReceipt:
    contract = resolve_scale_contract(config, formal_config, scale=scale)
    split_fractions = tuple(formal_config.split_fractions)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage_root = output / ("stages" if resume else f"stages-{uuid.uuid4().hex}")
    store = StageStore(stage_root)
    implementation = _implementation_hash()
    config_digest = _config_hash(config)
    formal_digest = _config_hash(formal_config)
    source_registry = PROJECT_ROOT / "data" / "manifests" / "source_registry.json"
    evaluation_response_files, tuning_response_files, response_input_hashes = (
        _load_response_file_manifest(
            response_manifest_path,
            model_ids=config.generators.model_ids,
        )
    )
    judge_requester, judge_response_hash = _load_judge_response_replay(
        judge_response_path
    )
    if config.track == "A":
        expected_models = set(config.generators.model_ids)
        if set(evaluation_response_files) != expected_models:
            raise TrackIsolationError(
                "Track A response manifest must cover every generator for evaluation"
            )
        if set(tuning_response_files) != expected_models:
            raise TrackIsolationError(
                "Track A response manifest must cover every generator for tuning"
            )
    configured_reference = (
        Path(config.published_targets) if config.published_targets is not None else None
    )
    profile_target_hash: str | None = None
    if configured_reference is not None:
        profile_target = configured_reference
        if not profile_target.is_absolute():
            profile_target = PROJECT_ROOT / profile_target
        profile_target_hash = _sha256(profile_target.resolve())
    paper_reference = paper_reference_path or configured_reference
    if paper_reference is None:
        raise TrackIsolationError(
            "formal paper comparison requires a post-computation paper reference file"
        )
    if not paper_reference.is_absolute():
        paper_reference = PROJECT_ROOT / paper_reference
    paper_reference = paper_reference.resolve()
    if not paper_reference.is_file():
        raise FileNotFoundError(f"paper reference is unavailable: {paper_reference}")
    source_digest = canonical_sha256(
        {
            "source_registry": _sha256(source_registry),
        }
    )
    receipts = []

    def run_stage(name: str, upstream: dict[str, str], producer):
        print(f"[formal] stage={name} status=starting", flush=True)
        receipt = store.run(
            name,
            input_hashes=upstream,
            implementation_hash=implementation,
            producer=producer,
        )
        status = "resumed" if receipt.resumed else "computed"
        print(f"[formal] stage={name} status={status}", flush=True)
        receipts.append(receipt)
        return receipt

    extraction_root = output / "extracted"
    datasets = load_formal_datasets(
        PROJECT_ROOT / "data",
        extraction_root=extraction_root,
        dbp_protocol=config.datasets.dbp15k,
    )

    sources = run_stage(
        "sources",
        {"registry": source_digest, "config": config_digest},
        lambda path: _write_json(
            path / "source_summary.json",
            {
                "scicueval_cases": len(datasets.scicueval),
                "dbp15k_language_pairs": sorted(datasets.dbp15k.language_pairs),
                "dbp15k_seed_fraction": config.datasets.dbp15k.seed_fraction,
                "dbp15k_full_candidate_pool": config.datasets.dbp15k.full_candidate_pool,
                "dbp15k_candidate_pruning": config.datasets.dbp15k.candidate_pruning,
                "source_registry_sha256": _sha256(source_registry),
            },
        ),
    )

    def produce_allocations(path: Path) -> None:
        allocations = build_formal_allocations(
            datasets,
            contract,
            seed=config.paper.seeds[0],
            strengths=config.boundary.standardized_radii,
        )
        _write_json(
            path / "allocations.json",
            {
                "scicueval": _allocation_payload(allocations.scicu),
                "dbp15k": _allocation_payload(allocations.dbp),
            },
        )

    allocations_stage = run_stage(
        "allocations",
        {"sources": sources.content_hash, "formal_config": formal_digest},
        produce_allocations,
    )
    allocations = _load_allocations(allocations_stage.path / "allocations.json")

    def produce_perturbations(path: Path) -> None:
        scicu_rows, resolved_dbp, dbp_rows, dbp_selection_audit = _execution_cases(
            datasets,
            allocations,
            config,
            seed=config.paper.seeds[0],
        )
        _write_jsonl(path / "execution_cases.jsonl", scicu_rows)
        _write_json(
            path / "resolved_dbp_allocations.json",
            _allocation_payload(resolved_dbp),
        )
        _write_jsonl(
            path / "dbp_perturbation_audit.jsonl",
            [
                {
                    "allocation_id": value.allocation_id,
                    "requested_case_id": dbp_selection_audit[value.allocation_id][
                        "requested_case_id"
                    ],
                    "selection_status": dbp_selection_audit[value.allocation_id][
                        "selection_status"
                    ],
                    "rejected_candidate_ids": list(
                        dbp_selection_audit[value.allocation_id][
                            "rejected_candidate_ids"
                        ]
                    ),
                    "family": value.family,
                    "strength": value.strength,
                    "applied_value": value.applied_value,
                    "language_pair": value.language_pair,
                    "source_id": value.source_id,
                    "target_id": value.target_id,
                    "changed_edges": [asdict(change) for change in value.changed_edges],
                    "changed_seed_alignments": [asdict(change) for change in value.changed_seed_alignments],
                    "valid": value.valid,
                    "validity_checks": list(value.validity_checks),
                }
                for value in dbp_rows.values()
            ],
        )

    perturbations_stage = run_stage(
        "perturbation_inputs",
        {"allocations": allocations_stage.content_hash},
        produce_perturbations,
    )
    execution_cases, execution_metadata = _load_execution_cases(
        perturbations_stage.path / "execution_cases.jsonl"
    )
    resolved_dbp_allocations = AllocationTable(
        tuple(
            AllocationRow(**row)
            for row in json.loads(
                (perturbations_stage.path / "resolved_dbp_allocations.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    )
    tuning_cases, tuning_split_by_id = _tuning_cases(
        datasets,
        execution_metadata,
        contract,
        split_fractions,
        seed=config.paper.seeds[0],
    )
    tuning_contract = ScaleContract(
        scale=scale,
        scicu_nonstress_per_regime=0,
        scicu_boundary=len(tuning_cases),
        dbp_nonstress_per_regime=0,
        dbp_boundary=0,
        generator_count=contract.generator_count,
        responses_per_task=contract.responses_per_task,
        seeds=contract.seeds,
    )
    evaluation_recovery_requests = _supplied_recovery_requests(
        evaluation_response_files,
        execution_cases,
        config,
    )
    tuning_recovery_requests = _supplied_recovery_requests(
        tuning_response_files,
        tuning_cases,
        config,
    )

    def produce_responses(path: Path) -> None:
        batch = acquire_formal_responses(
            execution_cases,
            config,
            contract,
            entropy_weight=formal_config.response_profile_entropy_weight,
            recovery_requests=evaluation_recovery_requests,
        )
        tuning_batch = acquire_formal_responses(
            tuning_cases,
            config,
            tuning_contract,
            entropy_weight=formal_config.response_profile_entropy_weight,
            recovery_requests=tuning_recovery_requests,
        )
        _write_parquet(
            path / "responses.parquet",
            [row.model_dump(mode="json") for row in batch.rows],
        )
        _write_parquet(
            path / "tuning_responses.parquet",
            [row.model_dump(mode="json") for row in tuning_batch.rows],
        )
        _write_json(
            path / "response_manifest.json",
            {
                "profiles": [
                    {
                        "model_id": profile.model_id,
                        "route": profile.route,
                        "parameters": profile.parameters,
                        "profile": profile.profile.model_dump(mode="json"),
                    }
                    for profile in batch.profiles
                ],
                "acquisitions": [row.model_dump(mode="json") for row in batch.acquisitions],
                "tuning_acquisitions": [
                    row.model_dump(mode="json")
                    for row in tuning_batch.acquisitions
                ],
                "target_derivation": dict(batch.target_derivation),
                "reference_labels": [asdict(row) for row in batch.reference_labels],
                "tuning_reference_labels": [
                    asdict(row) for row in tuning_batch.reference_labels
                ],
                "tuning_split_by_task": tuning_split_by_id,
            },
        )

    responses_stage = run_stage(
        "responses",
        {
            "perturbations": perturbations_stage.content_hash,
            "response_inputs": canonical_sha256(
                response_input_hashes
                | (
                    {"published_targets": profile_target_hash}
                    if profile_target_hash is not None
                    else {}
                )
            ),
        },
        produce_responses,
    )
    response_frame = pd.read_parquet(responses_stage.path / "responses.parquet")
    responses = tuple(
        ResponseRecord.model_validate(row)
        for row in response_frame.to_dict(orient="records")
    )
    tuning_response_frame = pd.read_parquet(
        responses_stage.path / "tuning_responses.parquet"
    )
    tuning_responses = tuple(
        ResponseRecord.model_validate(row)
        for row in tuning_response_frame.to_dict(orient="records")
    )
    response_manifest = json.loads(
        (responses_stage.path / "response_manifest.json").read_text(encoding="utf-8")
    )
    reference_label_by_key = {
        (row["task_id"], row["model_id"], int(row["replicate"])): ResponseReferenceLabel(
            task_id=row["task_id"],
            model_id=row["model_id"],
            replicate=int(row["replicate"]),
            endpoint_success=bool(row["endpoint_success"]),
            structural_validity=bool(row["structural_validity"]),
            failure_group=row["failure_group"],
            applied_transforms=tuple(row["applied_transforms"]),
            source=row["source"],
        )
        for row in response_manifest["reference_labels"]
    }
    tuning_reference_label_by_key = {
        (row["task_id"], row["model_id"], int(row["replicate"])): ResponseReferenceLabel(
            task_id=row["task_id"],
            model_id=row["model_id"],
            replicate=int(row["replicate"]),
            endpoint_success=bool(row["endpoint_success"]),
            structural_validity=bool(row["structural_validity"]),
            failure_group=row["failure_group"],
            applied_transforms=tuple(row["applied_transforms"]),
            source=row["source"],
        )
        for row in response_manifest["tuning_reference_labels"]
    }

    base_context = build_formal_context(
        config,
        training_texts=_training_texts(
            datasets,
            execution_metadata,
            split_fractions,
        ),
    )
    case_by_id = {case.task_id: case for case in execution_cases}
    tuning_case_by_id = {case.task_id: case for case in tuning_cases}

    def produce_method_selection(path: Path) -> None:
        examples = tuple(
            MethodSelectionExample(
                case=tuning_case_by_id[response.task_id],
                response=response,
                structural_validity=tuning_reference_label_by_key[
                    (response.task_id, response.model_id, response.replicate)
                ].structural_validity,
            )
            for response in tuning_responses
            if tuning_split_by_id[response.task_id] == "validation"
        )
        selected = select_method_parameters(
            examples,
            base_context,
            config,
            seed=config.paper.seeds[0],
        )
        _write_json(
            path / "method_selection.json",
            {
                "selection_split": "validation",
                "selection_criterion": "minimum structural-validity Brier score; ties use feasible rate then trial order",
                "selected_trial": selected.selected_trial,
                "trial_count": selected.trial_count,
                "selected_parameters": asdict(selected.parameters),
                "loss_normalizers": selected.loss_normalizers,
                "residual_normalizers": {
                    channel.value: value
                    for channel, value in selected.residual_normalizers.items()
                },
                "normalizer_statistic": {
                    "quantile": config.residuals.normalizer_percentile,
                    "floor": config.residuals.normalizer_floor,
                    "split": "validation",
                },
                "loss_normalizer_statistic": {
                    "quantile": config.transport.loss_normalizer_percentile,
                    "floor": config.transport.loss_normalizer_floor,
                    "split": "validation",
                    "graph_size_normalization": "unit transport marginals",
                },
                "text_models": base_context.text_models.audit_manifest(),
                "trials": [
                    {
                        "trial_index": trial.trial_index,
                        "parameters": asdict(trial.parameters),
                        "loss_normalizers": trial.loss_normalizers,
                        "validation_brier": trial.validation_brier,
                        "feasible_rate": trial.feasible_rate,
                    }
                    for trial in selected.trials
                ],
            },
        )

    method_selection_stage = run_stage(
        "method_selection",
        {"responses": responses_stage.content_hash},
        produce_method_selection,
    )
    method_selection = json.loads(
        (method_selection_stage.path / "method_selection.json").read_text(encoding="utf-8")
    )
    context = replace(
        base_context,
        parameters=MethodParameters(**method_selection["selected_parameters"]),
        loss_normalizers={
            str(component): float(value)
            for component, value in method_selection["loss_normalizers"].items()
        },
        residual_normalizers={
            ResidualChannel(channel): float(value)
            for channel, value in method_selection["residual_normalizers"].items()
        },
    )

    def produce_evaluation(path: Path) -> None:
        rows = []
        task_graph_records: dict[str, dict[str, object]] = {}
        response_graph_records: list[dict[str, object]] = []
        validator_rows: list[dict[str, object]] = []
        transport_rows: list[dict[str, object]] = []
        residual_rows: list[dict[str, object]] = []
        component_links: list[dict[str, object]] = []
        transport_plans: dict[str, np.ndarray] = {}
        for response in responses:
            case = case_by_id[response.task_id]
            result, components = evaluate_scicu_response_with_components(
                case=case,
                response=response,
                context=context,
            )
            ccs = _cross_context_consistency(
                case=case,
                response=response,
                context=context,
                reference_pcs=result.pcs,
            )
            payload = _evaluation_payload(
                result,
                execution_metadata[response.task_id],
                ccs,
                reference_label_by_key[
                    (response.task_id, response.model_id, response.replicate)
                ],
            )
            rows.append(payload)
            serialized = serialize_evaluated_components(
                components,
                model_id=response.model_id,
                replicate=response.replicate,
            )
            task_graph_id = str(serialized.task_graph_record["graph_id"])
            existing_task_graph = task_graph_records.get(task_graph_id)
            if existing_task_graph is not None and existing_task_graph != serialized.task_graph_record:
                raise ValueError("task graph changed across responses for the same task")
            task_graph_records[task_graph_id] = serialized.task_graph_record
            if serialized.plan_key in transport_plans:
                raise ValueError("duplicate transport plan key")
            transport_plans[serialized.plan_key] = serialized.plan
            response_graph_records.append(serialized.response_graph_record)
            validator_rows.extend(serialized.validator_rows)
            transport_rows.append(serialized.transport_row)
            residual_rows.extend(serialized.residual_rows)
            component_links.append(
                serialized.link_row
                | {
                    "allocation_id": payload["allocation_id"],
                    "original_task_id": payload["original_task_id"],
                    "regime": payload["regime"],
                    "family": payload["family"] or "not_applicable",
                    "strength": float(payload["strength"] or 0.0),
                    "baseline_case_id": serialized.instance_key,
                    "ablation_case_id": (
                        f"{response.task_id}\t{response.model_id}\t{response.replicate}"
                    ),
                }
            )
        _write_parquet(path / "per_case_metrics.parquet", rows)
        _write_jsonl(
            path / "task_graphs.jsonl",
            [task_graph_records[key] for key in sorted(task_graph_records)],
        )
        _write_jsonl(path / "response_graphs.jsonl", response_graph_records)
        _write_parquet(path / "validator_outputs.parquet", validator_rows)
        _write_parquet(path / "transport_diagnostics.parquet", transport_rows)
        _write_parquet(path / "residual_components.parquet", residual_rows)
        _write_parquet(path / "instance_component_links.parquet", component_links)
        np.savez_compressed(path / "transport_plans.npz", **transport_plans)

        tuning_rows = []
        for response in tuning_responses:
            case = tuning_case_by_id[response.task_id]
            result = evaluate_scicu_response(
                case=case,
                response=response,
                context=context,
            )
            ccs = _cross_context_consistency(
                case=case,
                response=response,
                context=context,
                reference_pcs=result.pcs,
            )
            split = tuning_split_by_id[response.task_id]
            payload = _evaluation_payload(
                result,
                {
                    "original_task_id": response.task_id,
                    "allocation": {
                        "allocation_id": f"tuning::{split}::{response.task_id}",
                        "regime": f"tuning_{split}",
                        "family": None,
                        "strength": None,
                    },
                },
                ccs,
                tuning_reference_label_by_key[
                    (response.task_id, response.model_id, response.replicate)
                ],
            )
            tuning_rows.append(payload | {"split": split})
        _write_parquet(path / "tuning_metrics.parquet", tuning_rows)

        original_case_by_id = {case.task_id: case for case in datasets.scicueval}
        profiles = {
            row["model_id"]: ReconstructionProfile.model_validate(row["profile"])
            for row in response_manifest["profiles"]
        }
        originals = []
        boundary_groups = {
            (
                str(meta["original_task_id"]),
                str(dict(meta["allocation"])["bsi_regime"]),
                str(dict(meta["allocation"])["family"]),
            )
            for meta in execution_metadata.values()
            if dict(meta["allocation"])["regime"] == "boundary_stress"
        }
        boundary_seed_task = {
            (
                str(meta["original_task_id"]),
                str(dict(meta["allocation"])["bsi_regime"]),
                str(dict(meta["allocation"])["family"]),
            ): task_id
            for task_id, meta in execution_metadata.items()
            if dict(meta["allocation"])["regime"] == "boundary_stress"
        }
        for original_id, bsi_regime, family in sorted(boundary_groups):
            case = original_case_by_id[original_id]
            for model_id, profile in profiles.items():
                for replicate in range(contract.responses_per_task):
                    seed_task = boundary_seed_task[
                        (original_id, bsi_regime, family)
                    ]
                    reconstructed = reconstruct_response_with_audit(
                        case,
                        profile,
                        seed=response_seed(config.paper.seeds[0], seed_task, model_id, replicate),
                        model_id=model_id,
                        replicate=replicate,
                    )
                    response = reconstructed.record
                    result = evaluate_scicu_response(case=case, response=response, context=context)
                    ccs = _cross_context_consistency(
                        case=case,
                        response=response,
                        context=context,
                        reference_pcs=result.pcs,
                    )
                    payload = _evaluation_payload(
                        result,
                        {
                            "original_task_id": original_id,
                            "allocation": {
                                "allocation_id": f"original::{family}::{original_id}",
                                "regime": "boundary_original",
                                "bsi_regime": bsi_regime,
                                "family": family,
                                "strength": 0.0,
                            },
                        },
                        ccs,
                        reference_label_from_audit(
                            response,
                            reconstructed.audit.applied_transforms,
                        ),
                    )
                    originals.append(payload | {"bsi_regime": bsi_regime})
        _write_parquet(path / "boundary_original_metrics.parquet", originals)

    evaluation_stage = run_stage(
        "evaluation",
        {
            "responses": responses_stage.content_hash,
            "method_selection": method_selection_stage.content_hash,
        },
        produce_evaluation,
    )
    metric_frame = pd.read_parquet(evaluation_stage.path / "per_case_metrics.parquet")
    tuning_metric_frame = pd.read_parquet(
        evaluation_stage.path / "tuning_metrics.parquet"
    )
    original_metric_frame = pd.read_parquet(
        evaluation_stage.path / "boundary_original_metrics.parquet"
    )

    def produce_state(path: Path) -> None:
        records = metric_frame.to_dict(orient="records")
        formal_rows = tuple(_formal_row(row) for row in records)
        tuning_records = tuning_metric_frame.to_dict(orient="records")
        tuning_rows = tuple(_formal_row(row) for row in tuning_records)
        train = tuple(
            row
            for row, record in zip(tuning_rows, tuning_records, strict=True)
            if record["split"] == "train"
        )
        validation = tuple(
            row
            for row, record in zip(tuning_rows, tuning_records, strict=True)
            if record["split"] == "validation"
        )
        if len(train) < 3 or len(validation) < 3:
            raise ValueError("formal state training split is too small")
        bundle = train_state_models(
            train,
            validation,
            config,
            formal_config=formal_config,
        )
        seedwise, energy = _infer_models(bundle.models, formal_rows, config)
        assignments = []
        for index, record in enumerate(records):
            item = {
                "task_id": record["task_id"],
                "model_id": record["model_id"],
                "replicate": record["replicate"],
                "mean_energy": energy[index],
            }
            item.update(
                {f"regime_seed_{seed}": seedwise[position][index] for position, seed in enumerate(config.paper.seeds)}
            )
            assignments.append(item)
        _write_parquet(path / "state_assignments.parquet", assignments)

        original_records = original_metric_frame.to_dict(orient="records")
        original_rows = tuple(_formal_row(row) for row in original_records)
        original_seedwise, original_energy = _infer_models(
            bundle.models,
            original_rows,
            config,
        )
        original_assignments = []
        for index, record in enumerate(original_records):
            item = {
                "task_id": record["task_id"],
                "original_task_id": record["original_task_id"],
                "bsi_regime": record["bsi_regime"],
                "family": record["family"],
                "model_id": record["model_id"],
                "replicate": record["replicate"],
                "mean_energy": original_energy[index],
            }
            item.update(
                {f"regime_seed_{seed}": original_seedwise[position][index] for position, seed in enumerate(config.paper.seeds)}
            )
            original_assignments.append(item)
        _write_parquet(path / "boundary_original_assignments.parquet", original_assignments)
        torch.save(
            {str(model.seed): model.state_dict for model in bundle.models},
            path / "state_models.pt",
        )
        _write_json(
            path / "state_training_manifest.json",
            {
                "thresholds": asdict(bundle.thresholds),
                "label_policy": {
                    "primary": (
                        "validation-fitted PCS and residual thresholds with "
                        "hard-feasibility override"
                    ),
                    "fallback": (
                        "deterministic ranked thirds by PCS minus mean residual "
                        "when a split does not contain all three regimes"
                    ),
                    "labels": {"stable": 0, "brittle": 1, "invalid": 2},
                    "training_counts": {
                        str(label): int(np.sum(bundle.training_labels == label))
                        for label in (0, 1, 2)
                    },
                    "validation_counts": {
                        str(label): int(np.sum(bundle.validation_labels == label))
                        for label in (0, 1, 2)
                    },
                },
                "architecture": {
                    "g_theta_dimensions": list(config.state_field.dimensions),
                    "regime_head_dimensions": [
                        config.state_field.dimensions[-1],
                        3,
                    ],
                    "e_phi_dimensions": list(config.state_field.energy_dimensions),
                    "activation": config.state_field.activation,
                    "gelu_approximate": config.state_field.gelu_approximate,
                    "linear_bias": config.state_field.linear_bias,
                    "initialization": config.state_field.initialization,
                    "dropout": config.state_field.dropout,
                    "dtype": "float64",
                },
                "models": [
                    {
                        "seed": model.seed,
                        "selected_trial": model.selected_trial,
                        "trial_count": model.trial_count,
                        "learning_rate": model.learning_rate,
                        "weight_decay": model.weight_decay,
                        "energy_loss_weight": model.energy_loss_weight,
                        "energy_margin": model.energy_margin,
                        "selected_epoch": model.selected_epoch,
                        "validation_loss": model.validation_loss,
                        "validation_accuracy": model.validation_accuracy,
                        "state_dict_hash": model.state_dict_hash,
                    }
                    for model in bundle.models
                ],
            },
        )

    state_stage = run_stage(
        "state_training",
        {"evaluation": evaluation_stage.content_hash},
        produce_state,
    )
    assignments = pd.read_parquet(state_stage.path / "state_assignments.parquet")
    original_assignments = pd.read_parquet(
        state_stage.path / "boundary_original_assignments.parquet"
    )

    def produce_boundaries(path: Path) -> None:
        merged = metric_frame.merge(assignments, on=["task_id", "model_id", "replicate"])
        original_lookup = {
            (
                row.original_task_id,
                row.bsi_regime,
                row.family,
                row.model_id,
                int(row.replicate),
                seed,
            ): int(getattr(row, f"regime_seed_{seed}"))
            for row in original_assignments.itertuples(index=False)
            for seed in config.paper.seeds
        }
        evaluations_by_regime: dict[str, list[BoundaryEvaluation]] = {}
        for row in merged[merged["regime"] == "boundary_stress"].itertuples(index=False):
            bsi_regime = str(row.bsi_regime)
            for seed in config.paper.seeds:
                evaluations_by_regime.setdefault(bsi_regime, []).append(
                    BoundaryEvaluation(
                        case_id=f"{row.original_task_id}\t{row.model_id}\t{row.replicate}",
                        family=str(row.family),
                        seed=seed,
                        radius=float(row.strength),
                        original_regime=original_lookup[
                            (
                                row.original_task_id,
                                row.bsi_regime,
                                row.family,
                                row.model_id,
                                int(row.replicate),
                                seed,
                            )
                        ],
                        perturbed_regime=int(getattr(row, f"regime_seed_{seed}")),
                        valid=bool(execution_metadata[str(row.task_id)]["valid"]),
                    )
                )
        boundary_rows = [
            asdict(row) | {"regime": bsi_regime, "source": "computed"}
            for bsi_regime, evaluations in sorted(evaluations_by_regime.items())
            for row in compute_boundary_rows(
                tuple(evaluations),
                rho_max=max(config.boundary.standardized_radii),
            )
        ]
        _write_parquet(
            path / "boundaries.parquet",
            boundary_rows,
        )

    boundaries_stage = run_stage(
        "boundaries",
        {"state_training": state_stage.content_hash},
        produce_boundaries,
    )
    boundary_frame = pd.read_parquet(boundaries_stage.path / "boundaries.parquet")

    def produce_open_set(path: Path) -> None:
        validation, unknown = _encode_open_set_groups(
            tuning_metric_frame[tuning_metric_frame["split"] == "validation"],
            metric_frame,
        )
        if len(validation) < 2:
            raise ValueError("open-set validation/test split is too small")
        motif_center = np.median(
            np.stack(
                validation["residuals"].map(
                    lambda value: np.asarray(value, dtype=np.float64)
                )
            ),
            axis=0,
        )
        validation_bundle = _open_bundle(validation, motif_center=motif_center)
        rows = []
        candidate_groups = (
            tuple(
                (regime, unknown[unknown["regime"] == regime].copy())
                for regime in config.regimes.names
            )
            if scale == "paper"
            else (("pooled_smoke", unknown.copy()),)
        )
        for regime, candidates in candidate_groups:
            if len(candidates) < 3:
                raise ValueError(
                    f"open-set test split is too small for regime {regime}: "
                    f"{len(candidates)}"
                )
            result = run_open_set_analysis(
                validation_bundle,
                _open_bundle(candidates, motif_center=motif_center),
                config,
            )
            rows.append(
                asdict(result)
                | {
                    "regime": regime,
                    "candidate_count": len(candidates),
                    "support_split": "independent_validation",
                    "candidate_split": "formal_final_test",
                    "source": "computed",
                }
            )
        _write_parquet(path / "open_set.parquet", rows)

    open_stage = run_stage(
        "open_set",
        {"state_training": state_stage.content_hash},
        produce_open_set,
    )
    open_frame = pd.read_parquet(open_stage.path / "open_set.parquet")

    def materialized_dbp() -> dict[str, MaterializedDBPPerturbation]:
        values = {}
        for allocation in resolved_dbp_allocations.rows:
            if allocation.regime != "boundary_stress":
                continue
            assert allocation.language_pair is not None
            values[allocation.allocation_id] = materialize_dbp_perturbation(
                datasets.dbp15k.language_pairs[allocation.language_pair],
                allocation,
                seed=config.paper.seeds[0],
                standardized_radii=config.boundary.standardized_radii,
                removal_ratios=config.boundary.dbp_removal_ratios,
                seed_fractions=config.boundary.dbp_seed_fractions,
            )
        return values

    def produce_dbp(path: Path) -> None:
        results = run_dbp_experiment(
            datasets,
            resolved_dbp_allocations,
            config,
            perturbations=materialized_dbp(),
            parameters=context.parameters,
        )
        _write_parquet(
            path / "dbp_metrics.parquet",
            [asdict(row) | {"source": "computed"} for row in results],
        )

    dbp_stage = run_stage(
        "dbp15k",
        {
            "perturbations": perturbations_stage.content_hash,
            "method_selection": method_selection_stage.content_hash,
        },
        produce_dbp,
    )
    dbp_frame = pd.read_parquet(dbp_stage.path / "dbp_metrics.parquet")
    dbp_rows = tuple(DBPCaseResult(**{key: row[key] for key in DBPCaseResult.__dataclass_fields__}) for row in dbp_frame.to_dict(orient="records"))

    def produce_baselines(path: Path) -> None:
        records = metric_frame.to_dict(orient="records")
        tuning_records = tuning_metric_frame.to_dict(orient="records")
        response_by_key = {(row.task_id, row.model_id, row.replicate): row for row in responses}
        tuning_response_by_key = {
            (row.task_id, row.model_id, row.replicate): row
            for row in tuning_responses
        }
        validation_support = np.stack(
            [
                _baseline_representation(
                    tuning_case_by_id[str(record["task_id"])],
                    tuning_response_by_key[
                        (
                            str(record["task_id"]),
                            str(record["model_id"]),
                            int(record["replicate"]),
                        )
                    ],
                    config.baseline_reconstruction.representation_dimensions,
                )
                for record in tuning_records
                if record["split"] == "validation"
            ]
        )
        scicu_baseline_cases = tuple(
            _baseline_case_scicu(
                record,
                case_by_id[str(record["task_id"])],
                response_by_key[(str(record["task_id"]), str(record["model_id"]), int(record["replicate"]))],
                validation_support,
                representation_dimensions=(
                    config.baseline_reconstruction.representation_dimensions
                ),
            )
            for record in records
        )
        tuning_baseline_cases = tuple(
            _baseline_case_scicu(
                record,
                tuning_case_by_id[str(record["task_id"])],
                tuning_response_by_key[
                    (
                        str(record["task_id"]),
                        str(record["model_id"]),
                        int(record["replicate"]),
                    )
                ],
                validation_support,
                representation_dimensions=(
                    config.baseline_reconstruction.representation_dimensions
                ),
            )
            for record in tuning_records
            if record["split"] == "validation"
        )
        scicu_structural_validity = {
            (
                f"{record['task_id']}::{record['model_id']}::"
                f"{int(record['replicate'])}"
            ): bool(record["reference_structural_validity"])
            for record in tuning_records
            if record["split"] == "validation"
        }
        dbp_validation_cases = _dbp_baseline_validation_cases(
            datasets,
            config,
            seed=config.paper.seeds[0],
            validation_fraction=split_fractions[1],
        )
        baseline_selection = select_formal_baseline_parameters(
            scicu_validation_cases=tuning_baseline_cases,
            scicu_structural_validity=scicu_structural_validity,
            dbp_validation_cases=dbp_validation_cases,
            config=config,
            seed=config.paper.seeds[0],
            pass_threshold=formal_config.proxy_pass_threshold,
            fail_threshold=formal_config.proxy_fail_threshold,
        )
        selected_parameters = {
            kind: dict(parameters)
            for kind, parameters in baseline_selection.parameters.items()
        }
        expected_tunable_baselines = {
            BaselineKind.UNCERTAINTY.value,
            BaselineKind.KNN_SUPPORT.value,
            BaselineKind.CENTROID_SUPPORT.value,
            BaselineKind.GRAPH_EDIT.value,
            BaselineKind.STRUCTURE_AWARE.value,
            BaselineKind.SINKHORN.value,
            BaselineKind.GROMOV.value,
        }
        if set(selected_parameters) != expected_tunable_baselines:
            raise ValueError("baseline selection did not resolve every tunable method")
        sinkhorn_epsilon = float(
            selected_parameters[BaselineKind.SINKHORN.value]["epsilon"]
        )
        final_dbp_baseline_cases = build_dbp_baseline_cases(
            datasets,
            resolved_dbp_allocations,
            config,
            perturbations=materialized_dbp(),
            sinkhorn_epsilon=sinkhorn_epsilon,
        )
        cases = (
            scicu_baseline_cases
            + tuning_baseline_cases
            + final_dbp_baseline_cases
        )
        registry = build_formal_baseline_registry(
            api_key=judge_response_hash,
            requester=judge_requester,
            pass_threshold=formal_config.proxy_pass_threshold,
            fail_threshold=formal_config.proxy_fail_threshold,
            judge_config=config.judge,
            run_config=config,
            selected_parameters=selected_parameters,
        )
        baseline_rows = list(run_formal_baselines(cases, registry))
        _write_json(
            path / "baseline_selection.json",
            {
                "selection_split": "independent_validation",
                "trial_count_per_tunable_baseline": config.paper.hyperparameter_trials,
                "selected_parameters": selected_parameters,
                "reconstruction_configuration": (
                    config.baseline_reconstruction.model_dump(mode="json")
                ),
                "dbp_reconstruction_configuration": (
                    config.dbp_reconstruction.model_dump(mode="json")
                ),
                "objectives": dict(baseline_selection.objectives),
                "trials": {
                    kind: {
                        "selected_parameters": dict(trial.parameters),
                        "validation_score": trial.validation_score,
                        "trial_count": trial.trial_count,
                        "seed": trial.seed,
                    }
                    for kind, trial in baseline_selection.trials.items()
                },
                "deterministic_single_run": [
                    BaselineKind.ENDPOINT.value,
                    BaselineKind.ATTRIBUTION.value,
                    BaselineKind.FIXED_TAXONOMY.value,
                    BaselineKind.RELATION_AWARE.value,
                    BaselineKind.HITS_AT_1.value,
                    BaselineKind.MRR.value,
                ],
                "llm_judge": {
                    "model": config.judge.model_id,
                    "temperature": config.judge.temperature,
                    "top_p": config.judge.top_p,
                    "max_output_tokens": config.judge.max_output_tokens,
                    "route": (
                        "exact_gpt4o_response_replay"
                        if judge_requester is not None
                        else "local_inferred_when_exact_responses_are_not_supplied"
                    ),
                    "source_sha256": judge_response_hash,
                    "required_fields": PAPER_JUDGE_FIELDS,
                },
                "dbp_validation_protocol": {
                    "source": "deterministic_holdout_from_official_30_percent_seed_pool",
                    "holdout_fraction": split_fractions[1],
                    "case_count": len(dbp_validation_cases),
                    "full_target_candidate_pool": True,
                    "final_reference_pairs_used_for_selection": False,
                },
                "source": "computed",
            },
        )

        original_case_by_id = {case.task_id: case for case in datasets.scicueval}
        profiles = {
            row["model_id"]: ReconstructionProfile.model_validate(row["profile"])
            for row in response_manifest["profiles"]
        }
        boundary_seed_task = {
            (
                str(meta["original_task_id"]),
                str(dict(meta["allocation"])["bsi_regime"]),
                str(dict(meta["allocation"])["family"]),
            ): task_id
            for task_id, meta in execution_metadata.items()
            if dict(meta["allocation"])["regime"] == "boundary_stress"
        }
        original_baseline_cases = []
        for record in original_metric_frame.to_dict(orient="records"):
            original_id = str(record["original_task_id"])
            bsi_regime = str(record["bsi_regime"])
            family = str(record["family"])
            model_id = str(record["model_id"])
            replicate = int(record["replicate"])
            case = original_case_by_id[original_id]
            reconstructed = reconstruct_response_with_audit(
                case,
                profiles[model_id],
                seed=response_seed(
                    config.paper.seeds[0],
                    boundary_seed_task[(original_id, bsi_regime, family)],
                    model_id,
                    replicate,
                ),
                model_id=model_id,
                replicate=replicate,
            )
            original_baseline_cases.append(
                _baseline_case_scicu(
                    record,
                    case,
                    reconstructed.record,
                    validation_support,
                    representation_dimensions=(
                        config.baseline_reconstruction.representation_dimensions
                    ),
                    case_id=(
                        f"baseline-original\t{original_id}\t{bsi_regime}\t{family}\t"
                        f"{model_id}\t{replicate}"
                    ),
                )
            )
        baseline_rows.extend(run_formal_baselines(tuple(original_baseline_cases), registry))
        baseline_payload = [
            asdict(row) | {"source": "computed"} for row in baseline_rows
        ]
        _write_parquet(path / "baselines.parquet", baseline_payload)
        baseline_data = pd.DataFrame(baseline_payload)
        _write_parquet(
            path / "baseline_boundaries.parquet",
            _build_baseline_boundary_rows(
                metric_frame,
                tuning_metric_frame,
                baseline_data,
                config,
                formal_config,
            ),
        )
        _write_parquet(
            path / "baseline_open_set.parquet",
            _build_baseline_open_set_rows(
                metric_frame,
                tuning_metric_frame,
                baseline_data,
                config,
                scale=scale,
            ),
        )

    baseline_stage = run_stage(
        "baselines",
        {
            "evaluation": evaluation_stage.content_hash,
            "dbp15k": dbp_stage.content_hash,
            "llm_judge": canonical_sha256(
                {
                    "route": (
                        "exact_gpt4o_response_replay"
                        if judge_response_hash is not None
                        else "local_inferred"
                    ),
                    "source_sha256": judge_response_hash,
                }
            ),
        },
        produce_baselines,
    )
    baseline_frame = pd.read_parquet(baseline_stage.path / "baselines.parquet")
    baseline_boundary_frame = pd.read_parquet(
        baseline_stage.path / "baseline_boundaries.parquet"
    )
    baseline_open_frame = pd.read_parquet(
        baseline_stage.path / "baseline_open_set.parquet"
    )

    original_bsi = {
        original_id: float(group["bsi_raw"].mean())
        for original_id, group in boundary_frame.assign(
            original_id=boundary_frame["case_id"].map(
                lambda value: str(value).split("\t", 1)[0]
            )
        ).groupby("original_id")
    }
    bsi_by_task = {
        task_id: original_bsi.get(str(meta["original_task_id"]), 0.0)
        for task_id, meta in execution_metadata.items()
    }
    ablation_bundle = _AblationBundle(
        scicu_cases=execution_cases,
        responses=responses,
        context=context,
        main_rows=tuple(metric_frame.to_dict(orient="records")),
        dbp_rows=dbp_rows,
        bsi_by_task=bsi_by_task,
        updq=float(open_frame["updq"].mean()),
        sc=float(open_frame["structural_calibration"].mean()),
    )

    def evaluator_factory(kind: AblationKind | None = None):
        controls = AblationControls() if kind is None else resolve_ablation_policy(kind).controls
        return _AblationEvaluator(controls)

    def produce_ablations(path: Path) -> None:
        rows = run_formal_ablations(ablation_bundle, evaluator_factory)
        _write_parquet(path / "ablations.parquet", [asdict(row) for row in rows])

    ablation_stage = run_stage(
        "ablations",
        {"evaluation": evaluation_stage.content_hash, "boundaries": boundaries_stage.content_hash, "open_set": open_stage.content_hash, "dbp15k": dbp_stage.content_hash},
        produce_ablations,
    )
    ablation_frame = pd.read_parquet(ablation_stage.path / "ablations.parquet")

    published_name = {
        "without_motif_structure": "without_motif_graph",
        "without_residual_channels": "without_residual_module",
        "without_boundary_estimation": "without_boundary_module",
        "without_open_set": "without_open_set_module",
    }

    def produce_statistics(path: Path) -> None:
        ablation_pairs = tuple(
            ComputedPairRow(
                pair_id=str(row.case_id).split("\t", 1)[0],
                dataset=row.dataset,
                regime="all",
                metric=row.metric,
                proposed_method="sciprism_x",
                comparison_method=row.ablation,
                proposed_value=float(row.main_value),
                comparison_value=float(row.metric_value),
                direction="lower" if row.metric == "BSI" else "higher",
                family="appendix_s14_s15",
                reference_path=(
                    "ablation_deltas",
                    published_name.get(row.ablation, row.ablation),
                    row.metric,
                ),
            )
            for row in ablation_frame.itertuples(index=False)
        )
        pairs = _build_main_statistic_pairs(
            metric_frame,
            baseline_frame,
            boundary_frame,
            baseline_boundary_frame,
            open_frame,
            baseline_open_frame,
            dbp_frame,
            baseline_task_success_threshold=(
                config.baseline_reconstruction.task_success_threshold
            ),
        ) + ablation_pairs
        result = compute_formal_statistics(pairs, config)
        _write_parquet(path / "bootstrap_results.parquet", [asdict(row) for row in result.bootstrap_rows])
        _write_parquet(path / "significance_results.parquet", [asdict(row) for row in result.significance_rows])

    statistics_stage = run_stage(
        "statistics",
        {
            "evaluation": evaluation_stage.content_hash,
            "dbp15k": dbp_stage.content_hash,
            "baselines": baseline_stage.content_hash,
            "ablations": ablation_stage.content_hash,
        },
        produce_statistics,
    )

    def produce_comparison(path: Path) -> None:
        bootstrap = pd.read_parquet(statistics_stage.path / "bootstrap_results.parquet")
        significance = pd.read_parquet(statistics_stage.path / "significance_results.parquet")
        result = _statistics_result(bootstrap, significance)
        discrepancies = build_discrepancy_report(result, paper_reference)
        shutil.copy2(paper_reference, path / "published_metrics.json")
        with (path / "discrepancy_report.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(asdict(discrepancies[0])))
            writer.writeheader()
            writer.writerows(asdict(row) for row in discrepancies)

    comparison_stage = run_stage(
        "paper_comparison",
        {
            "statistics": statistics_stage.content_hash,
            "paper_reference": _sha256(paper_reference),
        },
        produce_comparison,
    )

    def produce_plotting(path: Path) -> None:
        aggregate = _aggregate_metrics(metric_frame, boundary_frame, open_frame, dbp_frame)
        _write_json(path / "aggregate_metrics.json", aggregate)
        source_summary = json.loads(
            (sources.path / "source_summary.json").read_text(encoding="utf-8")
        )
        state_training_manifest = json.loads(
            (state_stage.path / "state_training_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        dbp_perturbation_audit = _read_jsonl(
            perturbations_stage.path / "dbp_perturbation_audit.jsonl"
        )
        _write_json(
            path / "implementation_contract.json",
            {
                "external_assets": {
                    "provided_in_inputs": bool(
                        config.track == "A" or judge_response_hash is not None
                    ),
                    "method_sources": [
                        "Manuscript.docx",
                        "Appendix.docx",
                        "pinned_official_SciCUEval_sources",
                        "pinned_official_DBP15K_JAPE_archive",
                    ],
                    "source_summary": source_summary,
                    "external_inputs": {
                        "dbp15k_original_split": {
                            "status": "recovered_from_official_0_3_files",
                            "policy": "reuse_unchanged_with_identical_entity_ids",
                            "candidate_pruning": False,
                        },
                        "scicueval_train_validation_split": {
                            "status": "external_manifest_not_supplied",
                            "replacement": (
                                "sha256_task_bucket_over_cases_excluded_from_all_"
                                "formal_final_allocations"
                            ),
                            "fractions": split_fractions,
                            "resolved_mapping": "final_allocations.tuning_split_by_task",
                        },
                        "supplied_model_responses": {
                            "status": (
                                "supplied_recovered_files"
                                if config.track == "A"
                                else "external_files_not_supplied"
                            ),
                            "replacement": (
                                "frozen_response_manifest_files"
                                if config.track == "A"
                                else "deterministic_profile_constrained_generation"
                            ),
                            "acquisition_records": (
                                "response_reconstruction.acquisitions"
                            ),
                        },
                        "baseline_configuration": {
                            "status": "external_configuration_not_supplied",
                            "replacement": "independent_validation_50_trial_selection",
                            "selection_artifact": "computed/baseline_selection.json",
                            "baseline_reconstruction": (
                                config.baseline_reconstruction.model_dump(mode="json")
                            ),
                            "dbp_reconstruction": (
                                config.dbp_reconstruction.model_dump(mode="json")
                            ),
                        },
                        "dbp15k_prior_system_outputs": {
                            "status": (
                                "not_supplied_use_configured_adapter"
                            ),
                            "replacement": (
                                "concept_compatible_deterministic_adapter"
                            ),
                            "scope": (
                                "auxiliary_ranking_and_structural_matching_comparators"
                            ),
                        },
                        "llm_judge_responses": {
                            "status": (
                                "frozen_response_replay_supplied"
                                if judge_response_hash is not None
                                else "external_response_dump_not_supplied"
                            ),
                            "replacement": (
                                "exact_six_field_replay"
                                if judge_response_hash is not None
                                else "configured_local_five_component_inference"
                            ),
                            "source_sha256": judge_response_hash,
                            "required_fields": PAPER_JUDGE_FIELDS,
                        },
                    },
                },
                "resolved_parameters": [
                    {
                        "path": record.path,
                        "value": record.value,
                        "provenance": record.provenance.value,
                    }
                    for record in config.parameter_records()
                ],
                "formal_execution_parameters": {
                    "values": formal_config.model_dump(mode="json"),
                    "provenance": "IMPLEMENTATION_SETTING",
                },
                "final_allocations": {
                    "scicueval": [
                        dict(meta["allocation"])
                        | {
                            "requested_case_id": meta["requested_case_id"],
                            "resolved_case_id": meta["original_task_id"],
                            "selection_status": meta["selection_status"],
                            "rejected_candidate_ids": meta[
                                "rejected_candidate_ids"
                            ],
                            "valid": meta["valid"],
                            "validity_checks": meta["validity_checks"],
                        }
                        for meta in execution_metadata.values()
                    ],
                    "dbp15k": _allocation_payload(resolved_dbp_allocations),
                    "dbp15k_perturbation_audit": dbp_perturbation_audit,
                    "split_fractions": split_fractions,
                    "tuning_split_by_task": response_manifest[
                        "tuning_split_by_task"
                    ],
                },
                "response_reconstruction": {
                    "profiles": response_manifest["profiles"],
                    "acquisitions": response_manifest["acquisitions"],
                    "tuning_acquisitions": response_manifest[
                        "tuning_acquisitions"
                    ],
                    "target_derivation": response_manifest["target_derivation"],
                    "label_policy": {
                        "generated_rows": "deterministic_transform_audit",
                        "recovered_rows": (
                            "exact_reference_match_without_supplied_"
                            "adjudication_records"
                        ),
                    },
                },
                "state_training": state_training_manifest,
                "baseline_selection_artifact": "computed/baseline_selection.json",
                "paper_reference_isolation": {
                    "track": config.track,
                    "published_targets_namespace": "paper_reference",
                    "direct_metric_or_statistic_injection": False,
                    "inverse_response_profile_constraint": (
                        "Appendix worked-case targets identify the missing pooled response "
                        "profile before response generation"
                        if config.track == "B"
                        else "disabled; supplied response files are frozen before the "
                        "paper reference enters the post-computation comparison stage"
                    ),
                },
                "source": "computed",
            },
        )
        _write_json(
            path / "plotting_input_manifest.json",
            {
                "computed": [
                    "responses.parquet", "tuning_responses.parquet",
                    "method_selection.json", "per_case_metrics.parquet",
                    "tuning_metrics.parquet", "baseline_selection.json",
                    "baselines.parquet",
                    "baseline_boundaries.parquet", "baseline_open_set.parquet",
                    "ablations.parquet", "boundaries.parquet", "open_set.parquet",
                    "bootstrap_results.parquet", "significance_results.parquet",
                    "aggregate_metrics.json", "implementation_contract.json",
                    "sample_manifest.parquet", "task_graphs.jsonl",
                    "response_graphs.jsonl", "validator_outputs.parquet",
                    "transport_plans.npz", "transport_diagnostics.parquet",
                    "residual_components.parquet", "per_instance_outputs.parquet",
                    "state_assignments.parquet", "state_training_manifest.json",
                    "state_models.pt", "effective_hyperparameters.json",
                    "learned_modules.json", "solver_environment.json",
                    "protocol_manifest.json", "plotting_input_manifest.json",
                ],
                "comparison": ["discrepancy_report.csv"],
                "figures": [
                    "aggregate_metrics.svg",
                    "ablation_delta_heatmap.svg",
                    "figure_manifest.json",
                ],
            },
        )

    plotting_stage = run_stage(
        "plotting_inputs",
        {
            "evaluation": evaluation_stage.content_hash,
            "sources": sources.content_hash,
            "allocations": allocations_stage.content_hash,
            "perturbations": perturbations_stage.content_hash,
            "responses": responses_stage.content_hash,
            "boundaries": boundaries_stage.content_hash,
            "open_set": open_stage.content_hash,
            "dbp15k": dbp_stage.content_hash,
            "baselines": baseline_stage.content_hash,
            "state_training": state_stage.content_hash,
            "ablations": ablation_stage.content_hash,
            "statistics": statistics_stage.content_hash,
            "comparison": comparison_stage.content_hash,
        },
        produce_plotting,
    )

    plotting_config_path = PROJECT_ROOT / "configs" / "plotting.yaml"
    release_config_path = PROJECT_ROOT / "configs" / "release_package.yaml"
    annotation_guide_path = PROJECT_ROOT / "docs" / "annotation_guidelines.md"

    def produce_release_contract(path: Path) -> None:
        state_training_manifest = json.loads(
            (state_stage.path / "state_training_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        sample_rows = build_sample_manifest_rows(
            scicu_cases=execution_cases,
            scicu_metadata=execution_metadata,
            dbp_allocations=resolved_dbp_allocations.rows,
            tuning_cases=tuning_cases,
            tuning_split_by_task=tuning_split_by_id,
            seed=config.paper.seeds[0],
        )
        _write_parquet(path / "sample_manifest.parquet", sample_rows)
        _write_json(
            path / "effective_hyperparameters.json",
            build_effective_hyperparameters(
                config,
                formal_config,
                method_selection,
                state_training_manifest,
            ),
        )
        _write_json(
            path / "learned_modules.json",
            build_learned_modules_manifest(
                config,
                formal_config,
                method_selection,
                state_training_manifest,
            ),
        )
        _write_json(
            path / "solver_environment.json",
            build_solver_environment(config, project_root=PROJECT_ROOT),
        )
        _write_json(path / "protocol_manifest.json", protocol_manifest())
        (path / "motif_dictionary.yaml").write_text(
            protocol_asset_text("motif_dictionary.yaml"),
            encoding="utf-8",
            newline="\n",
        )
        (path / "validator_registry.yaml").write_text(
            protocol_asset_text("validator_registry.yaml"),
            encoding="utf-8",
            newline="\n",
        )
        shutil.copy2(annotation_guide_path, path / "annotation_guidelines.md")
        shutil.copy2(release_config_path, path / "release_package.yaml")
        shutil.copy2(plotting_config_path, path / "plotting.yaml")
        _write_json(
            path / "availability.json",
            {
                "schema_version": 1,
                "review_availability": "immediate_on_request",
                "public_availability": "upon_publication",
                "actual_build_time_written_by": "build_release_package.py",
                "instance_identifier_policy": (
                    "preserve public benchmark IDs; remove personal and machine identity"
                ),
            },
        )
        _write_json(
            path / "package_contract.json",
            {
                "schema_version": 1,
                "path_root": "release_package_root",
                "requirements": {
                    "D1": [
                        "availability.json",
                        "package_manifest.json",
                        "results/release/availability.json",
                        "results/release/motif_dictionary.yaml",
                        "results/release/validator_registry.yaml",
                        "results/computed/per_instance_outputs.parquet",
                    ],
                    "D2": [
                        "project/src",
                        "project/scripts",
                        "project/notebooks",
                        "project/configs/plotting.yaml",
                        "results/figures/figure_manifest.json",
                    ],
                    "D3": ["results/computed/sample_manifest.parquet"],
                    "D4": [
                        "results/release/motif_dictionary.yaml",
                        "results/release/validator_registry.yaml",
                        "results/release/annotation_guidelines.md",
                    ],
                    "D5": ["results/computed/effective_hyperparameters.json"],
                    "D6": [
                        "results/computed/learned_modules.json",
                        "results/computed/state_models.pt",
                    ],
                    "D7": [
                        "project/pyproject.toml",
                        "project/requirements.lock.txt",
                        "project/uv.lock",
                        "results/computed/solver_environment.json",
                    ],
                    "D8": [
                        "results/computed/task_graphs.jsonl",
                        "results/computed/response_graphs.jsonl",
                        "results/computed/validator_outputs.parquet",
                        "results/computed/transport_plans.npz",
                        "results/computed/residual_components.parquet",
                        "results/computed/per_instance_outputs.parquet",
                        "results/computed/bootstrap_results.parquet",
                        "results/computed/significance_results.parquet",
                    ],
                },
                "paper_result_values_hardcoded_in_computation": False,
                "computed_scores_generated_only_by_workflow": True,
            },
        )

        component_links = pd.read_parquet(
            evaluation_stage.path / "instance_component_links.parquet"
        )
        metric_columns = [
            "task_id",
            "model_id",
            "replicate",
            "endpoint_success",
            "tsr",
            "pcs",
            "ccs",
            "residual_consistency",
            "node_loss",
            "structure_loss",
            "constraint_loss",
            "hard_feasible",
            "transport_converged",
            "state_vector",
            "diagnostic_reason",
            "reference_structural_validity",
            "reference_group",
            "reference_label_source",
            "perturbation_valid",
        ]
        enriched = component_links.merge(
            metric_frame[metric_columns],
            on=["task_id", "model_id", "replicate"],
            how="inner",
            validate="one_to_one",
        ).merge(
            assignments,
            on=["task_id", "model_id", "replicate"],
            how="inner",
            validate="one_to_one",
        )
        if len(enriched) != len(component_links):
            raise ValueError("per-instance release links lost evaluation rows")
        enriched["baseline_artifact"] = "computed/baselines.parquet"
        enriched["ablation_artifact"] = "computed/ablations.parquet"
        enriched["validator_artifact"] = "computed/validator_outputs.parquet"
        enriched["transport_artifact"] = "computed/transport_plans.npz"
        enriched["residual_artifact"] = "computed/residual_components.parquet"
        _write_parquet(
            path / "per_instance_outputs.parquet",
            enriched.to_dict(orient="records"),
        )

    release_stage = run_stage(
        "release_contract",
        {
            "allocations": allocations_stage.content_hash,
            "perturbations": perturbations_stage.content_hash,
            "responses": responses_stage.content_hash,
            "method_selection": method_selection_stage.content_hash,
            "evaluation": evaluation_stage.content_hash,
            "state_training": state_stage.content_hash,
            "baselines": baseline_stage.content_hash,
            "ablations": ablation_stage.content_hash,
            "statistics": statistics_stage.content_hash,
            "plotting_inputs": plotting_stage.content_hash,
            "protocol": canonical_sha256(protocol_manifest()),
            "plotting_config": _sha256(plotting_config_path),
            "release_config": _sha256(release_config_path),
            "annotation_guide": _sha256(annotation_guide_path),
        },
        produce_release_contract,
    )

    def produce_figures(path: Path) -> None:
        style = load_plot_style(plotting_config_path)
        aggregate_path = plotting_stage.path / "aggregate_metrics.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate_output = render_aggregate_metrics_svg(
            aggregate,
            style=style,
            target=path / "aggregate_metrics.svg",
        )
        ablation_input = ablation_stage.path / "ablations.parquet"
        ablation_output = render_ablation_heatmap_svg(
            pd.read_parquet(ablation_input),
            style=style,
            target=path / "ablation_delta_heatmap.svg",
        )
        write_figure_manifest(
            target=path / "figure_manifest.json",
            style_path=plotting_config_path,
            inputs={
                "aggregate_metrics": aggregate_path,
                "ablation_rows": ablation_input,
            },
            outputs={
                "aggregate_metrics": aggregate_output,
                "ablation_delta_heatmap": ablation_output,
            },
        )

    figures_stage = run_stage(
        "figures",
        {
            "plotting_inputs": plotting_stage.content_hash,
            "ablations": ablation_stage.content_hash,
            "plotting_config": _sha256(plotting_config_path),
        },
        produce_figures,
    )

    specs = (
        FormalArtifactSpec("computed/responses.parquet", "responses", "responses.parquet"),
        FormalArtifactSpec("computed/tuning_responses.parquet", "responses", "tuning_responses.parquet"),
        FormalArtifactSpec("computed/method_selection.json", "method_selection", "method_selection.json"),
        FormalArtifactSpec("computed/per_case_metrics.parquet", "evaluation", "per_case_metrics.parquet"),
        FormalArtifactSpec("computed/tuning_metrics.parquet", "evaluation", "tuning_metrics.parquet"),
        FormalArtifactSpec("computed/baseline_selection.json", "baselines", "baseline_selection.json"),
        FormalArtifactSpec("computed/baselines.parquet", "baselines", "baselines.parquet"),
        FormalArtifactSpec("computed/baseline_boundaries.parquet", "baselines", "baseline_boundaries.parquet"),
        FormalArtifactSpec("computed/baseline_open_set.parquet", "baselines", "baseline_open_set.parquet"),
        FormalArtifactSpec("computed/ablations.parquet", "ablations", "ablations.parquet"),
        FormalArtifactSpec("computed/boundaries.parquet", "boundaries", "boundaries.parquet"),
        FormalArtifactSpec("computed/open_set.parquet", "open_set", "open_set.parquet"),
        FormalArtifactSpec("computed/bootstrap_results.parquet", "statistics", "bootstrap_results.parquet"),
        FormalArtifactSpec("computed/significance_results.parquet", "statistics", "significance_results.parquet"),
        FormalArtifactSpec("computed/aggregate_metrics.json", "plotting_inputs", "aggregate_metrics.json"),
        FormalArtifactSpec("computed/implementation_contract.json", "plotting_inputs", "implementation_contract.json"),
        FormalArtifactSpec("computed/sample_manifest.parquet", "release_contract", "sample_manifest.parquet"),
        FormalArtifactSpec("computed/task_graphs.jsonl", "evaluation", "task_graphs.jsonl"),
        FormalArtifactSpec("computed/response_graphs.jsonl", "evaluation", "response_graphs.jsonl"),
        FormalArtifactSpec("computed/validator_outputs.parquet", "evaluation", "validator_outputs.parquet"),
        FormalArtifactSpec("computed/transport_plans.npz", "evaluation", "transport_plans.npz"),
        FormalArtifactSpec("computed/transport_diagnostics.parquet", "evaluation", "transport_diagnostics.parquet"),
        FormalArtifactSpec("computed/residual_components.parquet", "evaluation", "residual_components.parquet"),
        FormalArtifactSpec("computed/per_instance_outputs.parquet", "release_contract", "per_instance_outputs.parquet"),
        FormalArtifactSpec("computed/state_assignments.parquet", "state_training", "state_assignments.parquet"),
        FormalArtifactSpec("computed/state_training_manifest.json", "state_training", "state_training_manifest.json"),
        FormalArtifactSpec("computed/state_models.pt", "state_training", "state_models.pt"),
        FormalArtifactSpec("computed/effective_hyperparameters.json", "release_contract", "effective_hyperparameters.json"),
        FormalArtifactSpec("computed/learned_modules.json", "release_contract", "learned_modules.json"),
        FormalArtifactSpec("computed/solver_environment.json", "release_contract", "solver_environment.json"),
        FormalArtifactSpec("computed/protocol_manifest.json", "release_contract", "protocol_manifest.json"),
        FormalArtifactSpec("computed/plotting_input_manifest.json", "plotting_inputs", "plotting_input_manifest.json"),
        FormalArtifactSpec("release/motif_dictionary.yaml", "release_contract", "motif_dictionary.yaml"),
        FormalArtifactSpec("release/validator_registry.yaml", "release_contract", "validator_registry.yaml"),
        FormalArtifactSpec("release/annotation_guidelines.md", "release_contract", "annotation_guidelines.md"),
        FormalArtifactSpec("release/availability.json", "release_contract", "availability.json"),
        FormalArtifactSpec("release/package_contract.json", "release_contract", "package_contract.json"),
        FormalArtifactSpec("release/release_package.yaml", "release_contract", "release_package.yaml"),
        FormalArtifactSpec("release/plotting.yaml", "release_contract", "plotting.yaml"),
        FormalArtifactSpec("figures/aggregate_metrics.svg", "figures", "aggregate_metrics.svg"),
        FormalArtifactSpec("figures/ablation_delta_heatmap.svg", "figures", "ablation_delta_heatmap.svg"),
        FormalArtifactSpec("figures/figure_manifest.json", "figures", "figure_manifest.json"),
        FormalArtifactSpec("paper_reference/published_metrics.json", "paper_comparison", "published_metrics.json"),
        FormalArtifactSpec("comparison/discrepancy_report.csv", "paper_comparison", "discrepancy_report.csv"),
    )
    scalar_columns = (
        "tsr",
        "pcs",
        "ccs",
        "residual_consistency",
        "node_loss",
        "structure_loss",
        "constraint_loss",
    )
    audited_metric_frames = (metric_frame, tuning_metric_frame)
    nonfinite = sum(
        int(not math.isfinite(float(value)))
        for frame in audited_metric_frames
        for column in scalar_columns
        for value in frame[column]
    )
    applicable = sum(len(frame) for frame in audited_metric_frames) * len(
        scalar_columns
    )
    applicable += len(boundary_frame) * 2 + len(dbp_frame) * 4 + 2
    paper_sources = sum(
        int((frame["source"] != "computed").sum())
        for frame in audited_metric_frames
    )
    return assemble_formal_results(
        output,
        scale=scale,
        stages=tuple(receipts),
        artifacts=specs,
        computed_response_count=len(response_frame),
        applicable_metric_cells=applicable,
        nonfinite_metric_cells=nonfinite,
        paper_sources_in_computed_rows=paper_sources,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal SciPRISM-X methodology")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--scale", choices=("smoke", "paper"), default="paper")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--response-manifest",
        type=Path,
        help=(
            "JSON mapping each generator to evaluation_file and tuning_file; "
            "required for Track A"
        ),
    )
    parser.add_argument(
        "--paper-reference",
        type=Path,
        help=(
            "post-computation published-metric file; required for Track A and "
            "defaults to Track B published_targets"
        ),
    )
    parser.add_argument(
        "--judge-response-file",
        type=Path,
        help=(
            "frozen GPT-4o structured judgments keyed by baseline case_id; "
            "otherwise the documented deterministic local route is used"
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_formal_experiment(
        load_run_config(args.config),
        load_formal_config(args.formal_config),
        scale=args.scale,
        output_root=args.output,
        resume=args.resume,
        response_manifest_path=args.response_manifest,
        paper_reference_path=args.paper_reference,
        judge_response_path=args.judge_response_file,
    )
    print(
        json.dumps(
            {
                "run_id": receipt.run_id,
                "scale": receipt.scale,
                "final_path": str(receipt.final_path),
                "completed_stages": receipt.completed_stages,
                "computed_response_count": receipt.computed_response_count,
                "applicable_metric_cells": receipt.applicable_metric_cells,
                "nonfinite_metric_cells": receipt.nonfinite_metric_cells,
                "paper_sources_in_computed_rows": receipt.paper_sources_in_computed_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
