"""Acceptance-only fixture workflow.

The paper-scale implementation is ``sciprism_x.formal.workflow``. This module
keeps the small worked-example and isolation checks; its explicitly recorded
first-grid values must not be used as paper-scale selected hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from sciprism_x.config import RunConfig, TrackIsolationError, load_run_config
from sciprism_x.data.schema import SciCUCase
from sciprism_x.formal.evaluator import build_formal_context, evaluate_scicu_components
from sciprism_x.metrics.scientific import compute_ccs, compute_sc
from sciprism_x.motifs.extract import build_response_graph, build_task_graph
from sciprism_x.motifs.salience import SalienceWeights
from sciprism_x.motifs.schema import TypedReasoningGraph
from sciprism_x.provenance import ProvenanceClass
from sciprism_x.residuals.compute import compute_residuals
from sciprism_x.residuals.schema import ResidualChannel, ResidualVector
from sciprism_x.residuals.scores import build_score_bundle
from sciprism_x.responses.adapters import (
    ModelLoader,
    RecoveryRequest,
    RecoveryResult,
    ResponseRecoveryError,
    recover_responses,
)
from sciprism_x.responses.fit import (
    ProfileFitRequest,
    ProfileGroup,
    fit_profiles,
)
from sciprism_x.responses.reconstruct import fixture_cases
from sciprism_x.responses.schema import ReconstructionProfile, ResponseRecord
from sciprism_x.results.schema import RunReceipt
from sciprism_x.results.writer import ArtifactWriter
from sciprism_x.transport.costs import CostWeights, build_costs
from sciprism_x.transport.schema import TransportResult
from sciprism_x.transport.solver import solve_transport
from sciprism_x.validators.deterministic import default_deterministic_validators
from sciprism_x.validators.registry import ValidatorRegistry
from sciprism_x.validators.schema import ValidationPair, ValidationResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ARTIFACTS = (
    "config_resolved.yaml",
    "implementation_settings.json",
    "environment.json",
    "input_manifest.json",
    "source_hashes.json",
    "responses.parquet",
    "task_graphs.jsonl",
    "response_graphs.jsonl",
    "validator_outputs.parquet",
    "transport_plans",
    "per_case_metrics.parquet",
    "aggregate_metrics.json",
    "bootstrap_results.csv",
    "significance_results.csv",
    "ablation_results.csv",
    "logs",
    "figures",
)


@dataclass(slots=True)
class _EvaluatedResponse:
    case: SciCUCase
    response: ResponseRecord
    task_graph: TypedReasoningGraph
    response_graph: TypedReasoningGraph
    validations: tuple[tuple[ValidationResult, ...], ...]
    transport: TransportResult
    provisional_residuals: ResidualVector | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_tree_hash() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _published_target_path(config: RunConfig) -> Path:
    if config.published_targets is None:
        raise TrackIsolationError("Track B requires a published target path")
    path = Path(config.published_targets)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"published target manifest is unavailable: {resolved}")
    return resolved.resolve()


def _derive_profile_targets(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
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
        "length_ratio": "paper does not report response length; neutral ratio 1.0",
        "evidence_synthesis": "1 - worked_case.secondary_diagnostic_gap.value",
    }
    return targets, derivation


def _fit_published_profile(
    config: RunConfig,
    target_path: Path,
) -> tuple[ReconstructionProfile, dict[str, Any], dict[str, Any]]:
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    targets, derivation = _derive_profile_targets(payload)
    worked = payload["worked_case"]
    request = ProfileFitRequest(
        group=ProfileGroup(
            model_id=str(worked["response_model"]),
            regime=str(worked["regime"]).lower(),
            domain=str(worked["domain"]),
            competency=str(worked["competency"]),
        ),
        targets=targets,
        weights={name: 1.0 for name in targets},
    )
    fit = fit_profiles((request,), seed=config.paper.seeds[0])[0]
    audit = {
        "source": target_path.as_posix(),
        "source_sha256": _sha256_file(target_path),
        "target_derivation": derivation,
        "fit": fit.model_dump(mode="json"),
    }
    return fit.profile, audit, payload


def _recover_track_b(
    config: RunConfig,
    cases: tuple[SciCUCase, ...],
    target_path: Path,
) -> tuple[RecoveryResult, dict[str, Any], dict[str, Any]]:
    profile, fit_audit, published = _fit_published_profile(config, target_path)
    recovered = recover_responses(
        RecoveryRequest(
            cases=cases,
            model_id=config.generators.model_ids[0],
            responses_per_task=1,
            seed=config.paper.seeds[0],
            reconstruction_profile=profile,
        )
    )
    return recovered, fit_audit, published


def _recover_track_a(
    config: RunConfig,
    cases: tuple[SciCUCase, ...],
    *,
    supplied_responses: Path | None,
    exact_model_loader: ModelLoader | None,
    exact_endpoint: str | None,
    official_call_authorized: bool,
) -> RecoveryResult:
    try:
        return recover_responses(
            RecoveryRequest(
                cases=cases,
                model_id=config.generators.model_ids[0],
                responses_per_task=1,
                seed=config.paper.seeds[0],
                reconstruction_profile=None,
                supplied_file=supplied_responses,
                exact_model_loader=exact_model_loader,
                exact_endpoint=exact_endpoint,
                official_call_authorized=official_call_authorized,
            )
        )
    except ResponseRecoveryError as error:
        raise TrackIsolationError(
            "Track A requires supplied or exactly acquired responses and never reconstructs them"
        ) from error


def _graph_settings(config: RunConfig) -> dict[str, Any]:
    weights = SalienceWeights(**config.graph.salience_weights.model_dump())
    return {
        "short_fragment_token_threshold": config.graph.short_fragment_token_threshold,
        "explicit_cue_confidence": config.graph.explicit_cue_confidence,
        "lexical_cue_confidence": config.graph.lexical_cue_confidence,
        "salience_weights": weights,
    }


def _validation_matrix(
    task_graph: TypedReasoningGraph,
    response_graph: TypedReasoningGraph,
    reference_context: str,
    registry: ValidatorRegistry,
) -> tuple[tuple[ValidationResult, ...], ...]:
    return tuple(
        tuple(
            registry.validate(
                ValidationPair(
                    task_node=task_node,
                    response_node=response_node,
                    reference_context=reference_context,
                )
            )
            for response_node in response_graph.nodes
        )
        for task_node in task_graph.nodes
    )


def _evaluate_responses(
    config: RunConfig,
    cases: tuple[SciCUCase, ...],
    responses: tuple[ResponseRecord, ...],
) -> tuple[
    list[_EvaluatedResponse],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    case_by_id = {case.task_id: case for case in cases}
    context = build_formal_context(
        config,
        training_texts=tuple(
            text
            for case in cases
            for text in (case.prompt, case.reference_context, case.reference_answer)
        ),
    )
    task_graphs: dict[str, TypedReasoningGraph] = {}
    evaluated: list[_EvaluatedResponse] = []
    response_graph_rows: list[dict[str, Any]] = []
    validator_rows: list[dict[str, Any]] = []
    plan_manifest: list[dict[str, Any]] = []

    for index, response in enumerate(responses):
        case = case_by_id[response.task_id]
        components = evaluate_scicu_components(
            case=case,
            response=response,
            context=context,
        )
        task_graph = components.task_graph
        response_graph = components.response_graph
        validations = components.validations
        transport = components.transport
        task_graphs[case.task_id] = task_graph
        evaluated.append(
            _EvaluatedResponse(
                case,
                response,
                task_graph,
                response_graph,
                validations,
                transport,
                components.residuals,
            )
        )
        response_graph_rows.append(response_graph.model_dump(mode="json"))
        for task_index, row in enumerate(validations):
            for response_index, result in enumerate(row):
                validator_rows.append(
                    {
                        "task_id": case.task_id,
                        "response_source_id": response_graph.source_id,
                        "task_node_id": task_graph.nodes[task_index].node_id,
                        "response_node_id": response_graph.nodes[response_index].node_id,
                        "status": result.status.value,
                        "confidence": result.confidence,
                        "reason_code": result.reason_code,
                        "validator_id": result.validator_id,
                    }
                )
        plan_manifest.append(
            {
                "index": index,
                "task_id": case.task_id,
                "response_source_id": response_graph.source_id,
                "diagnostics": {
                    "feasible": transport.diagnostics.feasible,
                    "converged": transport.diagnostics.converged,
                    "iterations": transport.diagnostics.iterations,
                    "marginal_error": transport.diagnostics.marginal_error,
                    "objective_change": transport.diagnostics.objective_change,
                    "reason": transport.diagnostics.reason,
                },
            }
        )
    return (
        evaluated,
        [task_graphs[case.task_id].model_dump(mode="json") for case in cases],
        response_graph_rows,
        validator_rows,
        plan_manifest,
    )


def _fit_residual_normalizers(
    evaluated: list[_EvaluatedResponse], config: RunConfig
) -> dict[ResidualChannel, float]:
    normalizers: dict[ResidualChannel, float] = {}
    for channel in ResidualChannel:
        values = [
            item.provisional_residuals.components[channel].raw
            for item in evaluated
            if item.provisional_residuals is not None
        ]
        normalizers[channel] = (
            max(
                float(
                    np.quantile(
                        values,
                        config.residuals.normalizer_percentile,
                        method="linear",
                    )
                ),
                config.residuals.normalizer_floor,
            )
            if values
            else config.residuals.normalizer_floor
        )
    return normalizers


def _metric_rows(
    evaluated: list[_EvaluatedResponse], config: RunConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalizers = _fit_residual_normalizers(evaluated, config)
    rows: list[dict[str, Any]] = []
    for item in evaluated:
        endpoint = item.response.response_text.strip() == item.case.reference_answer.strip()
        if item.transport.objective is None or not item.transport.diagnostics.feasible:
            row = {
                "task_id": item.case.task_id,
                "model_id": item.response.model_id,
                "replicate": item.response.replicate,
                "endpoint_success": endpoint,
                "tsr": float(endpoint),
                "pcs": 0.0,
                "residual_consistency": 0.0,
                "node_loss": 0.0,
                "structure_loss": 0.0,
                "constraint_loss": 0.0,
                "loss_defined": False,
                "transport_converged": False,
                "diagnostic_regime": "INVALID",
                "diagnostic_reason": item.transport.diagnostics.reason,
            }
            row.update(
                {f"residual_{channel.value}": 1.0 for channel in ResidualChannel}
            )
            rows.append(row)
            continue
        residuals = compute_residuals(
            transport=item.transport,
            task_graph=item.task_graph,
            response_graph=item.response_graph,
            validations=item.validations,
            omega_missing=config.residuals.missing_penalty_grid[0],
            omega_unsupported=config.residuals.unsupported_penalty_grid[0],
            minimum_support=config.residuals.minimum_support_threshold,
            normalizers=normalizers,
            normalizer_floor=config.residuals.normalizer_floor,
        )
        assert item.transport.objective is not None
        score = build_score_bundle(
            endpoint_success=(endpoint,),
            transport_objective=item.transport.objective,
            residuals=residuals,
            eta_node=config.residuals.pcs_coefficient_grid[0],
            eta_structure=config.residuals.pcs_coefficient_grid[0],
            eta_constraint=config.residuals.pcs_coefficient_grid[0],
            beta=np.asarray(config.residuals.residual_consistency_beta, dtype=np.float64),
        )
        row: dict[str, Any] = {
            "task_id": item.case.task_id,
            "model_id": item.response.model_id,
            "replicate": item.response.replicate,
            "endpoint_success": endpoint,
            "tsr": score.tsr,
            "pcs": score.pcs,
            "residual_consistency": score.residual_consistency,
            "node_loss": score.node_loss,
            "structure_loss": score.structure_loss,
            "constraint_loss": score.constraint_loss,
            "loss_defined": True,
            "transport_converged": item.transport.diagnostics.converged,
            "diagnostic_regime": "EVALUATED",
            "diagnostic_reason": item.transport.diagnostics.reason,
        }
        row.update(
            {
                f"residual_{channel.value}": residuals.components[channel].normalized
                for channel in ResidualChannel
            }
        )
        rows.append(row)

    tsr_values = np.asarray([row["tsr"] for row in rows], dtype=np.float64)
    pcs_values = np.asarray([row["pcs"] for row in rows], dtype=np.float64)
    confidence = np.clip(pcs_values, 0.0, 1.0)
    validity = np.asarray([row["endpoint_success"] for row in rows], dtype=np.int64)
    calibration = compute_sc(
        confidence=confidence,
        structural_validity=validity,
        bins=config.open_set.calibration_bins,
        minimum_bin_samples=config.open_set.calibration_minimum_bin_samples,
        minimum_bin_fraction=config.open_set.calibration_minimum_bin_fraction,
    )
    aggregate = {
        "fixture_only": True,
        "case_count": len(rows),
        "computed": {
            "TSR": float(tsr_values.mean()),
            "PCS": float(pcs_values.mean()),
            "CCS": compute_ccs(pcs_values, pcs_values),
            "SC": calibration.sc,
            "residual_consistency": float(
                np.mean([row["residual_consistency"] for row in rows])
            ),
        },
        "not_applicable_to_unperturbed_acceptance_fixture": {
            "BSI": "requires materialized validity-preserving perturbations",
            "UPDQ": "requires audited unfamiliar failure categories and seed-wise clusterings",
            "SCS_TP_HitsAt1_MRR": "DBP15K-only metrics",
        },
        "residual_normalizers": {
            channel.value: normalizers[channel] for channel in ResidualChannel
        },
        "calibration_bins": [
            {
                "index": bin_result.index,
                "lower": bin_result.lower,
                "upper": bin_result.upper,
                "count": bin_result.count,
                "fraction": bin_result.fraction,
                "mean_confidence": bin_result.mean_confidence,
                "observed_validity": bin_result.observed_validity,
                "absolute_gap": bin_result.absolute_gap,
                "sparse": bin_result.sparse,
            }
            for bin_result in calibration.bins
        ],
    }
    return rows, aggregate


def _response_table(rows: tuple[ResponseRecord, ...]) -> pa.Table:
    schema = pa.schema(
        (
            ("task_id", pa.string()),
            ("model_id", pa.string()),
            ("replicate", pa.int64()),
            ("response_text", pa.string()),
        )
    )
    return pa.Table.from_pylist([row.model_dump(mode="json") for row in rows], schema=schema)


def _published_report_rows(
    published: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if published is None:
        return (
            [{"status": "not_available_in_track_a", "source": "none"}],
            [{"status": "not_available_in_track_a", "source": "none"}],
            [{"status": "not_run_in_workflow", "source": "none"}],
        )
    bootstrap: list[dict[str, Any]] = []
    significance: list[dict[str, Any]] = []
    for dataset in ("scicueval", "dbp15k"):
        for metric, regimes in published[dataset].items():
            for regime, values in regimes.items():
                if not isinstance(values, dict):
                    continue
                ci = values.get("ci95")
                if ci is not None:
                    bootstrap.append(
                        {
                            "source": "paper_published_target",
                            "dataset": dataset,
                            "metric": metric,
                            "regime": regime,
                            "n": values["n"],
                            "observed_difference": values["delta"],
                            "ci_lower": ci[0],
                            "ci_upper": ci[1],
                            "resamples": 1000,
                        }
                    )
                significance.append(
                    {
                        "source": "paper_published_target",
                        "dataset": dataset,
                        "metric": metric,
                        "regime": regime,
                        "reported_adjusted_significance": values["adjusted_significance"],
                        "exact_p_available": False,
                    }
                )
    ablations = [
        {
            "source": "paper_published_target",
            "ablation": name,
            "metric": metric,
            "delta": value,
        }
        for name, metrics in published["ablation_deltas"].items()
        for metric, value in metrics.items()
    ]
    return bootstrap, significance, ablations


def _environment_payload() -> dict[str, Any]:
    packages = (
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "pydantic",
        "PyYAML",
        "networkx",
        "sympy",
        "pint",
        "spacy",
        "click",
        "torch",
        "transformers",
        "scikit-learn",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _write_run(
    *,
    config: RunConfig,
    output_root: Path,
    cases: tuple[SciCUCase, ...],
    recovered: RecoveryResult,
    cache_components: dict[str, Any],
    published_fit: dict[str, Any] | None,
    published: dict[str, Any] | None,
) -> RunReceipt:
    implementation_hash = _source_tree_hash()
    cache_hash = _canonical_hash(
        {**cache_components, "implementation_sha256": implementation_hash}
    )
    writer = ArtifactWriter(output_root)
    handle = writer.create_run(cache_hash)
    try:
        evaluated, task_graphs, response_graphs, validators, plan_manifest = _evaluate_responses(
            config,
            cases,
            recovered.rows,
        )
        metric_rows, aggregate = _metric_rows(evaluated, config)
        implementation_settings = [
            {
                "path": record.path,
                "value": record.value,
                "provenance": record.provenance.value,
            }
            for record in config.parameter_records()
            if record.provenance is ProvenanceClass.IMPLEMENTATION_SETTING
        ]
        implementation_settings.extend(
            [
                {
                    "path": "workflow.fixture_grid_selection",
                    "value": "first value of each frozen validation grid",
                    "provenance": ProvenanceClass.IMPLEMENTATION_SETTING.value,
                },
                {
                    "path": "workflow.fixture_scope",
                    "value": (
                        "acceptance only; not a replacement for the formal "
                        "600/2400 allocation"
                    ),
                    "provenance": ProvenanceClass.IMPLEMENTATION_SETTING.value,
                },
                {
                    "path": "workflow.infeasible_transport_mapping",
                    "value": (
                        "preserve hard constraints; map infeasible_hard_mask to INVALID, "
                        "PCS=0, residual_consistency=0, all normalized residuals=1; "
                        "loss fields are 0 only with loss_defined=false"
                    ),
                    "provenance": ProvenanceClass.IMPLEMENTATION_SETTING.value,
                },
            ]
        )
        source_hashes = {
            "implementation": implementation_hash,
            "cases": {case.task_id: case.source_hash for case in cases},
            "responses": recovered.acquisition.source_sha256,
        }
        if published_fit is not None:
            source_hashes["published_targets"] = published_fit["source_sha256"]

        writer.write_yaml(handle, "config_resolved.yaml", config.model_dump(mode="json"))
        writer.write_json(handle, "implementation_settings.json", implementation_settings)
        writer.write_json(handle, "environment.json", _environment_payload())
        writer.write_json(
            handle,
            "input_manifest.json",
            {
                "fixture_only": True,
                "fixture_case_count": len(cases),
                "acquisition": recovered.acquisition.model_dump(mode="json"),
            },
        )
        writer.write_json(handle, "source_hashes.json", source_hashes)
        writer.write_parquet(handle, "responses.parquet", _response_table(recovered.rows))
        writer.write_jsonl(handle, "task_graphs.jsonl", task_graphs)
        writer.write_jsonl(handle, "response_graphs.jsonl", response_graphs)
        writer.write_parquet(
            handle,
            "validator_outputs.parquet",
            pa.Table.from_pylist(validators),
        )
        for index, item in enumerate(evaluated):
            writer.write_numpy(
                handle,
                f"transport_plans/plan-{index:05d}.npy",
                item.transport.plan,
            )
        writer.write_json(handle, "transport_plans/manifest.json", plan_manifest)
        writer.write_parquet(
            handle,
            "per_case_metrics.parquet",
            pa.Table.from_pylist(metric_rows),
        )
        writer.write_json(handle, "aggregate_metrics.json", aggregate)

        bootstrap, significance, ablations = _published_report_rows(published)
        writer.write_csv(
            handle,
            "bootstrap_results.csv",
            bootstrap,
            tuple(bootstrap[0]),
        )
        writer.write_csv(
            handle,
            "significance_results.csv",
            significance,
            tuple(significance[0]),
        )
        writer.write_csv(
            handle,
            "ablation_results.csv",
            ablations,
            tuple(ablations[0]),
        )
        if published_fit is not None:
            writer.write_json(handle, "published_target_fit.json", published_fit)
        writer.write_text(
            handle,
            "logs/workflow.log",
            f"track={config.track}\nfixture_only=true\ncases={len(cases)}\nstatus=complete\n",
        )
        writer.write_json(
            handle,
            "figures/manifest.json",
            {
                "status": "deferred",
                "reason": "plot styling will be applied from the user-supplied figure template",
                "source_artifacts": ["aggregate_metrics.json", "per_case_metrics.parquet"],
            },
        )
        return writer.finalize(handle, track=config.track)
    except BaseException:
        writer.abort(handle)
        raise


def run_track_a(
    config: RunConfig,
    *,
    output_root: Path = Path("D:/SciPRISM-X-runs/fixture"),
    fixture_count: int = 12,
    supplied_responses: Path | None = None,
    exact_model_loader: ModelLoader | None = None,
    exact_endpoint: str | None = None,
    official_call_authorized: bool = False,
) -> RunReceipt:
    if config.track != "A" or config.published_targets is not None:
        raise TrackIsolationError("Track A cannot receive published target data")
    if fixture_count < 1:
        raise ValueError("fixture_count must be positive")
    cases = fixture_cases(fixture_count)
    recovered = _recover_track_a(
        config,
        cases,
        supplied_responses=supplied_responses,
        exact_model_loader=exact_model_loader,
        exact_endpoint=exact_endpoint,
        official_call_authorized=official_call_authorized,
    )
    supplied_hash = (
        _sha256_file(supplied_responses)
        if supplied_responses is not None and supplied_responses.is_file()
        else None
    )
    cache_components = {
        "config": config.model_dump(mode="json"),
        "fixture_count": fixture_count,
        "supplied_responses_sha256": supplied_hash,
        "exact_endpoint": exact_endpoint,
    }
    return _write_run(
        config=config,
        output_root=output_root,
        cases=cases,
        recovered=recovered,
        cache_components=cache_components,
        published_fit=None,
        published=None,
    )


def run_track_b(
    config: RunConfig,
    *,
    output_root: Path = Path("D:/SciPRISM-X-runs/fixture"),
    fixture_count: int = 12,
) -> RunReceipt:
    if config.track != "B" or config.published_targets is None:
        raise TrackIsolationError("Track B requires published targets")
    if fixture_count < 1:
        raise ValueError("fixture_count must be positive")
    cases = fixture_cases(fixture_count)
    target_path = _published_target_path(config)
    recovered, fit_audit, published = _recover_track_b(config, cases, target_path)
    cache_components = {
        "config": config.model_dump(mode="json"),
        "fixture_count": fixture_count,
        "published_targets_sha256": fit_audit["source_sha256"],
    }
    return _write_run(
        config=config,
        output_root=output_root,
        cases=cases,
        recovered=recovered,
        cache_components=cache_components,
        published_fit=fit_audit,
        published=published,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the isolated SciPRISM-X workflow")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixture-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = load_run_config(arguments.config)
    if config.track == "A":
        parser.error("Track A CLI requires supplied or authorized exact responses")
    receipt = run_track_b(
        config,
        output_root=arguments.output,
        fixture_count=arguments.fixture_count,
    )
    print(
        json.dumps(
            {
                "run_id": receipt.run_id,
                "cache_hash": receipt.cache_hash,
                "track": receipt.track,
                "final_path": receipt.final_path.as_posix(),
                "artifact_count": len(receipt.artifact_hashes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
