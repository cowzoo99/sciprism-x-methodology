from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import sys

import torch

from sciprism_x.config import RunConfig
from sciprism_x.data.schema import SciCUCase
from sciprism_x.formal.config import FormalExecutionConfig


_DEPENDENCIES = (
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
    "nbformat",
    "nbclient",
    "ipykernel",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"release manifest requires mapping: {key}")
    return value


def build_effective_hyperparameters(
    run: RunConfig,
    formal: FormalExecutionConfig,
    method_selection: Mapping[str, object],
    state_training: Mapping[str, object],
) -> dict[str, object]:
    selected = _required_mapping(method_selection, "selected_parameters")
    required = {
        "epsilon",
        "lambda_structure",
        "lambda_constraint",
        "omega_missing",
        "omega_unsupported",
        "minimum_support",
        "eta_node",
        "eta_structure",
        "eta_constraint",
    }
    if set(selected) != required:
        raise ValueError("method selection does not expose every paper coefficient")
    if int(method_selection.get("trial_count", -1)) != run.paper.hyperparameter_trials:
        raise ValueError("method selection trial count differs from paper budget")
    thresholds = _required_mapping(state_training, "thresholds")
    if set(thresholds) != {
        "stable_pcs",
        "stable_residual",
        "invalid_pcs",
        "invalid_residual",
    }:
        raise ValueError("state training does not expose all regime thresholds")
    loss_normalizers = _required_mapping(method_selection, "loss_normalizers")
    residual_normalizers = _required_mapping(
        method_selection,
        "residual_normalizers",
    )
    return {
        "schema_version": 1,
        "method": {
            "lambda_str": float(selected["lambda_structure"]),
            "lambda_con": float(selected["lambda_constraint"]),
            "epsilon": float(selected["epsilon"]),
            "eta_1_node": float(selected["eta_node"]),
            "eta_2_structure": float(selected["eta_structure"]),
            "eta_3_constraint": float(selected["eta_constraint"]),
            "gamma": float(selected["minimum_support"]),
            "gamma_role": "minimum_support_threshold",
            "omega_miss": float(selected["omega_missing"]),
            "omega_unsup": float(selected["omega_unsupported"]),
            "selected_trial": int(method_selection["selected_trial"]),
            "selection_split": "validation",
        },
        "normalizers": {
            "transport": {key: float(value) for key, value in loss_normalizers.items()},
            "residual_channels": {
                key: float(value) for key, value in residual_normalizers.items()
            },
            "transport_quantile": run.transport.loss_normalizer_percentile,
            "transport_floor": run.transport.loss_normalizer_floor,
            "residual_quantile": run.residuals.normalizer_percentile,
            "residual_floor": run.residuals.normalizer_floor,
        },
        "validator_thresholds": {
            "learned_entailment_pass": run.validators.learned_pass_threshold,
            "learned_entailment_fail": run.validators.learned_fail_threshold,
            "prototype_recognizer_cosine": run.graph.learned_recognizer_threshold,
            "prototype_recognizer_margin": run.graph.learned_recognizer_margin,
            "symbolic_timeout_seconds": run.validators.symbolic_timeout_seconds,
            "rule_first": True,
        },
        "regime_thresholds": {
            key: float(value) for key, value in thresholds.items()
        },
        "solver": {
            "algorithm": "stabilized_sinkhorn_with_iterative_structural_updates",
            "dtype": run.transport.dtype,
            "maximum_iterations": run.transport.max_iterations,
            "marginal_tolerance": run.transport.marginal_tolerance,
            "objective_tolerance": run.transport.objective_tolerance,
            "required_consecutive_stable_iterations": run.transport.stable_iterations,
            "stopping_rule": (
                "stop after both marginal and objective tolerances hold for the "
                "configured number of consecutive iterations"
            ),
        },
        "search": {
            "method_trials": run.paper.hyperparameter_trials,
            "state_field_trials_per_seed": run.paper.hyperparameter_trials,
            "maximum_epochs": run.paper.max_epochs,
            "early_stopping_patience": run.paper.patience,
            "method_spaces": {
                "epsilon": list(run.transport.epsilon_grid),
                "lambda_str": list(run.transport.lambda_structure_grid),
                "lambda_con": list(run.transport.lambda_constraint_grid),
                "eta_1_to_3": list(run.residuals.pcs_coefficient_grid),
                "gamma": list(run.residuals.gamma_grid),
                "omega_miss": list(run.residuals.missing_penalty_grid),
                "omega_unsup": list(run.residuals.unsupported_penalty_grid),
            },
            "state_field_spaces": {
                "learning_rate": list(run.state_field.learning_rate_grid),
                "weight_decay": list(run.state_field.weight_decay_grid),
                "energy_loss_weight": list(run.state_field.energy_loss_weight_grid),
                "energy_margin": list(run.state_field.energy_margin_grid),
            },
        },
        "regime_threshold_selection": {
            "stable_pcs_quantile": formal.stable_pcs_quantile,
            "stable_residual_quantile": formal.stable_residual_quantile,
            "invalid_pcs_quantile": formal.invalid_pcs_quantile,
            "invalid_residual_quantile": formal.invalid_residual_quantile,
            "split": "independent_validation",
        },
        "perturbations": {
            "standardized_radii": list(run.boundary.standardized_radii),
            "symbolic_depth_steps": list(run.boundary.symbolic_depth_steps),
            "context_composition_steps": list(run.boundary.context_composition_steps),
            "dbp_removal_ratios": list(run.boundary.dbp_removal_ratios),
            "dbp_seed_fractions": list(run.boundary.dbp_seed_fractions),
        },
        "open_set": run.open_set.model_dump(mode="json"),
        "statistics": {
            "bootstrap_resamples": run.paper.bootstrap_resamples,
            "confidence_level": run.paper.confidence_level,
            "two_sided_tests": run.paper.two_sided_tests,
            "multiple_testing_correction": run.paper.correction,
            "bootstrap_seed": run.paper.seeds[0],
        },
    }


def build_learned_modules_manifest(
    run: RunConfig,
    formal: FormalExecutionConfig,
    method_selection: Mapping[str, object],
    state_training: Mapping[str, object],
) -> dict[str, object]:
    text_models = _required_mapping(method_selection, "text_models")
    state_dimensions = run.state_field.dimensions
    energy_dimensions = run.state_field.energy_dimensions
    activation = (
        f"{run.state_field.activation}_exact"
        if run.state_field.gelu_approximate == "none"
        else f"{run.state_field.activation}_{run.state_field.gelu_approximate}"
    )
    return {
        "schema_version": 1,
        "g_theta": {
            "role": "reasoning_state_encoder",
            "dimensions": list(state_dimensions),
            "layers": [
                f"linear_{state_dimensions[0]}_{state_dimensions[1]}",
                activation,
                f"linear_{state_dimensions[1]}_{state_dimensions[2]}",
            ],
            "regime_classifier": {
                "dimensions": [state_dimensions[-1], 3],
                "activation": "none_logits",
                "labels": {"stable": 0, "brittle": 1, "invalid": 2},
            },
            "dropout": run.state_field.dropout,
            "linear_bias": run.state_field.linear_bias,
            "initialization": run.state_field.initialization,
            "dtype": "float64",
        },
        "e_phi": {
            "role": "scalar_reasoning_energy",
            "input": (
                f"concatenate_g_theta_{state_dimensions[-1]}d_embedding_and_"
                f"{state_dimensions[0]}d_state"
            ),
            "dimensions": list(energy_dimensions),
            "layers": [
                f"linear_{energy_dimensions[0]}_{energy_dimensions[1]}",
                activation,
                f"linear_{energy_dimensions[1]}_{energy_dimensions[2]}",
            ],
            "dropout": run.state_field.dropout,
            "linear_bias": run.state_field.linear_bias,
            "initialization": run.state_field.initialization,
            "dtype": "float64",
        },
        "implicit_motif_recognizer": {
            "route": str(text_models["scientific_route"]),
            "model_id": run.validators.scibert.model_id,
            "revision": run.validators.scibert.revision,
            "pooling": run.validators.scibert_pooling,
            "maximum_length": run.validators.scibert_max_length,
            "decision_rule": (
                "only units with no explicit cue are assigned to the highest "
                "prototype cosine above the frozen threshold and margin"
            ),
            "cosine_threshold": run.graph.learned_recognizer_threshold,
            "minimum_margin": run.graph.learned_recognizer_margin,
            "training_data": "frozen pretrained checkpoint; no final-test fine-tuning",
            "training_labels": (
                "six motif-family labels and versioned prototypes frozen in "
                "motif_dictionary.yaml"
            ),
            "objective": "highest prototype cosine subject to threshold and family margin",
            "fallback": "training-split-only TF-IDF encoder when frozen assets are absent",
        },
        "entailment_validator": {
            "route": str(text_models["entailment_route"]),
            "model_id": run.validators.bart_mnli.model_id,
            "revision": run.validators.bart_mnli.revision,
            "maximum_length": run.validators.bart_max_length,
            "pass_threshold": run.validators.learned_pass_threshold,
            "fail_threshold": run.validators.learned_fail_threshold,
            "training_data": "frozen pretrained NLI checkpoint; no final-test fine-tuning",
            "training_labels": "pretrained MNLI entailment, neutral, and contradiction labels",
            "objective": (
                "map entailment probability to PASS and contradiction probability to "
                "FAIL only above frozen thresholds"
            ),
            "fallback": "deterministic five-feature NLI proxy with frozen weights",
        },
        "state_field_training": {
            "training_data": "independent tuning-train structured state vectors",
            "validation_data": "independent tuning-validation structured state vectors",
            "label_source": (
                "validation-fitted PCS and residual thresholds with hard-feasibility "
                "override; deterministic ranked thirds only if a split lacks a regime"
            ),
            "labels": {"stable": 0, "brittle": 1, "invalid": 2},
            "objective": (
                "inverse-frequency weighted three-class cross entropy plus weighted "
                "ordered-pair energy margin loss"
            ),
            "optimizer": run.state_field.optimizer,
            "optimizer_betas": list(run.state_field.adamw_betas),
            "optimizer_epsilon": run.state_field.adamw_eps,
            "batch_size": run.state_field.batch_size,
            "maximum_epochs": run.paper.max_epochs,
            "early_stopping_patience": run.paper.patience,
            "early_stopping_min_delta": run.state_field.early_stopping_min_delta,
            "class_weighting": run.state_field.class_weighting,
            "deterministic_shuffling": run.state_field.deterministic_shuffling,
            "threshold_quantiles": {
                "stable_pcs": formal.stable_pcs_quantile,
                "stable_residual": formal.stable_residual_quantile,
                "invalid_pcs": formal.invalid_pcs_quantile,
                "invalid_residual": formal.invalid_residual_quantile,
            },
            "resolved_thresholds": dict(state_training.get("thresholds", {})),
            "checkpoints": list(state_training.get("models", [])),
        },
    }


def build_sample_manifest_rows(
    *,
    scicu_cases: Sequence[SciCUCase],
    scicu_metadata: Mapping[str, Mapping[str, object]],
    dbp_allocations: Sequence[object],
    tuning_cases: Sequence[SciCUCase],
    tuning_split_by_task: Mapping[str, str],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in scicu_cases:
        meta = scicu_metadata[case.task_id]
        allocation = _required_mapping(meta, "allocation")
        regime = str(allocation["regime"])
        original_id = str(meta["original_task_id"])
        rows.append(
            {
                "dataset": "scicueval",
                "sample_id": case.task_id,
                "original_sample_id": original_id,
                "parent_sample_id": original_id,
                "allocation_id": str(allocation["allocation_id"]),
                "split": "formal_final_test",
                "regime": regime,
                "stratum": f"{case.domain}::{case.competency}",
                "family": str(allocation.get("family") or "not_applicable"),
                "strength": float(allocation.get("strength") or 0.0),
                "language_pair": "not_applicable",
                "bsi_regime": str(allocation.get("bsi_regime") or "not_applicable"),
                "selection_status": str(meta["selection_status"]),
                "valid": bool(meta["valid"]),
                "seed": seed,
            }
        )
    for allocation in dbp_allocations:
        payload = asdict(allocation) if is_dataclass(allocation) else dict(allocation)
        regime = str(payload["regime"])
        language = str(payload.get("language_pair") or "not_applicable")
        rows.append(
            {
                "dataset": "dbp15k",
                "sample_id": str(payload["allocation_id"]),
                "original_sample_id": str(payload["case_id"]),
                "parent_sample_id": str(payload["case_id"]),
                "allocation_id": str(payload["allocation_id"]),
                "split": "formal_final_test",
                "regime": regime,
                "stratum": f"{language}::{regime}",
                "family": str(payload.get("family") or "not_applicable"),
                "strength": float(payload.get("strength") or 0.0),
                "language_pair": language,
                "bsi_regime": str(payload.get("bsi_regime") or "not_applicable"),
                "selection_status": "deterministic_allocated",
                "valid": True,
                "seed": seed,
            }
        )
    for case in tuning_cases:
        split = str(tuning_split_by_task[case.task_id])
        rows.append(
            {
                "dataset": "scicueval",
                "sample_id": case.task_id,
                "original_sample_id": case.task_id,
                "parent_sample_id": case.task_id,
                "allocation_id": f"tuning::{split}::{case.task_id}",
                "split": f"tuning_{split}",
                "regime": f"tuning_{split}",
                "stratum": f"{case.domain}::{case.competency}",
                "family": "not_applicable",
                "strength": 0.0,
                "language_pair": "not_applicable",
                "bsi_regime": "not_applicable",
                "selection_status": "independent_hash_split",
                "valid": True,
                "seed": seed,
            }
        )
    identifiers = [(row["dataset"], row["sample_id"], row["split"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("sample manifest identifiers must be unique within each split")
    return rows


def _memory_record() -> dict[str, object]:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"status": "reported", "total_physical_bytes": status.total_physical}
        return {"status": "os_query_failed", "total_physical_bytes": 0}
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 0
    pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    total = int(page_size) * int(pages) if page_size and pages else 0
    return {
        "status": "reported" if total else "os_query_unavailable",
        "total_physical_bytes": total,
    }


def build_solver_environment(
    run: RunConfig,
    *,
    project_root: Path,
) -> dict[str, object]:
    root = Path(project_root)
    lock_files = ("pyproject.toml", "requirements.lock.txt", "uv.lock")
    locks = {}
    for name in lock_files:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"environment lock file is missing: {name}")
        locks[name] = _sha256(path)
    gpu_devices = []
    if torch.cuda.is_available():
        gpu_devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    return {
        "schema_version": 1,
        "software": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in _DEPENDENCIES},
        },
        "lock_sha256": locks,
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "hardware": {
            "cpu": processor or "not_reported_by_operating_system",
            "memory": _memory_record(),
            "gpu_availability": "cuda" if gpu_devices else "cpu_only",
            "gpu_devices": gpu_devices,
        },
        "randomness": {
            "seeds": list(run.paper.seeds),
            "python_hash_seed_policy": "set by seed_everything before learned stages",
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "solver": {
            "algorithm": "stabilized_sinkhorn_with_iterative_structural_updates",
            "dtype": run.transport.dtype,
            "maximum_iterations": run.transport.max_iterations,
            "marginal_tolerance": run.transport.marginal_tolerance,
            "objective_tolerance": run.transport.objective_tolerance,
            "required_consecutive_stable_iterations": run.transport.stable_iterations,
        },
        "preprocessing": {
            "scientific_unit_segmentation": {
                "short_fragment_token_threshold": (
                    run.graph.short_fragment_token_threshold
                ),
            },
            "motif_recognition": {
                "explicit_cue_confidence": run.graph.explicit_cue_confidence,
                "lexical_cue_confidence": run.graph.lexical_cue_confidence,
                "learned_cosine_threshold": run.graph.learned_recognizer_threshold,
                "learned_family_margin": run.graph.learned_recognizer_margin,
            },
            "scientific_encoder": {
                "pooling": run.validators.scibert_pooling,
                "maximum_length": run.validators.scibert_max_length,
            },
            "entailment_tokenizer": {
                "maximum_length": run.validators.bart_max_length,
            },
            "tfidf_fallback": run.validators.tfidf_proxy.model_dump(mode="json"),
        },
        "identity_fields_intentionally_omitted": [
            "hostname",
            "username",
            "home_directory",
        ],
        "python_executable_scope": "environment_version_only",
        "byte_order": sys.byteorder,
    }
