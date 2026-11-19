from __future__ import annotations

from pathlib import Path

from sciprism_x.config import load_run_config
from sciprism_x.formal.config import load_formal_config
from sciprism_x.release.manifests import (
    build_effective_hyperparameters,
    build_learned_modules_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_effective_parameters_name_gamma_as_selected_minimum_support() -> None:
    run = load_run_config(PROJECT_ROOT / "configs" / "track_a.yaml")
    formal = load_formal_config(PROJECT_ROOT / "configs" / "formal_execution.yaml")
    selection = {
        "selected_trial": 7,
        "trial_count": 50,
        "selected_parameters": {
            "epsilon": 0.03,
            "lambda_structure": 1.0,
            "lambda_constraint": 0.5,
            "omega_missing": 2.0,
            "omega_unsupported": 1.0,
            "minimum_support": 0.25,
            "eta_node": 1.0,
            "eta_structure": 0.5,
            "eta_constraint": 0.25,
        },
        "loss_normalizers": {"node": 1.0, "structure": 2.0, "constraint": 3.0},
        "residual_normalizers": {
            "physical": 1.0,
            "logical": 1.0,
            "unit_dimensional": 1.0,
            "causal": 1.0,
            "modal_evidential": 1.0,
            "uncertainty": 1.0,
        },
        "text_models": {
            "model_root": "data/raw/models",
            "scientific_route": "frozen_scibert",
            "entailment_route": "frozen_entailment",
            "scibert_revision": run.validators.scibert.revision,
            "bart_mnli_revision": run.validators.bart_mnli.revision,
        },
    }
    state = {
        "thresholds": {
            "stable_pcs": 0.8,
            "stable_residual": 0.2,
            "invalid_pcs": 0.3,
            "invalid_residual": 0.7,
        },
        "models": [],
    }

    payload = build_effective_hyperparameters(run, formal, selection, state)

    assert payload["method"]["gamma"] == 0.25
    assert payload["method"]["gamma_role"] == "minimum_support_threshold"
    assert payload["method"]["lambda_str"] == 1.0
    assert payload["method"]["lambda_con"] == 0.5
    assert payload["search"]["method_trials"] == 50
    assert payload["solver"]["maximum_iterations"] == 1000


def test_learning_manifest_fully_specifies_state_field_and_frozen_text_modules() -> None:
    run = load_run_config(PROJECT_ROOT / "configs" / "track_a.yaml")
    formal = load_formal_config(PROJECT_ROOT / "configs" / "formal_execution.yaml")
    selection = {
        "text_models": {
            "model_root": "data/raw/models",
            "scientific_route": "frozen_scibert",
            "entailment_route": "frozen_entailment",
            "scibert_revision": run.validators.scibert.revision,
            "bart_mnli_revision": run.validators.bart_mnli.revision,
        }
    }
    state = {"thresholds": {}, "models": []}

    payload = build_learned_modules_manifest(run, formal, selection, state)

    assert payload["g_theta"]["dimensions"] == [15, 64, 16]
    assert payload["e_phi"]["dimensions"] == [31, 32, 1]
    assert payload["state_field_training"]["optimizer"] == "adamw"
    assert payload["state_field_training"]["maximum_epochs"] == 100
    assert payload["entailment_validator"]["pass_threshold"] == 0.8
