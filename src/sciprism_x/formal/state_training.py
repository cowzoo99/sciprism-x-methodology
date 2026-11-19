from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sciprism_x.config import RunConfig
from sciprism_x.state_field.train import (
    StateTrainingConfig,
    StateTrainingSet,
    train_state_field,
)
from sciprism_x.tuning import build_trials

from .config import FormalExecutionConfig, load_formal_config
from .evaluator import FormalEvaluationRow


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    stable_pcs: float
    stable_residual: float
    invalid_pcs: float
    invalid_residual: float


@dataclass(frozen=True, slots=True)
class StateModelCheckpoint:
    seed: int
    selected_trial: int
    trial_count: int
    learning_rate: float
    weight_decay: float
    energy_loss_weight: float
    energy_margin: float
    selected_epoch: int
    validation_loss: float
    validation_accuracy: float
    state_dict_hash: str
    state_dict: dict[str, torch.Tensor] = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class StateModelBundle:
    models: tuple[StateModelCheckpoint, ...]
    thresholds: RegimeThresholds
    training_labels: np.ndarray
    validation_labels: np.ndarray


def fit_regime_thresholds(
    validation_rows: tuple[FormalEvaluationRow, ...],
    formal_config: FormalExecutionConfig,
) -> RegimeThresholds:
    if not validation_rows:
        raise ValueError("regime thresholds require validation rows")
    pcs = np.asarray([row.pcs for row in validation_rows], dtype=np.float64)
    residuals = np.asarray(
        [float(np.mean(row.residuals)) for row in validation_rows],
        dtype=np.float64,
    )
    if not np.isfinite(pcs).all() or not np.isfinite(residuals).all():
        raise ValueError("regime threshold inputs must be finite")
    return RegimeThresholds(
        stable_pcs=float(np.quantile(pcs, formal_config.stable_pcs_quantile)),
        stable_residual=float(
            np.quantile(residuals, formal_config.stable_residual_quantile)
        ),
        invalid_pcs=float(np.quantile(pcs, formal_config.invalid_pcs_quantile)),
        invalid_residual=float(
            np.quantile(residuals, formal_config.invalid_residual_quantile)
        ),
    )


def infer_regime(row: FormalEvaluationRow, thresholds: RegimeThresholds) -> int:
    residual_burden = float(np.mean(row.residuals))
    if (
        not row.hard_feasible
        or row.pcs <= thresholds.invalid_pcs
        or residual_burden >= thresholds.invalid_residual
    ):
        return 2
    if (
        row.pcs >= thresholds.stable_pcs
        and residual_burden <= thresholds.stable_residual
    ):
        return 0
    return 1


def assign_regime_labels(
    rows: tuple[FormalEvaluationRow, ...],
    thresholds: RegimeThresholds,
) -> np.ndarray:
    if len(rows) < 3:
        raise ValueError("regime labeling requires at least three rows")
    labels = np.asarray([infer_regime(row, thresholds) for row in rows], dtype=np.int64)
    if set(labels.tolist()) == {0, 1, 2}:
        return labels
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            -(rows[index].pcs - float(np.mean(rows[index].residuals))),
            rows[index].task_id,
            rows[index].model_id,
            rows[index].replicate,
        ),
    )
    labels.fill(1)
    first_cut = max(1, len(rows) // 3)
    second_cut = min(len(rows) - 1, 2 * len(rows) // 3)
    labels[ranked[:first_cut]] = 0
    labels[ranked[second_cut:]] = 2
    return labels


def _training_config(config: RunConfig, trial: dict[str, float]) -> StateTrainingConfig:
    state = config.state_field
    return StateTrainingConfig(
        learning_rate=float(trial["learning_rate"]),
        weight_decay=float(trial["weight_decay"]),
        batch_size=state.batch_size,
        max_epochs=config.paper.max_epochs,
        patience=config.paper.patience,
        min_delta=state.early_stopping_min_delta,
        energy_loss_weight=float(trial["energy_loss_weight"]),
        energy_margin=float(trial["energy_margin"]),
        adamw_betas=state.adamw_betas,
        adamw_eps=state.adamw_eps,
        dimensions=state.dimensions,
        energy_dimensions=state.energy_dimensions,
        activation=state.activation,
        gelu_approximate=state.gelu_approximate,
        linear_bias=state.linear_bias,
        initialization=state.initialization,
        dropout=state.dropout,
        dropout_inplace=state.dropout_inplace,
        optimizer=state.optimizer,
        class_weighting=state.class_weighting,
        deterministic_shuffling=state.deterministic_shuffling,
    )


def train_state_models(
    training_rows: tuple[FormalEvaluationRow, ...],
    validation_rows: tuple[FormalEvaluationRow, ...],
    config: RunConfig,
    *,
    formal_config: FormalExecutionConfig | None = None,
) -> StateModelBundle:
    formal = formal_config or load_formal_config(
        PROJECT_ROOT / "configs" / "formal_execution.yaml"
    )
    thresholds = fit_regime_thresholds(validation_rows, formal)
    training_labels = assign_regime_labels(training_rows, thresholds)
    validation_labels = assign_regime_labels(validation_rows, thresholds)
    dataset = StateTrainingSet(
        train_states=np.stack([row.state_vector for row in training_rows]).astype(np.float64),
        train_labels=training_labels,
        validation_states=np.stack([row.state_vector for row in validation_rows]).astype(
            np.float64
        ),
        validation_labels=validation_labels,
    )
    checkpoints: list[StateModelCheckpoint] = []
    for seed in config.paper.seeds:
        trials = build_trials(
            {
                "energy_loss_weight": config.state_field.energy_loss_weight_grid,
                "energy_margin": config.state_field.energy_margin_grid,
                "learning_rate": config.state_field.learning_rate_grid,
                "weight_decay": config.state_field.weight_decay_grid,
            },
            count=config.paper.hyperparameter_trials,
            seed=seed,
        )
        if len({tuple(sorted(trial.items())) for trial in trials}) != len(trials):
            raise ValueError("state-field paper budget must contain 50 unique trials")
        candidates = []
        for trial_index, trial in enumerate(trials):
            result = train_state_field(
                dataset,
                seed=seed,
                config=_training_config(config, trial),
            )
            selected = next(
                metric
                for metric in result.epoch_metrics
                if metric.epoch == result.selected_epoch
            )
            candidates.append(
                (
                    selected.validation_loss,
                    -selected.validation_accuracy,
                    trial_index,
                    trial,
                    result,
                    selected,
                )
            )
        _, _, trial_index, trial, result, selected = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in result.model.state_dict().items()
        }
        checkpoints.append(
            StateModelCheckpoint(
                seed=seed,
                selected_trial=trial_index,
                trial_count=len(trials),
                learning_rate=float(trial["learning_rate"]),
                weight_decay=float(trial["weight_decay"]),
                energy_loss_weight=float(trial["energy_loss_weight"]),
                energy_margin=float(trial["energy_margin"]),
                selected_epoch=result.selected_epoch,
                validation_loss=selected.validation_loss,
                validation_accuracy=selected.validation_accuracy,
                state_dict_hash=result.state_dict_hash,
                state_dict=state_dict,
            )
        )
    return StateModelBundle(
        models=tuple(checkpoints),
        thresholds=thresholds,
        training_labels=training_labels,
        validation_labels=validation_labels,
    )
