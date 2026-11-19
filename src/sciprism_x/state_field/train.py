from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from sciprism_x.reproducibility import seed_everything

from .model import StateFieldModel


@dataclass(frozen=True, slots=True)
class StateTrainingSet:
    train_states: np.ndarray
    train_labels: np.ndarray
    validation_states: np.ndarray
    validation_labels: np.ndarray

    def __post_init__(self) -> None:
        pairs = (
            (self.train_states, self.train_labels, "train"),
            (self.validation_states, self.validation_labels, "validation"),
        )
        for states, labels, name in pairs:
            if states.ndim != 2 or states.shape[1] != 15:
                raise ValueError(f"{name} states must have shape (n, 15)")
            if labels.shape != (states.shape[0],):
                raise ValueError(f"{name} labels must have shape (n,)")
            if not np.all(np.isfinite(states)):
                raise ValueError(f"{name} states must be finite")
            if not np.all(np.isin(labels, (0, 1, 2))):
                raise ValueError(f"{name} labels must be 0, 1, or 2")
        if set(np.asarray(self.train_labels).tolist()) != {0, 1, 2}:
            raise ValueError("training data must contain all three regimes")


@dataclass(frozen=True, slots=True)
class StateTrainingConfig:
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    energy_loss_weight: float
    energy_margin: float
    adamw_betas: tuple[float, float]
    adamw_eps: float
    dimensions: tuple[int, int, int] = (15, 64, 16)
    energy_dimensions: tuple[int, int, int] = (31, 32, 1)
    activation: str = "gelu"
    gelu_approximate: str = "none"
    linear_bias: bool = True
    initialization: str = "kaiming_uniform_linear_default"
    dropout: float = 0.0
    dropout_inplace: bool = False
    optimizer: str = "adamw"
    class_weighting: str = "inverse_frequency_mean_one"
    deterministic_shuffling: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.adamw_eps <= 0.0:
            raise ValueError("optimizer learning rate and epsilon must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight decay must be nonnegative")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("training counts must be positive")
        if self.min_delta < 0.0 or self.energy_loss_weight < 0.0:
            raise ValueError("loss controls must be nonnegative")
        if self.energy_margin <= 0.0:
            raise ValueError("energy margin must be positive")
        if self.activation != "gelu" or self.gelu_approximate != "none":
            raise ValueError("state field activation must be exact GELU")
        if self.linear_bias is not True or self.dropout_inplace is not False:
            raise ValueError("state field bias/dropout controls differ from the method contract")
        if self.initialization != "kaiming_uniform_linear_default":
            raise ValueError("unsupported state field initialization")
        if not all(0.0 <= beta < 1.0 for beta in self.adamw_betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.optimizer != "adamw":
            raise ValueError("paper state field requires AdamW")
        if self.class_weighting != "inverse_frequency_mean_one":
            raise ValueError("paper state field requires inverse-frequency class weighting")
        if self.deterministic_shuffling is not True:
            raise ValueError("paper state field requires deterministic shuffling")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    state_dict_hash: str
    selected_epoch: int
    epoch_metrics: tuple[EpochMetrics, ...]
    optimizer_arguments: dict[str, object]
    model: StateFieldModel = field(compare=False, repr=False)


def _class_weights(labels: Tensor) -> Tensor:
    counts = torch.bincount(labels, minlength=3).to(torch.float64)
    inverse = counts.sum() / (3.0 * counts)
    return inverse / inverse.mean()


def _energy_margin_loss(energy: Tensor, labels: Tensor, margin: float) -> Tensor:
    values = energy.reshape(-1)
    lower = labels[:, None] < labels[None, :]
    if not bool(lower.any()):
        return values.sum() * 0.0
    differences = values[None, :] - values[:, None]
    return torch.relu(margin - differences[lower]).mean()


def _combined_loss(
    logits: Tensor,
    energy: Tensor,
    labels: Tensor,
    *,
    class_weights: Tensor,
    energy_loss_weight: float,
    energy_margin: float,
) -> Tensor:
    classification = nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
    )
    margin = _energy_margin_loss(energy, labels, energy_margin)
    return classification + energy_loss_weight * margin


def _state_dict_hash(model: StateFieldModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def train_state_field(
    dataset: StateTrainingSet,
    *,
    seed: int,
    config: StateTrainingConfig,
) -> TrainingResult:
    seed_everything(seed, deterministic=True)
    model = StateFieldModel(
        dimensions=config.dimensions,
        energy_dimensions=config.energy_dimensions,
        activation=config.activation,
        gelu_approximate=config.gelu_approximate,
        linear_bias=config.linear_bias,
        initialization=config.initialization,
        dropout=config.dropout,
        dropout_inplace=config.dropout_inplace,
        dtype=torch.float64,
    )
    train_states = torch.as_tensor(dataset.train_states, dtype=torch.float64)
    train_labels = torch.as_tensor(dataset.train_labels, dtype=torch.int64)
    validation_states = torch.as_tensor(dataset.validation_states, dtype=torch.float64)
    validation_labels = torch.as_tensor(dataset.validation_labels, dtype=torch.int64)
    class_weights = _class_weights(train_labels)
    optimizer_arguments: dict[str, object] = {
        "lr": config.learning_rate,
        "betas": config.adamw_betas,
        "eps": config.adamw_eps,
        "weight_decay": config.weight_decay,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_arguments)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_states), generator=generator)
        total_train_loss = 0.0
        for start in range(0, len(train_states), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            states = train_states[indices]
            labels = train_labels[indices]
            optimizer.zero_grad(set_to_none=True)
            logits, energy = model(states)
            loss = _combined_loss(
                logits,
                energy,
                labels,
                class_weights=class_weights,
                energy_loss_weight=config.energy_loss_weight,
                energy_margin=config.energy_margin,
            )
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach()) * len(indices)

        model.eval()
        with torch.inference_mode():
            validation_logits, validation_energy = model(validation_states)
            validation_loss = float(
                _combined_loss(
                    validation_logits,
                    validation_energy,
                    validation_labels,
                    class_weights=class_weights,
                    energy_loss_weight=config.energy_loss_weight,
                    energy_margin=config.energy_margin,
                )
            )
            validation_accuracy = float(
                (validation_logits.argmax(dim=1) == validation_labels)
                .to(torch.float64)
                .mean()
            )
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=total_train_loss / len(train_states),
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
            )
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a selected epoch")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        state_dict_hash=_state_dict_hash(model),
        selected_epoch=best_epoch,
        epoch_metrics=tuple(history),
        optimizer_arguments=optimizer_arguments,
        model=model,
    )
