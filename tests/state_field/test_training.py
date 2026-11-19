import numpy as np
import torch


def test_state_model_shapes() -> None:
    from sciprism_x.state_field.model import StateFieldModel

    model = StateFieldModel(dtype=torch.float64)
    state = torch.zeros((4, 15), dtype=torch.float64)
    regime_logits, energy = model(state)

    assert regime_logits.shape == (4, 3)
    assert energy.shape == (4, 1)
    assert all(parameter.dtype is torch.float64 for parameter in model.parameters())


def _tiny_training_set():
    from sciprism_x.state_field.train import StateTrainingSet

    rng = np.random.default_rng(7)
    train_states = rng.normal(size=(18, 15)).astype(np.float64)
    train_labels = np.repeat(np.array([0, 1, 2], dtype=np.int64), 6)
    validation_states = rng.normal(size=(9, 15)).astype(np.float64)
    validation_labels = np.repeat(np.array([0, 1, 2], dtype=np.int64), 3)
    return StateTrainingSet(
        train_states=train_states,
        train_labels=train_labels,
        validation_states=validation_states,
        validation_labels=validation_labels,
    )


def _tiny_config():
    from sciprism_x.state_field.train import StateTrainingConfig

    return StateTrainingConfig(
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=6,
        max_epochs=8,
        patience=3,
        min_delta=0.0,
        energy_loss_weight=0.5,
        energy_margin=1.0,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1.0e-8,
    )


def test_training_repeats_under_seed_42() -> None:
    from sciprism_x.state_field.train import train_state_field

    first = train_state_field(_tiny_training_set(), seed=42, config=_tiny_config())
    second = train_state_field(_tiny_training_set(), seed=42, config=_tiny_config())

    assert first.state_dict_hash == second.state_dict_hash
    assert first.selected_epoch == second.selected_epoch
    assert first.epoch_metrics == second.epoch_metrics


def test_training_records_explicit_optimizer_controls() -> None:
    from sciprism_x.state_field.train import train_state_field

    result = train_state_field(_tiny_training_set(), seed=42, config=_tiny_config())

    assert result.optimizer_arguments == {
        "lr": 1.0e-3,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 1.0e-4,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    assert 1 <= result.selected_epoch <= 8
    assert result.epoch_metrics
