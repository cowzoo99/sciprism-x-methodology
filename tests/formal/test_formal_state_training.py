import numpy as np

from sciprism_x.formal.evaluator import FormalEvaluationRow
from sciprism_x.formal.state_training import train_state_models

from .helpers import run_config


def _rows(count: int, namespace: str) -> tuple[FormalEvaluationRow, ...]:
    rows = []
    for index in range(count):
        regime = index % 3
        pcs = (0.9, 0.5, 0.1)[regime]
        residual = (0.1, 0.5, 0.9)[regime]
        state = np.linspace(0.0, 1.0, 15, dtype=np.float64)
        state = state + regime * 0.5 + index * 1.0e-4
        rows.append(
            FormalEvaluationRow(
                task_id=f"{namespace}-{index:03d}",
                model_id="fixture-model",
                replicate=0,
                endpoint_success=regime == 0,
                tsr=float(regime == 0),
                pcs=pcs,
                residual_consistency=1.0 - residual,
                node_loss=residual,
                structure_loss=residual,
                constraint_loss=residual,
                residuals=np.full(6, residual, dtype=np.float64),
                hard_feasible=regime != 2,
                transport_converged=regime != 2,
                state_vector=state,
                diagnostic_reason="synthetic_training_row",
            )
        )
    return tuple(rows)


def test_state_training_produces_all_five_seed_checkpoints() -> None:
    config = run_config()
    bundle = train_state_models(_rows(18, "train"), _rows(9, "validation"), config)

    assert tuple(model.seed for model in bundle.models) == (42, 0, 1, 123, 1234)
    assert all(model.selected_epoch <= 100 for model in bundle.models)
    assert all(model.trial_count == 50 for model in bundle.models)
    assert all(np.isfinite(model.validation_loss) for model in bundle.models)
    assert all(model.state_dict for model in bundle.models)
    assert set(bundle.training_labels.tolist()) == {0, 1, 2}
    assert set(bundle.validation_labels.tolist()) == {0, 1, 2}
