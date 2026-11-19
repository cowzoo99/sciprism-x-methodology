import math

import numpy as np

from sciprism_x.formal.evaluator import build_formal_context, evaluate_scicu_response
from sciprism_x.responses.reconstruct import fixture_cases
from sciprism_x.responses.schema import ResponseRecord

from .helpers import run_config


def test_formal_evaluator_returns_finite_state_and_losses(tmp_path) -> None:
    case = fixture_cases(1)[0]
    response = ResponseRecord(
        task_id=case.task_id,
        model_id=run_config().generators.model_ids[0],
        replicate=0,
        response_text=case.reference_answer,
    )
    context = build_formal_context(
        run_config(),
        training_texts=(case.prompt, case.reference_context, case.reference_answer),
        model_root=tmp_path,
    )

    row = evaluate_scicu_response(case=case, response=response, context=context)

    assert row.state_vector.shape == (15,)
    assert np.isfinite(row.state_vector).all()
    assert math.isfinite(row.pcs)
    assert len(row.residuals) == 6
    assert np.isfinite(row.residuals).all()
    assert row.hard_feasible in {True, False}
    assert row.transport_converged in {True, False}


def test_formal_evaluator_is_repeatable(tmp_path) -> None:
    case = fixture_cases(1)[0]
    response = ResponseRecord(
        task_id=case.task_id,
        model_id="gpt-4o-2024-08-06",
        replicate=0,
        response_text=case.reference_answer,
    )
    context = build_formal_context(
        run_config(),
        training_texts=(case.prompt, case.reference_context, case.reference_answer),
        model_root=tmp_path,
    )

    first = evaluate_scicu_response(case=case, response=response, context=context)
    second = evaluate_scicu_response(case=case, response=response, context=context)

    np.testing.assert_array_equal(first.state_vector, second.state_vector)
    assert first.pcs == second.pcs
    np.testing.assert_array_equal(first.residuals, second.residuals)
