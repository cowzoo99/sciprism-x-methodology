def test_shared_tuning_budget_is_repeatable_and_balanced() -> None:
    from sciprism_x.tuning import build_trials

    search_space = {
        "epsilon": (0.01, 0.03, 0.05, 0.10),
        "lambda_structure": (0.25, 0.50, 1.00, 2.00),
    }

    first = build_trials(search_space, count=50, seed=42)
    second = build_trials(search_space, count=50, seed=42)

    assert len(first) == 50
    assert first == second
    assert len({tuple(sorted(trial.items())) for trial in first[:16]}) == 16


def test_tuning_budget_cannot_drift_from_paper() -> None:
    from sciprism_x.tuning import build_trials

    try:
        build_trials({"epsilon": (0.01, 0.03)}, count=49, seed=42)
    except ValueError as error:
        assert str(error) == "method contract requires exactly 50 trials"
    else:
        raise AssertionError("49 trials must be rejected")
