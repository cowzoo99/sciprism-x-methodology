import numpy as np
import pytest


def test_paired_bootstrap_matches_direct_seeded_calculation() -> None:
    from sciprism_x.statistics.bootstrap import paired_bootstrap

    result = paired_bootstrap(
        pair_ids=("a", "b"),
        proposed=np.array([0.0, 4.0]),
        baseline=np.array([1.0, 1.0]),
        resamples=1000,
        confidence_level=0.95,
        seed=7,
        direction="higher",
    )

    differences = np.array([-1.0, 3.0])
    rng = np.random.default_rng(7)
    indices = rng.integers(0, 2, size=(1000, 2))
    expected = differences[indices].mean(axis=1)
    expected_p = min(
        1.0,
        2.0 * min(np.mean(expected <= 0.0), np.mean(expected >= 0.0)),
    )

    assert result.observed_difference == pytest.approx(1.0)
    assert result.resamples == 1000
    np.testing.assert_allclose(result.bootstrap_differences, expected)
    assert result.ci_lower == pytest.approx(np.quantile(expected, 0.025))
    assert result.ci_upper == pytest.approx(np.quantile(expected, 0.975))
    assert result.p_value == pytest.approx(expected_p)


def test_paired_bootstrap_resamples_unique_task_ids_not_rows() -> None:
    from sciprism_x.statistics.bootstrap import paired_bootstrap

    result = paired_bootstrap(
        pair_ids=("task-a", "task-a", "task-b", "task-b"),
        proposed=np.array([2.0, 4.0, 10.0, 14.0]),
        baseline=np.array([1.0, 1.0, 8.0, 8.0]),
        resamples=1000,
        confidence_level=0.95,
        seed=42,
        direction="higher",
    )

    assert result.pair_ids == ("task-a", "task-b")
    assert result.paired_differences == pytest.approx((2.0, 4.0))
    assert result.observed_difference == pytest.approx(3.0)


def test_lower_is_better_direction_is_normalized() -> None:
    from sciprism_x.statistics.bootstrap import paired_bootstrap

    result = paired_bootstrap(
        pair_ids=("a", "b"),
        proposed=np.array([0.2, 0.4]),
        baseline=np.array([0.3, 0.6]),
        resamples=1000,
        confidence_level=0.95,
        seed=42,
        direction="lower",
    )

    assert result.observed_difference == pytest.approx(0.15)
