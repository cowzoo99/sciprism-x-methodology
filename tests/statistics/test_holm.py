import pytest


def test_holm_adjustment_matches_hand_calculation() -> None:
    from sciprism_x.statistics.multiple_testing import holm_bonferroni

    results = holm_bonferroni(
        comparisons=(("a", 0.01), ("b", 0.04), ("c", 0.03)),
        alpha=0.05,
    )
    by_id = {result.comparison_id: result for result in results}

    assert by_id["a"].adjusted_p == pytest.approx(0.03)
    assert by_id["b"].adjusted_p == pytest.approx(0.06)
    assert by_id["c"].adjusted_p == pytest.approx(0.06)
    assert by_id["a"].reject is True
    assert by_id["b"].reject is False
    assert by_id["c"].reject is False


def test_holm_adjusted_values_are_monotone_in_sorted_raw_p() -> None:
    from sciprism_x.statistics.multiple_testing import holm_bonferroni

    results = holm_bonferroni(
        comparisons=(("x", 0.001), ("y", 0.02), ("z", 0.5), ("w", 0.04)),
        alpha=0.05,
    )
    ordered = sorted(results, key=lambda result: result.raw_p)

    assert [result.adjusted_p for result in ordered] == sorted(
        result.adjusted_p for result in ordered
    )
