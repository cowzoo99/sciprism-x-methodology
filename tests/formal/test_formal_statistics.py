from dataclasses import replace
import math
from pathlib import Path

import pandas as pd
import pytest

from sciprism_x.formal.statistics import (
    ComputedPairRow,
    build_discrepancy_report,
    compute_formal_statistics,
    write_computed_rows,
)

from .helpers import run_config


def test_main_pairs_cover_scicueval_and_dbp_regimes() -> None:
    from sciprism_x.formal.workflow import _build_main_statistic_pairs

    scicu = pd.DataFrame(
        [
            {
                "task_id": "scicueval::iid::00000",
                "original_task_id": "task-a",
                "model_id": "model-a",
                "replicate": 0,
                "regime": "iid",
                "tsr": 1.0,
                "pcs": 0.8,
                "ccs": 0.75,
                "residual_consistency": 0.7,
            }
        ]
    )
    dbp = pd.DataFrame(
        [
            {
                "allocation_id": "dbp15k::motif_or_structure::ZH-EN::00000",
                "language_pair": "ZH-EN",
                "source_id": 1,
                "target_id": 2,
                "regime": "motif_or_structure",
                "scs": 0.75,
                "transport_purity": 0.65,
            }
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "case_id": "scicueval::iid::00000::model-a::0",
                "dataset": "scicueval",
                "baseline": "structure_aware",
                "metric": "alignment_similarity",
                "score": 0.6,
            },
            {
                "case_id": "dbp15k::motif_or_structure::ZH-EN::00000::ZH-EN::1::2",
                "dataset": "dbp15k",
                "baseline": "sinkhorn",
                "metric": "alignment_similarity",
                "score": 0.5,
            },
            {
                "case_id": "dbp15k::motif_or_structure::ZH-EN::00000::ZH-EN::1::2",
                "dataset": "dbp15k",
                "baseline": "sinkhorn",
                "metric": "TP",
                "score": 0.45,
            },
        ]
    )

    empty_boundaries = pd.DataFrame(
        columns=["regime", "case_id", "family", "bsi_raw", "baseline"]
    )
    empty_open_set = pd.DataFrame(columns=["regime"])
    pairs = _build_main_statistic_pairs(
        scicu,
        baselines,
        empty_boundaries,
        empty_boundaries,
        empty_open_set,
        empty_open_set,
        dbp,
        baseline_task_success_threshold=0.5,
    )

    assert {(row.dataset, row.regime, row.metric) for row in pairs} == {
        ("scicueval", "iid", "TSR"),
        ("scicueval", "iid", "PCS"),
        ("scicueval", "iid", "CCS"),
        ("dbp15k", "motif_or_structure", "SCS"),
        ("dbp15k", "motif_or_structure", "TP"),
    }
    assert {row.pair_id for row in pairs} == {"task-a", "ZH-EN::1::2"}
    assert all(row.family.startswith("manuscript_") for row in pairs)
    assert not [row for row in pairs if not row.reference_path]


def _computed_pairs() -> tuple[ComputedPairRow, ...]:
    return tuple(
        ComputedPairRow(
            pair_id=pair_id,
            dataset="scicueval",
            regime="iid",
            metric="PCS",
            proposed_method="sciprism_x",
            comparison_method=baseline,
            proposed_value=proposed,
            comparison_value=comparison,
            direction="higher",
            family="manuscript_table_4",
            reference_path=("statistics_fixture", baseline),
        )
        for baseline, values in {
            "endpoint": (("case-1", 0.8, 0.4), ("case-2", 0.6, 0.3)),
            "llm_judge": (("case-1", 0.8, 0.7), ("case-2", 0.6, 0.5)),
        }.items()
        for pair_id, proposed, comparison in values
    )


def test_computed_statistics_use_current_paired_rows() -> None:
    result = compute_formal_statistics(_computed_pairs(), run_config())

    assert len(result.bootstrap_rows) == 2
    assert all(row.source == "computed" for row in result.bootstrap_rows)
    assert all(row.resamples == 1000 for row in result.bootstrap_rows)
    assert all(row.confidence_level == 0.95 for row in result.bootstrap_rows)
    assert all(row.seed == 42 for row in result.bootstrap_rows)
    assert all(math.isfinite(row.p_value) for row in result.bootstrap_rows)
    assert {
        row.comparison_method: row.observed_difference
        for row in result.bootstrap_rows
    } == pytest.approx({"endpoint": 0.35, "llm_judge": 0.1})


def test_paper_reference_cannot_enter_computed_namespace(tmp_path: Path) -> None:
    result = compute_formal_statistics(_computed_pairs(), run_config())
    contaminated = replace(result.bootstrap_rows[0], source="paper_reference")

    with pytest.raises(ValueError, match="paper reference"):
        write_computed_rows(tmp_path / "computed", (contaminated,))


def test_holm_family_contains_complete_table_comparisons() -> None:
    result = compute_formal_statistics(_computed_pairs(), run_config())
    comparison_ids = tuple(sorted(row.comparison_id for row in result.bootstrap_rows))

    assert len(result.significance_rows) == 2
    assert all(row.family_members == comparison_ids for row in result.significance_rows)
    assert all(row.source == "computed" for row in result.significance_rows)
    assert all(0.0 <= row.adjusted_p_value <= 1.0 for row in result.significance_rows)


def test_discrepancy_reads_reference_only_after_computed_statistics(
    tmp_path: Path,
) -> None:
    result = compute_formal_statistics(_computed_pairs(), run_config())
    reference = tmp_path / "published_targets.json"
    reference.write_text(
        '{"statistics_fixture":{"endpoint":0.30,"llm_judge":0.05}}\n',
        encoding="utf-8",
    )

    rows = build_discrepancy_report(result, reference)
    by_method = {row.comparison_method: row for row in rows}

    assert by_method["endpoint"].computed_value == pytest.approx(0.35)
    assert by_method["endpoint"].absolute_difference == pytest.approx(0.05)
    assert by_method["endpoint"].relative_difference == pytest.approx(1.0 / 6.0)
    assert all(row.source == "paper_reference_comparison" for row in rows)
