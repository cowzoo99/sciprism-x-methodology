import math

import numpy as np
import pandas as pd
import pytest

from sciprism_x.formal.results import REQUIRED_FORMAL_ARTIFACTS
from sciprism_x.formal.workflow import (
    FORMAL_STAGE_ORDER,
    _aggregate_metrics,
    _baseline_case_scicu,
    _select_open_set_frames,
    build_parser,
)
from sciprism_x.results.schema import FormalRunReceipt
from sciprism_x.responses.reconstruct import fixture_cases
from sciprism_x.responses.schema import ResponseRecord

from .helpers import smoke_contract


def test_formal_cli_defaults_to_paper() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/track_b.yaml",
            "--formal-config",
            "configs/formal_execution.yaml",
            "--output",
            "runs/formal",
        ]
    )

    assert args.scale == "paper"
    assert args.resume is True


def test_smoke_workflow_executes_every_stage(
    formal_smoke_receipt: FormalRunReceipt,
) -> None:
    receipt = formal_smoke_receipt
    assert receipt.completed_stages == FORMAL_STAGE_ORDER
    assert receipt.computed_response_count == smoke_contract().scicu_response_count
    assert receipt.applicable_metric_cells > 0
    assert receipt.nonfinite_metric_cells == 0
    assert receipt.paper_sources_in_computed_rows == 0
    assert set(dict(receipt.artifact_hashes)) == set(REQUIRED_FORMAL_ARTIFACTS)
    metrics = pd.read_parquet(receipt.final_path / "computed/per_case_metrics.parquet")
    tuning = pd.read_parquet(receipt.final_path / "computed/tuning_metrics.parquet")
    assert len(metrics) == smoke_contract().scicu_response_count
    assert all(math.isfinite(value) for value in metrics["pcs"])
    assert "source" in metrics
    assert set(metrics["source"]) == {"computed"}
    assert set(tuning["split"]) == {"train", "validation"}
    assert set(tuning["original_task_id"]).isdisjoint(
        set(metrics["original_task_id"])
    )


def test_aggregate_metrics_reports_raw_bsi_not_stability_complement() -> None:
    metrics = pd.DataFrame(
        {
            "tsr": [0.5, 1.0],
            "pcs": [0.4, 0.8],
            "ccs": [0.3, 0.7],
            "residual_consistency": [0.2, 0.6],
        }
    )
    boundaries = pd.DataFrame(
        {"bsi_raw": [0.2, 0.4], "bsi_stability": [0.8, 0.6]}
    )
    open_set = pd.DataFrame({"updq": [0.6], "structural_calibration": [0.7]})
    dbp = pd.DataFrame({"scs": [0.8, 0.6], "transport_purity": [0.9, 0.7]})

    aggregate = _aggregate_metrics(metrics, boundaries, open_set, dbp)

    assert aggregate["BSI"] == pytest.approx(0.3)
    assert aggregate["ccs"] == pytest.approx(0.5)
    assert aggregate["UPDQ"] == pytest.approx(0.6)
    assert aggregate["SC"] == pytest.approx(0.7)
    assert aggregate["SCS"] == pytest.approx(0.7)
    assert aggregate["TP"] == pytest.approx(0.8)


def test_open_set_validation_and_unknown_cases_are_disjoint() -> None:
    rows = []
    regimes = ("iid", "motif_or_structure", "domain_transfer", "boundary_stress")
    for index in range(500):
        rows.append(
            {
                "original_task_id": f"case-{index:04d}",
                "regime": regimes[index % len(regimes)],
            }
        )
    frame = pd.DataFrame(rows)

    validation, unknown = _select_open_set_frames(frame, (0.70, 0.15, 0.15))

    assert set(validation["original_task_id"]).isdisjoint(
        set(unknown["original_task_id"])
    )
    assert set(validation["regime"]).issubset({"iid", "motif_or_structure"})
    assert set(unknown["regime"]).issubset(set(regimes))
    assert set(unknown["split"]) == {"test"}


def test_scicu_baseline_case_uses_only_raw_allowed_inputs() -> None:
    case = fixture_cases(1)[0]
    response = ResponseRecord(
        task_id=case.task_id,
        model_id="model-a",
        replicate=0,
        response_text=case.reference_answer,
    )
    support = np.vstack((np.zeros(256), np.ones(256)))

    projected = _baseline_case_scicu(
        {"endpoint_success": True},
        case,
        response,
        support,
        representation_dimensions=256,
    )

    assert projected.predicted_target == case.reference_answer
    assert projected.representation.shape == (256,)
    assert projected.validation_support is support
    assert projected.residuals.tolist() == [0.0] * 6
    assert not projected.transport_plan.any()
