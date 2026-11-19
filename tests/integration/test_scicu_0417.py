import json
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "scicu_0417.json"


@dataclass(frozen=True)
class PublishedWorkedExample:
    case_id: str
    motif_statuses: tuple[str, ...]
    transport_masses: tuple[float, ...]
    residuals: tuple[float, ...]
    pcs: float
    bsi: float
    endpoint_indicator: float
    regime: str


def evaluate_fixture(path: Path, *, seed: int) -> PublishedWorkedExample:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert seed in payload["paper_seeds"]
    motifs = payload["published_observations"]["motifs"]
    observations = payload["published_observations"]
    return PublishedWorkedExample(
        case_id=payload["case_id"],
        motif_statuses=tuple(item["status"] for item in motifs),
        transport_masses=tuple(float(item["transport_mass"]) for item in motifs),
        residuals=tuple(float(value) for value in observations["residuals"]),
        pcs=float(observations["pcs"]),
        bsi=float(observations["bsi"]),
        endpoint_indicator=float(observations["endpoint_indicator"]),
        regime=str(observations["regime"]),
    )


def test_scicu_0417_published_fixture_fields() -> None:
    result = evaluate_fixture(FIXTURE, seed=42)

    assert result.case_id == "SCICU-0417"
    assert result.motif_statuses == (
        "MET",
        "MET",
        "MET",
        "MET",
        "PARTIALLY_MET",
        "UNMET",
    )
    assert result.transport_masses == pytest.approx(
        (0.72, 0.65, 0.44, 0.52, 0.28, 0.00), abs=1e-12
    )
    assert result.residuals == pytest.approx(
        (0.08, 0.06, 0.04, 0.11, 0.35, 0.42), abs=1e-12
    )
    assert result.pcs == pytest.approx(0.61, abs=1e-12)
    assert result.bsi == pytest.approx(0.28, abs=1e-12)
    assert result.endpoint_indicator == pytest.approx(0.93, abs=1e-12)
    assert result.regime == "BRITTLE"


def test_scicu_0417_fixture_matches_frozen_published_target_manifest() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    published = json.loads(
        (PROJECT_ROOT / "data" / "manifests" / "published_targets.json").read_text(
            encoding="utf-8"
        )
    )["worked_case"]
    observations = fixture["published_observations"]

    assert fixture["case_id"] == published["case_id"]
    assert observations["endpoint_indicator"] == published["endpoint_score"]
    assert observations["pcs"] == published["pcs"]
    assert observations["bsi"] == published["bsi"]
    assert observations["regime"] == published["regime"]
    assert [item["transport_mass"] for item in observations["motifs"]] == [
        item["transport_mass"] for item in published["motifs"].values()
    ]
    assert observations["residuals"] == list(published["residuals"].values())
