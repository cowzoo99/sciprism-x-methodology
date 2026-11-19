from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_committed_registry_uses_fixed_official_commits() -> None:
    from sciprism_x.data.acquisition import load_registry

    records = load_registry(PROJECT_ROOT / "data" / "manifests" / "source_registry.json")

    assert len(records) == 41
    assert len({record.source_id for record in records}) == 41
    assert all("423484f99541beed575b28645661afed86dd4ff9" in record.url for record in records[:-1])
    assert records[-1].source_id == "dbp15k-jape-0_3"
    assert "73729a2df0547d79ad07c2758d27c9873930adec" in records[-1].url


def test_verify_present_sources_reports_verified_and_absent(tmp_path: Path) -> None:
    from sciprism_x.data.acquisition import verify_present_sources
    from sciprism_x.data.schema import SourceRecord

    payload = b"verified"
    present = SourceRecord(
        source_id="present",
        url="fixture://present",
        sha256=sha256(payload).hexdigest(),
        destination="raw/present.bin",
    )
    absent = SourceRecord(
        source_id="absent",
        url="fixture://absent",
        sha256="0" * 64,
        destination="raw/absent.bin",
    )
    target = tmp_path / present.destination
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    assert verify_present_sources((present, absent), tmp_path) == {
        "present": "VERIFIED",
        "absent": "ABSENT",
    }


def test_fetch_url_reads_data_uri_without_network() -> None:
    from sciprism_x.data.acquisition import fetch_url

    assert fetch_url("data:application/octet-stream;base64,YWJj", timeout_seconds=1.0) == b"abc"


def test_verify_only_cli_reports_absent_sources(tmp_path: Path, capsys) -> None:
    from sciprism_x.data.acquisition import main

    registry = PROJECT_ROOT / "data" / "manifests" / "source_registry.json"
    exit_code = main(
        ["--registry", str(registry), "--root", str(tmp_path), "--verify-only"]
    )

    assert exit_code == 0
    assert '"dbp15k-jape-0_3": "ABSENT"' in capsys.readouterr().out
