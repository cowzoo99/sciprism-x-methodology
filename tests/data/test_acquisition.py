from hashlib import sha256
from pathlib import Path

import pytest


def test_acquisition_rejects_hash_mismatch_without_partial_file(tmp_path: Path) -> None:
    from sciprism_x.data.acquisition import SourceHashMismatch, acquire_source
    from sciprism_x.data.schema import SourceRecord

    source = SourceRecord(
        source_id="fixture",
        url="fixture://bytes",
        sha256="0" * 64,
        destination="raw/fixture.bin",
    )

    with pytest.raises(SourceHashMismatch, match="fixture"):
        acquire_source(source, tmp_path, fetch=lambda _: b"real bytes")

    assert not (tmp_path / "raw" / "fixture.bin").exists()
    assert not (tmp_path / "raw" / "fixture.bin.partial").exists()


def test_acquisition_writes_verified_payload_atomically(tmp_path: Path) -> None:
    from sciprism_x.data.acquisition import acquire_source
    from sciprism_x.data.schema import SourceRecord

    payload = b"official fixture bytes"
    source = SourceRecord(
        source_id="fixture",
        url="fixture://bytes",
        sha256=sha256(payload).hexdigest(),
        destination="raw/fixture.bin",
    )

    result = acquire_source(source, tmp_path, fetch=lambda _: payload)

    assert result == tmp_path / "raw" / "fixture.bin"
    assert result.read_bytes() == payload
    assert not result.with_suffix(".bin.partial").exists()
