from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from .schema import SourceRecord


class SourceHashMismatch(ValueError):
    def __init__(self, source_id: str, expected: str, actual: str) -> None:
        super().__init__(
            f"source {source_id!r} SHA-256 mismatch: expected {expected}, got {actual}"
        )
        self.source_id = source_id
        self.expected = expected
        self.actual = actual


def acquire_source(
    source: SourceRecord,
    root: Path,
    fetch: Callable[[str], bytes],
) -> Path:
    payload = fetch(source.url)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != source.sha256:
        raise SourceHashMismatch(source.source_id, source.sha256, actual)

    target = root / Path(source.destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        temporary.write_bytes(payload)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def fetch_url(url: str, timeout_seconds: float = 120.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sciprism-x-methodology/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def load_registry(path: Path) -> tuple[SourceRecord, ...]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError("source registry must contain a 'sources' list")
    return tuple(SourceRecord.model_validate(item) for item in payload["sources"])


def verify_present_sources(
    records: Sequence[SourceRecord],
    root: Path,
) -> dict[str, str]:
    results: dict[str, str] = {}
    for record in records:
        target = root / Path(record.destination)
        if not target.exists():
            results[record.source_id] = "ABSENT"
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != record.sha256:
            raise SourceHashMismatch(record.source_id, record.sha256, actual)
        results[record.source_id] = "VERIFIED"
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    records = load_registry(arguments.registry)
    if arguments.verify_only:
        print(json.dumps(verify_present_sources(records, arguments.root), indent=2))
        return 0
    for record in records:
        acquire_source(record, arguments.root, fetch=fetch_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
