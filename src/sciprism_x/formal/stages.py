from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from sciprism_x.results.schema import StageReceipt


_HASH = re.compile(r"^[0-9a-f]{64}$")
_STAGE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_cache_key(
    name: str,
    input_hashes: Mapping[str, str],
    implementation_hash: str,
) -> str:
    if _STAGE_NAME.fullmatch(name) is None:
        raise ValueError("stage name must be lowercase snake_case")
    if _HASH.fullmatch(implementation_hash) is None:
        raise ValueError("implementation_hash must be a lowercase SHA-256 digest")
    normalized = tuple(sorted((str(key), str(value)) for key, value in input_hashes.items()))
    if not normalized or any(not key or _HASH.fullmatch(value) is None for key, value in normalized):
        raise ValueError("input_hashes must map non-empty names to SHA-256 digests")
    return canonical_sha256(
        {
            "name": name,
            "inputs": dict(normalized),
            "implementation": implementation_hash,
        }
    )


def _artifact_hashes(path: Path) -> tuple[tuple[str, str], ...]:
    files = []
    for artifact in sorted(path.rglob("*")):
        if artifact.is_symlink():
            raise ValueError("stage artifacts cannot contain symbolic links")
        if artifact.is_file() and artifact.name != "stage_receipt.json":
            files.append((artifact.relative_to(path).as_posix(), _file_sha256(artifact)))
    if not files:
        raise ValueError("stage producer wrote no artifacts")
    return tuple(files)


def _content_hash(artifacts: tuple[tuple[str, str], ...]) -> str:
    return canonical_sha256({"artifacts": dict(artifacts)})


def _write_receipt(path: Path, receipt: StageReceipt) -> None:
    payload = {
        "name": receipt.name,
        "cache_key": receipt.cache_key,
        "content_hash": receipt.content_hash,
        "input_hashes": dict(receipt.input_hashes),
        "implementation_hash": receipt.implementation_hash,
        "artifact_hashes": dict(receipt.artifact_hashes),
    }
    target = path / "stage_receipt.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_receipt(path: Path) -> StageReceipt:
    payload = json.loads((path / "stage_receipt.json").read_text(encoding="utf-8"))
    return StageReceipt(
        name=str(payload["name"]),
        cache_key=str(payload["cache_key"]),
        content_hash=str(payload["content_hash"]),
        path=path,
        resumed=False,
        input_hashes=tuple(sorted(dict(payload["input_hashes"]).items())),
        implementation_hash=str(payload["implementation_hash"]),
        artifact_hashes=tuple(sorted(dict(payload["artifact_hashes"]).items())),
    )


def verify_stage_receipt(receipt: StageReceipt) -> None:
    path = receipt.path.resolve()
    if not path.is_dir() or path.name.endswith(".partial"):
        raise ValueError("stage receipt path is missing or incomplete")
    saved = _read_receipt(path)
    if (
        saved.name != receipt.name
        or saved.cache_key != receipt.cache_key
        or saved.content_hash != receipt.content_hash
        or saved.input_hashes != receipt.input_hashes
        or saved.implementation_hash != receipt.implementation_hash
        or saved.artifact_hashes != receipt.artifact_hashes
    ):
        raise ValueError("stage receipt metadata does not match")
    actual_artifacts = _artifact_hashes(path)
    if actual_artifacts != saved.artifact_hashes:
        raise ValueError("stage artifact hashes do not match its receipt")
    if _content_hash(actual_artifacts) != saved.content_hash:
        raise ValueError("stage content hash does not match its receipt")


class StageStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_remove_partial(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.parent != self.root or not resolved.name.endswith(".partial"):
            raise ValueError("refusing to remove a path outside the stage store")
        if resolved.exists():
            shutil.rmtree(resolved)

    def verify(self, receipt: StageReceipt) -> None:
        path = receipt.path.resolve()
        if path.parent != self.root or not path.is_dir():
            raise ValueError("stage receipt path is outside its store or missing")
        verify_stage_receipt(receipt)

    def run(
        self,
        name: str,
        *,
        input_hashes: Mapping[str, str],
        implementation_hash: str,
        producer: Callable[[Path], object],
    ) -> StageReceipt:
        cache_key = stage_cache_key(name, input_hashes, implementation_hash)
        final = self.root / f"{name}-{cache_key}"
        if final.exists():
            saved = _read_receipt(final)
            expected_inputs = tuple(sorted(input_hashes.items()))
            if (
                saved.name != name
                or saved.cache_key != cache_key
                or saved.input_hashes != expected_inputs
                or saved.implementation_hash != implementation_hash
            ):
                raise ValueError("existing stage metadata does not match its cache key")
            self.verify(saved)
            return replace(saved, resumed=True)

        partial = self.root / f".{name}-{cache_key}.partial"
        self._safe_remove_partial(partial)
        partial.mkdir(parents=False, exist_ok=False)
        producer(partial)
        if (partial / "stage_receipt.json").exists():
            raise ValueError("stage producer cannot write the reserved stage_receipt.json")
        artifacts = _artifact_hashes(partial)
        for relative, _digest in artifacts:
            with (partial / Path(relative)).open("r+b") as stream:
                os.fsync(stream.fileno())
        receipt = StageReceipt(
            name=name,
            cache_key=cache_key,
            content_hash=_content_hash(artifacts),
            path=final,
            resumed=False,
            input_hashes=tuple(sorted(input_hashes.items())),
            implementation_hash=implementation_hash,
            artifact_hashes=artifacts,
        )
        _write_receipt(partial, receipt)
        os.replace(partial, final)
        return receipt
