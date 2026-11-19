from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import yaml

from .schema import RunHandle, RunReceipt


_HASH = re.compile(r"^[0-9a-f]{64}$")


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("artifact path must be relative and cannot contain '..'")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, cache_hash: str) -> RunHandle:
        if _HASH.fullmatch(cache_hash) is None:
            raise ValueError("cache_hash must be a lowercase SHA-256 digest")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        for _attempt in range(100):
            run_id = f"run-{timestamp}-{cache_hash[:12]}-{uuid.uuid4().hex[:8]}"
            staging = self.root / f".{run_id}.partial"
            final = self.root / run_id
            try:
                staging.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return RunHandle(run_id, cache_hash, staging, final)
        raise FileExistsError("could not allocate a unique run directory")

    def _target(self, handle: RunHandle, relative: str) -> Path:
        path = _relative_path(relative)
        target = handle.staging_path.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_bytes(self, handle: RunHandle, relative: str, payload: bytes) -> Path:
        target = self._target(handle, relative)
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return target

    def write_text(self, handle: RunHandle, relative: str, text: str) -> Path:
        return self.write_bytes(handle, relative, text.encode("utf-8"))

    def write_json(self, handle: RunHandle, relative: str, value: Any) -> Path:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        return self.write_text(handle, relative, text)

    def write_yaml(self, handle: RunHandle, relative: str, value: Any) -> Path:
        text = yaml.safe_dump(value, sort_keys=True, allow_unicode=True)
        return self.write_text(handle, relative, text)

    def write_jsonl(self, handle: RunHandle, relative: str, rows: list[dict[str, Any]]) -> Path:
        text = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        )
        return self.write_text(handle, relative, text)

    def write_parquet(self, handle: RunHandle, relative: str, table: pa.Table) -> Path:
        sink = pa.BufferOutputStream()
        parquet.write_table(
            table,
            sink,
            compression="zstd",
            version="2.6",
            use_dictionary=False,
            write_statistics=True,
        )
        return self.write_bytes(handle, relative, sink.getvalue().to_pybytes())

    def write_csv(
        self,
        handle: RunHandle,
        relative: str,
        rows: list[dict[str, Any]],
        fieldnames: tuple[str, ...],
    ) -> Path:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return self.write_text(handle, relative, stream.getvalue())

    def write_numpy(self, handle: RunHandle, relative: str, value: np.ndarray) -> Path:
        stream = io.BytesIO()
        np.save(stream, np.asarray(value), allow_pickle=False)
        return self.write_bytes(handle, relative, stream.getvalue())

    def _artifact_hashes(self, handle: RunHandle) -> tuple[tuple[str, str], ...]:
        return tuple(
            (path.relative_to(handle.staging_path).as_posix(), _sha256(path))
            for path in sorted(handle.staging_path.rglob("*"))
            if path.is_file() and path.name != "run_receipt.json"
        )

    def finalize(self, handle: RunHandle, *, track: Literal["A", "B"]) -> RunReceipt:
        if handle.final_path.exists():
            raise FileExistsError(f"run directory already exists: {handle.final_path}")
        hashes = self._artifact_hashes(handle)
        receipt = RunReceipt(
            run_id=handle.run_id,
            cache_hash=handle.cache_hash,
            track=track,
            final_path=handle.final_path,
            artifact_hashes=hashes,
        )
        self.write_json(
            handle,
            "run_receipt.json",
            {
                "run_id": receipt.run_id,
                "cache_hash": receipt.cache_hash,
                "track": receipt.track,
                "final_path": receipt.final_path.as_posix(),
                "artifact_hashes": dict(receipt.artifact_hashes),
            },
        )
        os.replace(handle.staging_path, handle.final_path)
        return receipt

    def abort(self, handle: RunHandle) -> None:
        staging = handle.staging_path.resolve()
        if staging.parent != self.root or not staging.name.endswith(".partial"):
            raise ValueError("refusing to remove a path outside the artifact root")
        if staging.exists():
            shutil.rmtree(staging)
