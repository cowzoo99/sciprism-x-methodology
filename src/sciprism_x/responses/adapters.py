from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from sciprism_x.data.schema import SciCUCase
from sciprism_x.reproducibility import derive_seed

from .reconstruct import reconstruct_response
from .schema import ReconstructionProfile, ResponseAcquisitionRecord, ResponseRecord

ModelLoader = Callable[[SciCUCase, int], str]


class ResponseRecoveryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    cases: tuple[SciCUCase, ...]
    model_id: str
    responses_per_task: int
    seed: int
    reconstruction_profile: ReconstructionProfile | None
    supplied_file: Path | None = None
    exact_model_loader: ModelLoader | None = None
    exact_endpoint: str | None = None
    nearest_model_id: str | None = None
    nearest_model_loader: ModelLoader | None = None
    nearest_endpoint: str | None = None
    official_call_authorized: bool = False


class RecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[ResponseRecord, ...]
    acquisition: ResponseAcquisitionRecord


class RecoveryAdapter(Protocol):
    def available(self) -> bool: ...

    def load(self) -> tuple[ResponseRecord, ...]: ...

    def manifest_record(self, rows: tuple[ResponseRecord, ...]) -> ResponseAcquisitionRecord: ...


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows_hash(rows: tuple[ResponseRecord, ...]) -> str:
    payload = [row.model_dump(mode="json") for row in rows]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _profile_hash(profile: ReconstructionProfile) -> str:
    encoded = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_rows(
    rows: tuple[ResponseRecord, ...],
    request: RecoveryRequest,
) -> tuple[ResponseRecord, ...]:
    expected = {
        (case.task_id, replicate)
        for case in request.cases
        for replicate in range(request.responses_per_task)
    }
    observed = {(row.task_id, row.replicate) for row in rows}
    if observed != expected:
        raise ResponseRecoveryError("response rows do not match requested task/replicate coverage")
    if any(row.model_id != request.model_id for row in rows):
        raise ResponseRecoveryError("response rows contain an unexpected model_id")
    return tuple(sorted(rows, key=lambda row: (row.task_id, row.replicate)))


def _read_response_file(path: Path) -> tuple[ResponseRecord, ...]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = tuple(
            ResponseRecord.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ResponseRecoveryError("JSON response input must contain a list")
        rows = tuple(ResponseRecord.model_validate(item) for item in payload)
    elif suffix == ".parquet":
        import pyarrow.parquet as parquet

        rows = tuple(
            ResponseRecord.model_validate(item)
            for item in parquet.read_table(path).to_pylist()
        )
    else:
        raise ResponseRecoveryError(f"unsupported response input format: {suffix}")
    if not rows:
        raise ResponseRecoveryError("response input is empty")
    return rows


class SuppliedFileAdapter:
    def __init__(self, request: RecoveryRequest) -> None:
        self.request = request

    def available(self) -> bool:
        return self.request.supplied_file is not None and self.request.supplied_file.is_file()

    def load(self) -> tuple[ResponseRecord, ...]:
        assert self.request.supplied_file is not None
        return _validate_rows(_read_response_file(self.request.supplied_file), self.request)

    def manifest_record(self, rows: tuple[ResponseRecord, ...]) -> ResponseAcquisitionRecord:
        assert self.request.supplied_file is not None
        return ResponseAcquisitionRecord(
            route="supplied_file",
            requested_model_id=self.request.model_id,
            resolved_model_id=self.request.model_id,
            endpoint=None,
            source_sha256=_sha256_bytes(self.request.supplied_file.read_bytes()),
            fit_sha256=None,
            settings={
                "file_role": "supplied_response_dump",
                "path_scope": "external_supplied_input",
            },
        )


class _ModelAdapter:
    route: str

    def __init__(
        self,
        request: RecoveryRequest,
        *,
        loader: ModelLoader | None,
        resolved_model_id: str | None,
        endpoint: str | None,
    ) -> None:
        self.request = request
        self.loader = loader
        self.resolved_model_id = resolved_model_id
        self.endpoint = endpoint

    def available(self) -> bool:
        return (
            self.request.official_call_authorized
            and self.loader is not None
            and self.resolved_model_id is not None
            and self.endpoint is not None
        )

    def load(self) -> tuple[ResponseRecord, ...]:
        if self.loader is None:
            raise ResponseRecoveryError("model loader is unavailable")
        rows = tuple(
            ResponseRecord(
                task_id=case.task_id,
                model_id=self.request.model_id,
                replicate=replicate,
                response_text=self.loader(case, replicate),
            )
            for case in self.request.cases
            for replicate in range(self.request.responses_per_task)
        )
        return _validate_rows(rows, self.request)

    def manifest_record(self, rows: tuple[ResponseRecord, ...]) -> ResponseAcquisitionRecord:
        return ResponseAcquisitionRecord(
            route=self.route,
            requested_model_id=self.request.model_id,
            resolved_model_id=self.resolved_model_id or self.request.model_id,
            endpoint=self.endpoint,
            source_sha256=_rows_hash(rows),
            fit_sha256=None,
            settings={
                "responses_per_task": self.request.responses_per_task,
                "temperature": 0.0,
                "max_output_tokens": 2048,
            },
        )


class ExactOfficialModelAdapter(_ModelAdapter):
    route = "exact_official_model"

    def __init__(self, request: RecoveryRequest) -> None:
        super().__init__(
            request,
            loader=request.exact_model_loader,
            resolved_model_id=request.model_id,
            endpoint=request.exact_endpoint,
        )


class NearestOfficialModelAdapter(_ModelAdapter):
    route = "nearest_official_model"

    def __init__(self, request: RecoveryRequest) -> None:
        super().__init__(
            request,
            loader=request.nearest_model_loader,
            resolved_model_id=request.nearest_model_id,
            endpoint=request.nearest_endpoint,
        )


class DeterministicReconstructionAdapter:
    def __init__(self, request: RecoveryRequest) -> None:
        self.request = request

    def available(self) -> bool:
        return self.request.reconstruction_profile is not None

    def load(self) -> tuple[ResponseRecord, ...]:
        profile = self.request.reconstruction_profile
        if profile is None:
            raise ResponseRecoveryError("response profile is unavailable")
        rows = tuple(
            reconstruct_response(
                case,
                profile,
                seed=derive_seed(
                    self.request.seed,
                    f"response:{self.request.model_id}:{case.task_id}:{replicate}",
                ),
                model_id=self.request.model_id,
                replicate=replicate,
            )
            for case in self.request.cases
            for replicate in range(self.request.responses_per_task)
        )
        return _validate_rows(rows, self.request)

    def manifest_record(self, rows: tuple[ResponseRecord, ...]) -> ResponseAcquisitionRecord:
        profile = self.request.reconstruction_profile
        assert profile is not None
        return ResponseAcquisitionRecord(
            route="deterministic_reconstruction",
            requested_model_id=self.request.model_id,
            resolved_model_id=self.request.model_id,
            endpoint=None,
            source_sha256=_rows_hash(rows),
            fit_sha256=_profile_hash(profile),
            settings={
                "base_seed": self.request.seed,
                "responses_per_task": self.request.responses_per_task,
                "profile": profile.model_dump(mode="json"),
            },
        )


def recover_responses(request: RecoveryRequest) -> RecoveryResult:
    if request.responses_per_task < 1:
        raise ResponseRecoveryError("responses_per_task must be positive")
    if not request.cases:
        raise ResponseRecoveryError("at least one case is required")
    adapters: tuple[RecoveryAdapter, ...] = (
        SuppliedFileAdapter(request),
        ExactOfficialModelAdapter(request),
        NearestOfficialModelAdapter(request),
        DeterministicReconstructionAdapter(request),
    )
    for adapter in adapters:
        if adapter.available():
            rows = adapter.load()
            return RecoveryResult(rows=rows, acquisition=adapter.manifest_record(rows))
    raise ResponseRecoveryError(f"no response recovery route is available for {request.model_id}")
