from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import ValidationPair, ValidationResult, pair_evidence

SCIBERT_REVISION = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
BART_MNLI_REVISION = "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"


class FrozenModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: dict[str, str] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def validate_file_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in value.values()
        ):
            raise ValueError("every model file requires a lowercase SHA-256")
        return value


class FrozenEntailmentValidator:
    route = "frozen_entailment"
    validator_id = "frozen_entailment_fallback"

    def __init__(
        self,
        *,
        manifest: FrozenModelManifest,
        model_dir: Path,
        pass_threshold: float,
        fail_threshold: float,
        max_length: int = 512,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        if not 0.0 <= fail_threshold <= 1.0:
            raise ValueError("fail_threshold must be in [0, 1]")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.manifest = manifest
        self.model_dir = model_dir
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _verify_files(self) -> str | None:
        if not self.model_dir.is_dir():
            return "model_files_missing"
        for relative, expected_hash in self.manifest.files.items():
            path = self.model_dir / relative
            if not path.is_file():
                return "model_files_missing"
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected_hash:
                return "model_hash_mismatch"
        return None

    def verify_assets(self) -> None:
        error = self._verify_files()
        if error == "model_files_missing":
            raise FileNotFoundError("frozen BART-MNLI manifest references missing files")
        if error == "model_hash_mismatch":
            raise ValueError("frozen BART-MNLI file hash mismatch")

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        return self._tokenizer, self._model

    def validate(self, pair: ValidationPair) -> ValidationResult:
        availability_error = self._verify_files()
        if availability_error is not None:
            return ValidationResult.unavailable(
                availability_error,
                validator_id=self.validator_id,
            )
        try:
            tokenizer, model = self._load()
        except (OSError, RuntimeError, ValueError):
            return ValidationResult.unavailable(
                "model_load_failed",
                validator_id=self.validator_id,
            )

        encoded = tokenizer(
            pair.task_node.text,
            pair.response_node.text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits[0], dim=-1)
        labels = {
            str(label).lower(): int(index)
            for index, label in model.config.id2label.items()
        }
        entailment_index = next(
            (index for label, index in labels.items() if "entail" in label),
            None,
        )
        contradiction_index = next(
            (index for label, index in labels.items() if "contrad" in label),
            None,
        )
        if entailment_index is None or contradiction_index is None:
            return ValidationResult.unresolved(
                "model_labels_unrecognized",
                confidence=0.0,
                validator_id=self.validator_id,
            )
        entailment = float(probabilities[entailment_index])
        contradiction = float(probabilities[contradiction_index])
        if entailment >= self.pass_threshold:
            return ValidationResult.pass_(
                "learned_entailment",
                confidence=entailment,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        if contradiction >= self.fail_threshold:
            return ValidationResult.fail(
                "learned_contradiction",
                confidence=contradiction,
                evidence_spans=pair_evidence(pair),
                validator_id=self.validator_id,
            )
        return ValidationResult.unresolved(
            "learned_low_certainty",
            confidence=max(entailment, contradiction),
            evidence_spans=pair_evidence(pair),
            validator_id=self.validator_id,
        )
