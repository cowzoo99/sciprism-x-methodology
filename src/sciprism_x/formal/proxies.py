from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from sciprism_x.config import FrozenModelConfig, TfidfProxyConfig, ValidatorsConfig
from sciprism_x.transport.costs import ScientificTextEncoder
from sciprism_x.validators.learned import FrozenEntailmentValidator, FrozenModelManifest
from sciprism_x.validators.registry import Validator
from sciprism_x.validators.schema import ValidationPair, ValidationResult, pair_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_QUANTITY = re.compile(
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?"
    r"(?:\s*(?:m/s|km/h|V/m|m\^2|m|s|kg|K|Pa|N|J|W|T))?",
    re.IGNORECASE,
)
_ANCHOR = re.compile(
    r"\b(?:table|figure|fig\.?|graph|image|record|passage|reference|source)"
    r"(?:\s+[A-Za-z0-9.-]+)?",
    re.IGNORECASE,
)
_CONTRADICTION = re.compile(
    r"\b(?:not|never|unrelated|unsupported|without\s+evidence|regardless|"
    r"incompatible|contradict(?:s|ed|ory)?|false|opposite)\b",
    re.IGNORECASE,
)


class TfidfScientificEncoder:
    route = "tfidf_proxy"

    def __init__(self, vectorizer: TfidfVectorizer) -> None:
        self.vectorizer = vectorizer

    @classmethod
    def fit(
        cls,
        *,
        train_texts: Sequence[str],
        config: TfidfProxyConfig,
        test_texts: Sequence[str] = (),
    ) -> "TfidfScientificEncoder":
        if tuple(test_texts):
            raise ValueError("TF-IDF proxy may only fit the training split")
        training = tuple(text.strip() for text in train_texts if text.strip())
        if not training:
            raise ValueError("TF-IDF proxy requires non-empty training text")
        vectorizer = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="strict",
            strip_accents=config.strip_accents,
            lowercase=config.lowercase,
            preprocessor=None,
            tokenizer=None,
            analyzer=config.analyzer,
            stop_words=config.stop_words,
            token_pattern=config.token_pattern,
            ngram_range=config.ngram_range,
            max_df=config.max_df,
            min_df=config.min_df,
            max_features=config.max_features,
            vocabulary=None,
            binary=config.binary,
            dtype=np.dtype(config.dtype).type,
            norm=config.norm,
            use_idf=config.use_idf,
            smooth_idf=config.smooth_idf,
            sublinear_tf=config.sublinear_tf,
        )
        vectorizer.fit(training)
        return cls(vectorizer)

    def cosine(self, left: str, right: str) -> float:
        vectors = self.vectorizer.transform((left, right))
        similarity = float((vectors[0] @ vectors[1].T).toarray()[0, 0])
        return float(np.clip(similarity, 0.0, 1.0))


class FrozenScientificEncoder:
    route = "frozen_scibert"

    def __init__(
        self,
        *,
        manifest: FrozenModelManifest,
        model_dir: Path,
        max_length: int,
        pooling: str,
    ) -> None:
        if max_length < 1:
            raise ValueError("SciBERT max_length must be positive")
        if pooling != "attention_masked_mean":
            raise ValueError("unsupported frozen SciBERT pooling rule")
        self.manifest = manifest
        self.model_dir = model_dir
        self.max_length = max_length
        self.pooling = pooling
        self._tokenizer = None
        self._model = None
        self._verify_files()

    def _verify_files(self) -> None:
        for relative, expected in self.manifest.files.items():
            path = self.model_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f"frozen SciBERT file is missing: {relative}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"frozen SciBERT hash mismatch: {relative}")

    def _load(self):
        from transformers import AutoModel, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.model_dir,
                local_files_only=True,
            ).to("cpu")
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        return self._tokenizer, self._model

    def cosine(self, left: str, right: str) -> float:
        tokenizer, model = self._load()
        encoded = tokenizer(
            (left, right),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            similarity = float(torch.dot(pooled[0], pooled[1]))
        return float(np.clip(similarity, -1.0, 1.0))


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text)}


def _coverage(required: set[str], observed: set[str]) -> float:
    return 1.0 if not required else len(required & observed) / len(required)


def _token_coverage(pair: ValidationPair) -> float:
    return _coverage(_tokens(pair.task_node.text), _tokens(pair.response_node.text))


def _quantity_agreement(pair: ValidationPair) -> float:
    required = {match.group(0).lower().replace(" ", "") for match in _QUANTITY.finditer(pair.task_node.text)}
    observed = {match.group(0).lower().replace(" ", "") for match in _QUANTITY.finditer(pair.response_node.text)}
    return _coverage(required, observed)


def _equation_agreement(pair: ValidationPair) -> float:
    normalize = lambda value: re.sub(r"\s+", "", value).lower()
    required = {
        normalize(value)
        for value in pair.task_node.feature_evidence.get("equations", ())
    }
    observed = {
        normalize(value)
        for value in pair.response_node.feature_evidence.get("equations", ())
    }
    return _coverage(required, observed)


def _anchor_agreement(pair: ValidationPair) -> float:
    normalize = lambda value: re.sub(r"\s+", "", value).lower().rstrip(".,;:")
    required = {normalize(match.group(0)) for match in _ANCHOR.finditer(pair.task_node.text)}
    observed = {normalize(match.group(0)) for match in _ANCHOR.finditer(pair.response_node.text)}
    return _coverage(required, observed)


def _contradiction_score(text: str, count_scale: float) -> float:
    return min(1.0, len(_CONTRADICTION.findall(text)) / count_scale)


class DeterministicNLIProxy:
    route = "deterministic_nli_proxy"
    validator_id = route

    def __init__(
        self,
        *,
        pass_threshold: float,
        fail_threshold: float,
        feature_weights: tuple[float, float, float, float, float],
        contradiction_count_scale: float,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        if not 0.0 <= fail_threshold <= 1.0:
            raise ValueError("fail_threshold must be in [0, 1]")
        if any(weight < 0.0 for weight in feature_weights) or not np.isclose(
            sum(feature_weights), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("feature_weights must be nonnegative and sum to one")
        if contradiction_count_scale <= 0.0:
            raise ValueError("contradiction_count_scale must be positive")
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold
        self.feature_weights = feature_weights
        self.contradiction_count_scale = contradiction_count_scale

    def validate(self, pair: ValidationPair) -> ValidationResult:
        features = np.asarray(
            (
                _token_coverage(pair),
                _quantity_agreement(pair),
                _equation_agreement(pair),
                _anchor_agreement(pair),
                1.0 - _contradiction_score(
                    pair.response_node.text,
                    self.contradiction_count_scale,
                ),
            ),
            dtype=np.float64,
        )
        confidence = float(
            np.dot(features, np.asarray(self.feature_weights, dtype=np.float64))
        )
        evidence = pair_evidence(pair)
        if confidence >= self.pass_threshold:
            return ValidationResult.pass_(
                "proxy_entailment",
                confidence=confidence,
                evidence_spans=evidence,
                validator_id=self.route,
            )
        contradiction = 1.0 - confidence
        if contradiction >= self.fail_threshold:
            return ValidationResult.fail(
                "proxy_contradiction",
                confidence=contradiction,
                evidence_spans=evidence,
                validator_id=self.route,
            )
        return ValidationResult.unresolved(
            "proxy_low_certainty",
            confidence=max(confidence, contradiction),
            evidence_spans=evidence,
            validator_id=self.route,
        )


@dataclass(frozen=True, slots=True)
class ResolvedTextModels:
    scientific_encoder: ScientificTextEncoder
    entailment_validator: Validator
    model_root: Path
    scientific_route: str
    entailment_route: str
    scibert_revision: str
    bart_mnli_revision: str

    def audit_manifest(self) -> dict[str, str]:
        try:
            model_root = self.model_root.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            model_root = "external_local_asset_root"
        return {
            "model_root": model_root,
            "scientific_route": self.scientific_route,
            "entailment_route": self.entailment_route,
            "scibert_revision": self.scibert_revision,
            "bart_mnli_revision": self.bart_mnli_revision,
        }


def _local_manifest(
    model_dir: Path,
    configured: FrozenModelConfig,
) -> FrozenModelManifest | None:
    path = model_dir / "manifest.json"
    if not path.is_file():
        return None
    manifest = FrozenModelManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.model_id != configured.model_id or manifest.revision != configured.revision:
        raise ValueError(f"frozen model manifest does not match configuration: {model_dir.name}")
    return manifest


def resolve_text_models(
    *,
    config: ValidatorsConfig,
    model_root: Path,
    training_texts: Sequence[str],
) -> ResolvedTextModels:
    root = Path(model_root).resolve()
    scibert_dir = root / "scibert"
    bart_dir = root / "bart_mnli"
    scibert_manifest = _local_manifest(scibert_dir, config.scibert)
    bart_manifest = _local_manifest(bart_dir, config.bart_mnli)
    scientific_encoder: ScientificTextEncoder
    entailment_validator: Validator
    if scibert_manifest is None:
        scientific_encoder = TfidfScientificEncoder.fit(
            train_texts=training_texts,
            config=config.tfidf_proxy,
        )
    else:
        scientific_encoder = FrozenScientificEncoder(
            manifest=scibert_manifest,
            model_dir=scibert_dir,
            max_length=config.scibert_max_length,
            pooling=config.scibert_pooling,
        )
    if bart_manifest is None:
        entailment_validator = DeterministicNLIProxy(
            pass_threshold=config.learned_pass_threshold,
            fail_threshold=config.learned_fail_threshold,
            feature_weights=config.nli_proxy_feature_weights,
            contradiction_count_scale=config.nli_proxy_contradiction_count_scale,
        )
    else:
        entailment_validator = FrozenEntailmentValidator(
            manifest=bart_manifest,
            model_dir=bart_dir,
            pass_threshold=config.learned_pass_threshold,
            fail_threshold=config.learned_fail_threshold,
            max_length=config.bart_max_length,
        )
        entailment_validator.verify_assets()
    return ResolvedTextModels(
        scientific_encoder=scientific_encoder,
        entailment_validator=entailment_validator,
        model_root=root,
        scientific_route=scientific_encoder.route,
        entailment_route=entailment_validator.route,
        scibert_revision=config.scibert.revision,
        bart_mnli_revision=config.bart_mnli.revision,
    )
