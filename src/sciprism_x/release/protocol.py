from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files
import re
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from sciprism_x.motifs.schema import MotifFamily


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CueRule(StrictModel):
    cue_id: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    ignore_case: bool
    operational_source: str = Field(min_length=1)


class MotifFamilyProtocol(StrictModel):
    family_id: str = Field(min_length=1)
    paper_label: str = Field(min_length=1)
    paper_source: str = Field(min_length=1)
    example_motifs: tuple[str, ...] = Field(min_length=1)
    input_roles: tuple[str, ...] = Field(min_length=1)
    output_role: str = Field(min_length=1)
    dependency_types: tuple[str, ...] = Field(min_length=1)
    validator_interfaces: tuple[str, ...] = Field(min_length=1)
    recognizer_prototypes: tuple[str, ...] = Field(min_length=1)
    cue_rules: tuple[CueRule, ...] = Field(min_length=1)


class MotifEdgeRule(StrictModel):
    rule_id: str = Field(min_length=1)
    applies_to: str = Field(pattern=r"^(task|response|both)$")
    relation: str = Field(min_length=1)
    direction: str = Field(pattern=r"^(forward|bidirectional)$")
    condition: str = Field(min_length=1)


class MotifDictionary(StrictModel):
    schema_version: int = Field(ge=1)
    dictionary_id: str = Field(min_length=1)
    fixed_before_final_test: bool
    retention_criteria: tuple[str, ...] = Field(min_length=3)
    exclusions: tuple[str, ...] = Field(min_length=1)
    aliases: dict[str, str] = Field(min_length=1)
    families: tuple[MotifFamilyProtocol, ...] = Field(min_length=1)
    edge_rules: tuple[MotifEdgeRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runtime_coverage(self) -> "MotifDictionary":
        # Import lazily: motifs.extract consumes this protocol while the motifs
        # package is being initialized.
        from sciprism_x.motifs.schema import EdgeRelation, MotifFamily

        family_ids = [item.family_id for item in self.families]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("motif dictionary family IDs must be unique")
        if set(family_ids) != {family.value for family in MotifFamily}:
            raise ValueError("motif dictionary must cover the six runtime families")
        if any(value not in set(family_ids) for value in self.aliases.values()):
            raise ValueError("motif alias references an unknown family")
        edge_ids = [item.rule_id for item in self.edge_rules]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("motif edge-rule IDs must be unique")
        if any(item.relation not in {value.value for value in EdgeRelation} for item in self.edge_rules):
            raise ValueError("motif edge rule references an unknown relation")
        return self


class ValidatorProtocol(StrictModel):
    validator_id: str = Field(min_length=1)
    kind: str = Field(pattern=r"^(deterministic|learned)$")
    order: int = Field(ge=1)
    applies_to: tuple[str, ...] = Field(min_length=1)
    decision_rule: str = Field(min_length=1)
    output_statuses: tuple[str, ...] = Field(min_length=1)
    threshold_source: str = Field(min_length=1)
    pass_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fail_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    model_id: str | None = None
    revision: str | None = None


class ValidatorRegistryProtocol(StrictModel):
    schema_version: int = Field(ge=1)
    registry_id: str = Field(min_length=1)
    fixed_before_final_test: bool
    policy: str = Field(pattern=r"^rule_first_learned_fallback$")
    allowed_inputs: tuple[str, ...] = Field(min_length=1)
    forbidden_inputs: tuple[str, ...] = Field(min_length=1)
    low_certainty_policy: str = Field(min_length=1)
    validators: tuple[ValidatorProtocol, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self) -> "ValidatorRegistryProtocol":
        identifiers = [item.validator_id for item in self.validators]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("validator IDs must be unique")
        if [item.order for item in self.validators] != list(
            range(1, len(self.validators) + 1)
        ):
            raise ValueError("validator order must be contiguous and explicit")
        learned = [item for item in self.validators if item.kind == "learned"]
        if len(learned) != 1 or learned[0] is not self.validators[-1]:
            raise ValueError("the single learned validator must be the final fallback")
        return self


_ASSET_NAMES = ("motif_dictionary.yaml", "validator_registry.yaml")


def _asset_text(name: str) -> str:
    if name not in _ASSET_NAMES:
        raise ValueError(f"unknown release protocol asset: {name}")
    return files(__package__).joinpath("assets", name).read_text(encoding="utf-8")


def _asset_mapping(name: str) -> dict[str, object]:
    payload = yaml.safe_load(_asset_text(name))
    if not isinstance(payload, dict):
        raise ValueError(f"release protocol asset must be a mapping: {name}")
    return payload


@lru_cache(maxsize=1)
def load_motif_dictionary() -> MotifDictionary:
    return MotifDictionary.model_validate(_asset_mapping("motif_dictionary.yaml"))


@lru_cache(maxsize=1)
def load_validator_registry() -> ValidatorRegistryProtocol:
    return ValidatorRegistryProtocol.model_validate(
        _asset_mapping("validator_registry.yaml")
    )


@lru_cache(maxsize=1)
def compiled_family_patterns() -> dict[MotifFamily, tuple[re.Pattern[str], ...]]:
    from sciprism_x.motifs.schema import MotifFamily

    dictionary = load_motif_dictionary()
    return {
        MotifFamily(family.family_id): tuple(
            re.compile(rule.pattern, re.IGNORECASE if rule.ignore_case else 0)
            for rule in family.cue_rules
        )
        for family in dictionary.families
    }


@lru_cache(maxsize=1)
def family_aliases() -> dict[str, MotifFamily]:
    from sciprism_x.motifs.schema import MotifFamily

    return {
        alias: MotifFamily(family)
        for alias, family in load_motif_dictionary().aliases.items()
    }


@lru_cache(maxsize=1)
def family_recognizer_prototypes() -> dict[MotifFamily, tuple[str, ...]]:
    from sciprism_x.motifs.schema import MotifFamily

    return {
        MotifFamily(family.family_id): family.recognizer_prototypes
        for family in load_motif_dictionary().families
    }


@lru_cache(maxsize=1)
def edge_rules_by_id() -> dict[str, MotifEdgeRule]:
    return {rule.rule_id: rule for rule in load_motif_dictionary().edge_rules}


def protocol_asset_text(name: str) -> str:
    """Return the exact frozen asset text for final-package materialization."""

    return _asset_text(name)


def protocol_manifest() -> dict[str, object]:
    assets = {}
    for name in _ASSET_NAMES:
        text = _asset_text(name)
        assets[name] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
        }
    return {
        "schema_version": 1,
        "motif_dictionary_id": load_motif_dictionary().dictionary_id,
        "validator_registry_id": load_validator_registry().registry_id,
        "assets": assets,
    }
