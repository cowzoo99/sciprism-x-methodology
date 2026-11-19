from __future__ import annotations

import re

from sciprism_x.motifs.schema import MotifFamily
from sciprism_x.release.protocol import (
    load_motif_dictionary,
    load_validator_registry,
    protocol_manifest,
)


def test_motif_dictionary_covers_every_runtime_family_and_compiles_cues() -> None:
    dictionary = load_motif_dictionary()

    assert {family.family_id for family in dictionary.families} == {
        value.value for value in MotifFamily
    }
    assert len(dictionary.families) == 6
    for family in dictionary.families:
        assert family.paper_source == "Appendix Table S1"
        assert family.cue_rules
        assert family.recognizer_prototypes
        for cue in family.cue_rules:
            flags = re.IGNORECASE if cue.ignore_case else 0
            re.compile(cue.pattern, flags)
    assert {rule.rule_id for rule in dictionary.edge_rules} == {
        "task_order",
        "response_order",
        "symbolic_shared_variable",
        "causal_target",
        "causal_justification",
        "evidence_source",
        "modality_source",
        "variable_reuse",
        "local_to_global",
    }


def test_validator_registry_matches_rule_first_runtime_contract() -> None:
    registry = load_validator_registry()

    assert registry.policy == "rule_first_learned_fallback"
    assert tuple(item.validator_id for item in registry.validators) == (
        "deterministic_symbolic_equivalence",
        "deterministic_dimensional_compatibility",
        "deterministic_causal_direction",
        "deterministic_evidence_binding",
        "deterministic_modality_compatibility",
        "deterministic_uncertainty_qualification",
        "deterministic_unsupported_claim",
        "deterministic_boundary_condition",
        "frozen_entailment_fallback",
    )
    learned = registry.validators[-1]
    assert learned.pass_threshold == 0.8
    assert learned.fail_threshold == 0.8
    assert "generator_identity" in registry.forbidden_inputs
    assert "final_test_outcomes" in registry.forbidden_inputs


def test_protocol_manifest_hashes_both_frozen_assets() -> None:
    manifest = protocol_manifest()

    assert manifest["schema_version"] == 1
    assert set(manifest["assets"]) == {
        "motif_dictionary.yaml",
        "validator_registry.yaml",
    }
    assert all(len(value["sha256"]) == 64 for value in manifest["assets"].values())
