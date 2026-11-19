from dataclasses import dataclass


def _pair_fixture():
    from sciprism_x.motifs.schema import MotifFamily, MotifNode
    from sciprism_x.validators.schema import ValidationPair

    task = MotifNode(
        node_id="T1",
        text="The evidence supports the claim.",
        motif_family=MotifFamily.EVIDENCE_BINDING,
        confidence=1.0,
        salience=1.0,
        source_span=(0, 32),
        variables=(),
        feature_evidence={},
    )
    response = task.model_copy(update={"node_id": "R1"})
    return ValidationPair(task_node=task, response_node=response, reference_context="")


@dataclass
class StubValidator:
    result: object
    call_count: int = 0

    def validate(self, pair):
        del pair
        self.call_count += 1
        return self.result


def test_learned_fallback_cannot_override_deterministic_failure() -> None:
    from sciprism_x.validators.registry import ValidatorRegistry
    from sciprism_x.validators.schema import ValidationResult, ValidationStatus

    deterministic = StubValidator(ValidationResult.fail("unit_mismatch", confidence=1.0))
    learned = StubValidator(ValidationResult.pass_("entails", confidence=0.99))
    registry = ValidatorRegistry(deterministic=[deterministic], learned=[learned])

    result = registry.validate(_pair_fixture())

    assert result.status is ValidationStatus.FAIL
    assert learned.call_count == 0


def test_learned_fallback_runs_only_after_all_deterministic_unresolved() -> None:
    from sciprism_x.validators.registry import ValidatorRegistry
    from sciprism_x.validators.schema import ValidationResult, ValidationStatus

    deterministic = StubValidator(
        ValidationResult.unresolved("implicit_claim", confidence=0.0)
    )
    learned = StubValidator(ValidationResult.pass_("entails", confidence=0.91))
    registry = ValidatorRegistry(deterministic=[deterministic], learned=[learned])

    result = registry.validate(_pair_fixture())

    assert result.status is ValidationStatus.PASS
    assert deterministic.call_count == 1
    assert learned.call_count == 1


def test_offline_missing_model_returns_unavailable_without_download(tmp_path) -> None:
    from sciprism_x.validators.learned import FrozenEntailmentValidator, FrozenModelManifest
    from sciprism_x.validators.schema import ValidationStatus

    model_dir = tmp_path / "missing-model"
    validator = FrozenEntailmentValidator(
        manifest=FrozenModelManifest(
            model_id="facebook/bart-large-mnli",
            revision="d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
            files={"config.json": "a" * 64},
        ),
        model_dir=model_dir,
        pass_threshold=0.8,
        fail_threshold=0.8,
    )

    result = validator.validate(_pair_fixture())

    assert result.status is ValidationStatus.UNAVAILABLE_EXTERNAL
    assert result.reason_code == "model_files_missing"
    assert not model_dir.exists()


def test_hash_mismatch_blocks_inference(tmp_path) -> None:
    from sciprism_x.validators.learned import FrozenEntailmentValidator, FrozenModelManifest
    from sciprism_x.validators.schema import ValidationStatus

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("not the captured file", encoding="utf-8")
    validator = FrozenEntailmentValidator(
        manifest=FrozenModelManifest(
            model_id="facebook/bart-large-mnli",
            revision="d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
            files={"config.json": "a" * 64},
        ),
        model_dir=model_dir,
        pass_threshold=0.8,
        fail_threshold=0.8,
    )

    result = validator.validate(_pair_fixture())

    assert result.status is ValidationStatus.UNAVAILABLE_EXTERNAL
    assert result.reason_code == "model_hash_mismatch"
