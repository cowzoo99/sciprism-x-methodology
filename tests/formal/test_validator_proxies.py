import math

import numpy as np
import pytest

from sciprism_x.formal.proxies import TfidfScientificEncoder, resolve_text_models
from sciprism_x.motifs.schema import MotifFamily, MotifNode, TypedReasoningGraph
from sciprism_x.transport.costs import CostWeights, build_costs
from sciprism_x.validators.schema import ValidationPair, ValidationResult

from .helpers import run_config


TRAIN_TEXTS = (
    "The force balances because the electric and magnetic fields oppose.",
    "According to Table 1, the measured speed is 12 m/s.",
    "Under ideal conditions the result is approximately valid.",
)


def _node(node_id: str, text: str) -> MotifNode:
    return MotifNode(
        node_id=node_id,
        text=text,
        motif_family=MotifFamily.EVIDENCE_BINDING,
        confidence=1.0,
        salience=1.0,
        source_span=(0, len(text)),
        variables=(),
        feature_evidence={"equations": ("v = E/B",)},
    )


def _validation_pair() -> ValidationPair:
    return ValidationPair(
        task_node=_node("T1", "According to Table 1, force balance gives v = E/B = 12 m/s."),
        response_node=_node("R1", "Table 1 shows force balance, so v = E/B = 12 m/s."),
        reference_context="Table 1 contains the field measurements.",
    )


def test_missing_frozen_assets_select_numeric_proxy(tmp_path) -> None:
    resolved = resolve_text_models(
        config=run_config().validators,
        model_root=tmp_path,
        training_texts=TRAIN_TEXTS,
    )

    assert resolved.scientific_encoder.route == "tfidf_proxy"
    assert resolved.entailment_validator.route == "deterministic_nli_proxy"
    similarity = resolved.scientific_encoder.cosine(
        "force balance",
        "forces are balanced",
    )
    result = resolved.entailment_validator.validate(_validation_pair())
    assert math.isfinite(similarity)
    assert 0.0 <= similarity <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.status.value in {"PASS", "FAIL", "UNRESOLVED"}


def test_tfidf_fit_rejects_test_text() -> None:
    with pytest.raises(ValueError, match="training split"):
        TfidfScientificEncoder.fit(
            train_texts=TRAIN_TEXTS,
            config=run_config().validators.tfidf_proxy,
            test_texts=("forbidden",),
        )


def test_tfidf_proxy_supplies_nonempty_scibert_cost_channel(tmp_path) -> None:
    encoder = resolve_text_models(
        config=run_config().validators,
        model_root=tmp_path,
        training_texts=TRAIN_TEXTS,
    ).scientific_encoder
    task_node = _node("T1", "force balance from Table 1")
    response_node = _node("R1", "unrelated unsupported mechanism")
    task = TypedReasoningGraph(source_id="task", side="task", nodes=(task_node,), edges=())
    response = TypedReasoningGraph(
        source_id="response",
        side="response",
        nodes=(response_node,),
        edges=(),
    )
    costs = build_costs(
        task,
        response,
        ((ValidationResult.unresolved("proxy", confidence=0.5),),),
        weights=CostWeights(
            motif=0.2,
            role=0.2,
            modality=0.1,
            evidence_anchor=0.1,
            variable_equation_anchor=0.2,
            scibert_cosine=0.2,
        ),
        scibert_embeddings=None,
        scientific_encoder=encoder,
        unresolved_soft_penalty=0.5,
        unavailable_soft_penalty=1.0,
    )

    assert costs.node_components["scibert_cosine"].shape == (1, 1)
    assert np.isfinite(costs.node_components["scibert_cosine"]).all()
    assert costs.node_components["scibert_cosine"][0, 0] > 0.0
