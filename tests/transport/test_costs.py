import numpy as np
import pytest


def _node(node_id, text, family, variables=(), evidence=None, salience=1.0):
    from sciprism_x.motifs.schema import MotifNode

    return MotifNode(
        node_id=node_id,
        text=text,
        motif_family=family,
        confidence=1.0,
        salience=salience,
        source_span=(0, len(text)),
        variables=tuple(variables),
        feature_evidence=evidence or {},
    )


def _graph(source_id, side, nodes, edges=()):
    from sciprism_x.motifs.schema import TypedReasoningGraph

    return TypedReasoningGraph(
        source_id=source_id,
        side=side,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def test_cost_components_are_normalized_and_mask_uses_validator_failures() -> None:
    from sciprism_x.motifs.schema import MotifFamily
    from sciprism_x.transport.costs import CostWeights, build_costs
    from sciprism_x.validators.schema import ValidationResult

    task = _graph(
        "task",
        "task",
        (
            _node(
                "T1",
                "According to Table 1, v = E/B = 12 m/s.",
                MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY,
                variables=("v", "E", "B"),
                evidence={"equations": ("v = E/B",), "units": ("12 m/s",)},
                salience=0.75,
            ),
            _node(
                "T2",
                "The force balances because the fields oppose.",
                MotifFamily.CAUSAL_MECHANISTIC_RELATION,
                salience=0.25,
            ),
        ),
    )
    response = _graph(
        "response",
        "response",
        (
            _node(
                "R1",
                "From Table 1, v = E/B = 12 m/s.",
                MotifFamily.UNIT_DIMENSIONAL_CONSISTENCY,
                variables=("v", "E", "B"),
                evidence={"equations": ("v = E/B",), "units": ("12 m/s",)},
                salience=0.6,
            ),
            _node(
                "R2",
                "The fields are unrelated.",
                MotifFamily.EVIDENCE_BINDING,
                salience=0.4,
            ),
        ),
    )
    validations = (
        (
            ValidationResult.pass_("match", confidence=1.0),
            ValidationResult.fail("invalid", confidence=1.0),
        ),
        (
            ValidationResult.unresolved("implicit", confidence=0.0),
            ValidationResult.unavailable("model_missing"),
        ),
    )
    embeddings = (
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
    )
    weights = CostWeights(
        motif=0.2,
        role=0.2,
        modality=0.15,
        evidence_anchor=0.15,
        variable_equation_anchor=0.2,
        scibert_cosine=0.1,
    )

    costs = build_costs(
        task,
        response,
        validations,
        weights=weights,
        scibert_embeddings=embeddings,
        unresolved_soft_penalty=0.5,
        unavailable_soft_penalty=1.0,
    )

    assert costs.node_cost.shape == (2, 2)
    assert costs.node_cost.dtype == np.float64
    assert all(
        np.all((component >= 0.0) & (component <= 1.0))
        for component in costs.node_components.values()
    )
    assert costs.hard_mask.tolist() == [[True, False], [True, True]]
    np.testing.assert_allclose(costs.source_mass, [0.75, 0.25])
    np.testing.assert_allclose(costs.target_mass, [0.6, 0.4])
    assert costs.soft_constraint_cost[1, 0] == 0.5
    assert costs.soft_constraint_cost[1, 1] == 1.0


def test_cost_weights_must_be_nonnegative_and_sum_to_one() -> None:
    from sciprism_x.transport.costs import CostWeights

    with pytest.raises(ValueError):
        CostWeights(
            motif=0.2,
            role=0.2,
            modality=0.2,
            evidence_anchor=0.2,
            variable_equation_anchor=0.2,
            scibert_cosine=0.2,
        )

    with pytest.raises(ValueError):
        CostWeights(
            motif=0.4,
            role=0.2,
            modality=0.2,
            evidence_anchor=0.2,
            variable_equation_anchor=0.1,
            scibert_cosine=-0.1,
        )
