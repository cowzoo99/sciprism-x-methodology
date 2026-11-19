import numpy as np
import pytest


def test_state_vector_has_explicit_15_feature_order() -> None:
    from sciprism_x.residuals.schema import (
        ResidualChannel,
        ResidualComponent,
        ResidualVector,
    )
    from sciprism_x.state_field.features import STATE_FEATURE_NAMES, build_state_vector

    residuals = ResidualVector(
        components={
            channel: ResidualComponent(
                matched=0.0,
                missing=0.0,
                unsupported=0.0,
                raw=0.0,
                normalizer=1.0,
                normalized=(index + 1) / 10.0,
            )
            for index, channel in enumerate(ResidualChannel)
        }
    )
    state = build_state_vector(
        pcs=0.91,
        residual_consistency=0.82,
        node_loss=0.1,
        structure_loss=0.2,
        constraint_loss=0.3,
        transport_entropy=0.4,
        motif_signature_discrepancy=0.5,
        residuals=residuals,
        task_graph_complexity=0.6,
        response_graph_complexity=0.7,
    )

    assert len(STATE_FEATURE_NAMES) == 15
    assert state.values.shape == (15,)
    np.testing.assert_allclose(
        state.values,
        [0.91, 0.82, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7],
    )
    assert state.feature_names == STATE_FEATURE_NAMES


def test_state_vector_rejects_nonfinite_values() -> None:
    from sciprism_x.residuals.schema import (
        ResidualChannel,
        ResidualComponent,
        ResidualVector,
    )
    from sciprism_x.state_field.features import build_state_vector

    residuals = ResidualVector(
        components={
            channel: ResidualComponent(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            for channel in ResidualChannel
        }
    )
    with pytest.raises(ValueError, match="finite"):
        build_state_vector(
            pcs=float("nan"),
            residual_consistency=1.0,
            node_loss=0.0,
            structure_loss=0.0,
            constraint_loss=0.0,
            transport_entropy=0.0,
            motif_signature_discrepancy=0.0,
            residuals=residuals,
            task_graph_complexity=0.0,
            response_graph_complexity=0.0,
        )
