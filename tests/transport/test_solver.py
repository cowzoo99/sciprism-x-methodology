import numpy as np


def test_masked_transport_obeys_marginals_and_hard_zeros() -> None:
    from sciprism_x.transport.solver import solve_transport

    node_cost = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    mask = np.array([[True, False], [False, True]])
    result = solve_transport(
        node_cost=node_cost,
        source_mass=np.array([0.4, 0.6], dtype=np.float64),
        target_mass=np.array([0.4, 0.6], dtype=np.float64),
        hard_mask=mask,
        epsilon=0.03,
        max_iterations=1000,
        marginal_tolerance=1.0e-9,
        objective_tolerance=1.0e-12,
        stable_iterations=5,
    )

    assert result.diagnostics.feasible is True
    assert result.diagnostics.converged is True
    np.testing.assert_allclose(result.plan.sum(axis=1), [0.4, 0.6], atol=1.0e-9)
    np.testing.assert_allclose(result.plan.sum(axis=0), [0.4, 0.6], atol=1.0e-9)
    assert np.all(result.plan[~mask] == 0.0)


def test_infeasible_mask_returns_diagnostics_without_objective() -> None:
    from sciprism_x.transport.solver import solve_transport

    result = solve_transport(
        node_cost=np.zeros((2, 2), dtype=np.float64),
        source_mass=np.array([0.5, 0.5], dtype=np.float64),
        target_mass=np.array([0.5, 0.5], dtype=np.float64),
        hard_mask=np.array([[True, False], [True, False]]),
        epsilon=0.03,
        max_iterations=100,
        marginal_tolerance=1.0e-9,
        objective_tolerance=1.0e-12,
        stable_iterations=5,
    )

    assert result.diagnostics.feasible is False
    assert result.diagnostics.converged is False
    assert result.objective is None
    assert np.all(result.plan == 0.0)


def test_dense_log_sinkhorn_reaches_asymmetric_marginals() -> None:
    from sciprism_x.transport.solver import solve_transport

    result = solve_transport(
        node_cost=np.array([[0.0, 0.4], [0.3, 0.1]], dtype=np.float64),
        source_mass=np.array([0.7, 0.3], dtype=np.float64),
        target_mass=np.array([0.2, 0.8], dtype=np.float64),
        hard_mask=np.ones((2, 2), dtype=np.bool_),
        epsilon=0.03,
        max_iterations=1000,
        marginal_tolerance=1.0e-9,
        objective_tolerance=1.0e-12,
        stable_iterations=5,
    )

    assert result.diagnostics.converged is True
    np.testing.assert_allclose(result.plan.sum(axis=1), [0.7, 0.3], atol=1.0e-9)
    np.testing.assert_allclose(result.plan.sum(axis=0), [0.2, 0.8], atol=1.0e-9)


def test_log_sinkhorn_handles_zero_mass_rows_and_columns_without_nan() -> None:
    from sciprism_x.transport.solver import solve_transport

    result = solve_transport(
        node_cost=np.array(
            [[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
        source_mass=np.array([0.5, 0.5, 0.0], dtype=np.float64),
        target_mass=np.array([0.5, 0.5, 0.0], dtype=np.float64),
        hard_mask=np.array(
            [[True, True, False], [True, True, False], [False, False, True]],
            dtype=np.bool_,
        ),
        epsilon=0.03,
        max_iterations=1000,
        marginal_tolerance=1.0e-9,
        objective_tolerance=1.0e-12,
        stable_iterations=5,
    )

    assert result.diagnostics.converged is True
    assert np.isfinite(result.plan).all()
    assert result.objective is not None
    assert np.isfinite(result.objective.total)
    np.testing.assert_allclose(result.plan.sum(axis=1), [0.5, 0.5, 0.0], atol=1.0e-9)
    np.testing.assert_allclose(result.plan.sum(axis=0), [0.5, 0.5, 0.0], atol=1.0e-9)


def test_objective_components_match_hand_calculation() -> None:
    from sciprism_x.transport.solver import transport_objective

    plan = np.array([[0.3, 0.2], [0.1, 0.4]], dtype=np.float64)
    node_cost = np.array([[0.0, 1.0], [2.0, 0.5]], dtype=np.float64)
    soft_cost = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    source_adjacency = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    target_adjacency = np.array([[[0.0, 0.5], [0.5, 0.0]]], dtype=np.float64)

    components = transport_objective(
        plan=plan,
        node_cost=node_cost,
        source_adjacency=source_adjacency,
        target_adjacency=target_adjacency,
        soft_constraint_cost=soft_cost,
        epsilon=0.2,
        lambda_structure=0.7,
        lambda_constraint=0.4,
    )

    expected_node = float(np.sum(plan * node_cost))
    expected_structure = sum(
        (source_adjacency[0, i, k] - target_adjacency[0, j, ell]) ** 2
        * plan[i, j]
        * plan[k, ell]
        for i in range(2)
        for k in range(2)
        for j in range(2)
        for ell in range(2)
    )
    expected_soft = float(np.sum(plan * soft_cost))
    expected_entropy = float(np.sum(plan * (np.log(plan) - 1.0)))
    expected_total = (
        expected_node
        + 0.7 * expected_structure
        + 0.4 * expected_soft
        + 0.2 * expected_entropy
    )

    assert abs(components.node - expected_node) <= 1.0e-12
    assert abs(components.structural - expected_structure) <= 1.0e-12
    assert abs(components.soft_constraint - expected_soft) <= 1.0e-12
    assert abs(components.negative_entropy - expected_entropy) <= 1.0e-12
    assert abs(components.total - expected_total) <= 1.0e-12


def test_structural_linear_cost_matches_literal_four_index_formula() -> None:
    from sciprism_x.transport.solver import _structural_linear_cost

    plan = np.array(
        [[0.12, 0.08], [0.20, 0.10], [0.18, 0.32]],
        dtype=np.float64,
    )
    source = np.array(
        [
            [[0.0, 1.0, 0.2], [0.4, 0.0, 0.8], [0.3, 0.6, 0.0]],
            [[0.0, 0.5, 0.7], [0.1, 0.0, 0.9], [0.2, 0.4, 0.0]],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [[0.0, 0.6], [0.2, 0.0]],
            [[0.0, 0.3], [0.9, 0.0]],
        ],
        dtype=np.float64,
    )
    literal = np.empty_like(plan)
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            literal[i, j] = np.mean(
                [
                    sum(
                        (source[relation, i, k] - target[relation, j, ell]) ** 2
                        * plan[k, ell]
                        for k in range(plan.shape[0])
                        for ell in range(plan.shape[1])
                    )
                    for relation in range(source.shape[0])
                ]
            )

    actual = _structural_linear_cost(plan, source, target)

    np.testing.assert_allclose(actual, literal, rtol=0.0, atol=1.0e-14)
