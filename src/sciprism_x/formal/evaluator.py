from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sciprism_x.config import RunConfig
from sciprism_x.data.schema import SciCUCase
from sciprism_x.motifs.extract import build_response_graph, build_task_graph
from sciprism_x.motifs.salience import SalienceWeights
from sciprism_x.motifs.schema import MotifFamily, TypedReasoningGraph
from sciprism_x.residuals.compute import compute_residuals
from sciprism_x.residuals.schema import (
    ResidualChannel,
    ResidualComponent,
    ResidualVector,
)
from sciprism_x.residuals.scores import build_score_bundle
from sciprism_x.responses.schema import ResponseRecord
from sciprism_x.state_field.features import build_state_vector
from sciprism_x.transport.costs import CostWeights, build_costs
from sciprism_x.transport.schema import (
    ObjectiveComponents,
    TransportDiagnostics,
    TransportResult,
)
from sciprism_x.transport.solver import solve_transport, transport_objective
from sciprism_x.validators.deterministic import default_deterministic_validators
from sciprism_x.validators.registry import ValidatorRegistry
from sciprism_x.validators.schema import ValidationPair, ValidationResult

from .proxies import ResolvedTextModels, resolve_text_models
from .ablations import AblationControls, typed_greedy_minimum_cost_matching


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOSS_COMPONENTS = ("node", "structure", "constraint")
ANONYMIZED_MODEL_ID = "anonymized_generator"


@dataclass(frozen=True, slots=True)
class MethodParameters:
    epsilon: float
    lambda_structure: float
    lambda_constraint: float
    omega_missing: float
    omega_unsupported: float
    minimum_support: float
    eta_node: float
    eta_structure: float
    eta_constraint: float


def fixture_method_parameters(config: RunConfig) -> MethodParameters:
    """Explicit first-grid fixture values; formal runs replace these after validation."""

    return MethodParameters(
        epsilon=config.transport.epsilon_grid[0],
        lambda_structure=config.transport.lambda_structure_grid[0],
        lambda_constraint=config.transport.lambda_constraint_grid[0],
        omega_missing=config.residuals.missing_penalty_grid[0],
        omega_unsupported=config.residuals.unsupported_penalty_grid[0],
        minimum_support=config.residuals.minimum_support_threshold,
        eta_node=config.residuals.pcs_coefficient_grid[0],
        eta_structure=config.residuals.pcs_coefficient_grid[0],
        eta_constraint=config.residuals.pcs_coefficient_grid[0],
    )


@dataclass(frozen=True, slots=True)
class FormalEvaluationContext:
    config: RunConfig
    text_models: ResolvedTextModels
    registry: ValidatorRegistry
    parameters: MethodParameters
    loss_normalizers: Mapping[str, float]
    residual_normalizers: Mapping[ResidualChannel, float]


@dataclass(frozen=True, slots=True)
class EvaluatedComponents:
    case: SciCUCase
    response: ResponseRecord
    task_graph: TypedReasoningGraph
    response_graph: TypedReasoningGraph
    validations: tuple[tuple[ValidationResult, ...], ...]
    transport: TransportResult
    residuals: ResidualVector | None


@dataclass(frozen=True, slots=True)
class FormalEvaluationRow:
    task_id: str
    model_id: str
    replicate: int
    endpoint_success: bool
    tsr: float
    pcs: float
    residual_consistency: float
    node_loss: float
    structure_loss: float
    constraint_loss: float
    residuals: np.ndarray
    hard_feasible: bool
    transport_converged: bool
    state_vector: np.ndarray
    diagnostic_reason: str


def build_formal_context(
    config: RunConfig,
    *,
    training_texts: Sequence[str],
    model_root: Path | None = None,
    parameters: MethodParameters | None = None,
    loss_normalizers: Mapping[str, float] | None = None,
    residual_normalizers: Mapping[ResidualChannel, float] | None = None,
) -> FormalEvaluationContext:
    configured_root = Path(config.validators.model_root)
    root = model_root or (
        configured_root if configured_root.is_absolute() else PROJECT_ROOT / configured_root
    )
    models = resolve_text_models(
        config=config.validators,
        model_root=root,
        training_texts=training_texts,
    )
    normalizers = residual_normalizers or {
        channel: 1.0 for channel in ResidualChannel
    }
    transport_normalizers = loss_normalizers or {
        component: 1.0 for component in LOSS_COMPONENTS
    }
    if set(transport_normalizers) != set(LOSS_COMPONENTS):
        raise ValueError("formal loss normalizers must cover node/structure/constraint")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in transport_normalizers.values()
    ):
        raise ValueError("formal loss normalizers must be finite and positive")
    if set(normalizers) != set(ResidualChannel):
        raise ValueError("formal residual normalizers must cover all six channels")
    if any(not math.isfinite(value) or value <= 0.0 for value in normalizers.values()):
        raise ValueError("formal residual normalizers must be finite and positive")
    deterministic_validators = default_deterministic_validators(
        symbolic_timeout_seconds=config.validators.symbolic_timeout_seconds
    )
    learned_validators = (models.entailment_validator,)
    from sciprism_x.release.protocol import load_validator_registry

    frozen_registry = load_validator_registry()
    expected_validator_order = tuple(
        item.validator_id for item in frozen_registry.validators
    )
    runtime_validator_order = tuple(
        str(getattr(validator, "validator_id", ""))
        for validator in (*deterministic_validators, *learned_validators)
    )
    if runtime_validator_order != expected_validator_order:
        raise ValueError("runtime validator order differs from the frozen registry")
    frozen_entailment = frozen_registry.validators[-1]
    if (
        frozen_entailment.model_id != config.validators.bart_mnli.model_id
        or frozen_entailment.revision != config.validators.bart_mnli.revision
        or frozen_entailment.pass_threshold != config.validators.learned_pass_threshold
        or frozen_entailment.fail_threshold != config.validators.learned_fail_threshold
    ):
        raise ValueError("runtime entailment configuration differs from the frozen registry")
    registry = ValidatorRegistry(
        deterministic=deterministic_validators,
        learned=learned_validators,
    )
    return FormalEvaluationContext(
        config=config,
        text_models=models,
        registry=registry,
        parameters=parameters or fixture_method_parameters(config),
        loss_normalizers=dict(transport_normalizers),
        residual_normalizers=dict(normalizers),
    )


def _graph_settings(context: FormalEvaluationContext) -> dict[str, object]:
    config = context.config
    return {
        "short_fragment_token_threshold": config.graph.short_fragment_token_threshold,
        "explicit_cue_confidence": config.graph.explicit_cue_confidence,
        "lexical_cue_confidence": config.graph.lexical_cue_confidence,
        "learned_recognizer_threshold": config.graph.learned_recognizer_threshold,
        "learned_recognizer_margin": config.graph.learned_recognizer_margin,
        "learned_recognizer": context.text_models.scientific_encoder,
        "salience_weights": SalienceWeights(**config.graph.salience_weights.model_dump()),
    }


def _validation_matrix(
    task_graph: TypedReasoningGraph,
    response_graph: TypedReasoningGraph,
    reference_context: str,
    registry: ValidatorRegistry,
) -> tuple[tuple[ValidationResult, ...], ...]:
    return tuple(
        tuple(
            registry.validate(
                ValidationPair(
                    task_node=task_node,
                    response_node=response_node,
                    reference_context=reference_context,
                )
            )
            for response_node in response_graph.nodes
        )
        for task_node in task_graph.nodes
    )


def _invalid_transport(task_nodes: int, response_nodes: int, reason: str) -> TransportResult:
    return TransportResult(
        plan=np.zeros((task_nodes, response_nodes), dtype=np.float64),
        objective=None,
        diagnostics=TransportDiagnostics(
            feasible=False,
            converged=False,
            iterations=0,
            marginal_error=math.inf,
            objective_change=None,
            consecutive_stable_iterations=0,
            reason=reason,
        ),
    )


def evaluate_scicu_components(
    *,
    case: SciCUCase,
    response: ResponseRecord,
    context: FormalEvaluationContext,
    controls: AblationControls | None = None,
) -> EvaluatedComponents:
    if response.task_id != case.task_id:
        raise ValueError("response task_id does not match the SciCUEval case")
    # Manuscript Sections 3.2 and 3.3: graph construction and validators must not
    # receive generator identity.  Preserve the original identifier only in the
    # outer result row used for paired aggregation.
    scoring_response = response.model_copy(
        update={"model_id": ANONYMIZED_MODEL_ID}
    )
    settings = _graph_settings(context)
    task_graph = build_task_graph(case, **settings)
    response_graph = build_response_graph(scoring_response, **settings)
    if not task_graph.nodes or not response_graph.nodes:
        return EvaluatedComponents(
            case=case,
            response=scoring_response,
            task_graph=task_graph,
            response_graph=response_graph,
            validations=(),
            transport=_invalid_transport(
                len(task_graph.nodes),
                len(response_graph.nodes),
                "empty_motif_graph",
            ),
            residuals=None,
        )
    validations = _validation_matrix(
        task_graph,
        response_graph,
        case.reference_context,
        context.registry,
    )
    config = context.config
    resolved = controls or AblationControls()
    costs = build_costs(
        task_graph,
        response_graph,
        validations,
        weights=CostWeights(**config.transport.node_cost_weights.model_dump()),
        scibert_embeddings=None,
        scientific_encoder=context.text_models.scientific_encoder,
        unresolved_soft_penalty=config.transport.unresolved_soft_penalty,
        unavailable_soft_penalty=config.transport.unavailable_soft_penalty,
    )
    hard_mask = (
        costs.hard_mask
        if resolved.hard_constraints
        else np.ones_like(costs.hard_mask, dtype=np.bool_)
    )
    source_adjacency = costs.source_adjacency if resolved.motif_structure else None
    target_adjacency = costs.target_adjacency if resolved.motif_structure else None
    soft_constraint = (
        costs.soft_constraint_cost
        if resolved.soft_penalties
        else np.zeros_like(costs.soft_constraint_cost)
    )
    parameters = context.parameters
    lambda_structure = parameters.lambda_structure if resolved.motif_structure else 0.0
    lambda_constraint = parameters.lambda_constraint if resolved.soft_penalties else 0.0
    if resolved.transport_mode == "optimal":
        transport = solve_transport(
            node_cost=costs.node_cost,
            source_mass=costs.source_mass,
            target_mass=costs.target_mass,
            hard_mask=hard_mask,
            epsilon=parameters.epsilon,
            max_iterations=config.transport.max_iterations,
            marginal_tolerance=config.transport.marginal_tolerance,
            objective_tolerance=config.transport.objective_tolerance,
            stable_iterations=config.transport.stable_iterations,
            source_adjacency=source_adjacency,
            target_adjacency=target_adjacency,
            soft_constraint_cost=soft_constraint,
            lambda_structure=lambda_structure,
            lambda_constraint=lambda_constraint,
        )
    else:
        try:
            plan = typed_greedy_minimum_cost_matching(
                node_cost=costs.node_cost,
                source_mass=costs.source_mass,
                target_mass=costs.target_mass,
                compatible=hard_mask,
                tolerance=config.transport.greedy_mass_tolerance,
                numerical_tolerance=config.transport.greedy_numerical_tolerance,
                lp_method=config.transport.greedy_lp_method,
                lp_presolve=config.transport.greedy_lp_presolve,
                lp_primal_feasibility_tolerance=(
                    config.transport.greedy_lp_primal_feasibility_tolerance
                ),
                lp_dual_feasibility_tolerance=(
                    config.transport.greedy_lp_dual_feasibility_tolerance
                ),
                lp_edge_weight_strategy=(
                    config.transport.greedy_lp_edge_weight_strategy
                ),
            )
        except ValueError:
            transport = _invalid_transport(
                len(task_graph.nodes),
                len(response_graph.nodes),
                "infeasible_typed_greedy_mask",
            )
        else:
            objective = transport_objective(
                plan=plan,
                node_cost=costs.node_cost,
                source_adjacency=source_adjacency,
                target_adjacency=target_adjacency,
                soft_constraint_cost=soft_constraint,
                epsilon=parameters.epsilon,
                lambda_structure=lambda_structure,
                lambda_constraint=lambda_constraint,
            )
            transport = TransportResult(
                plan=plan,
                objective=objective,
                diagnostics=TransportDiagnostics(
                    feasible=True,
                    converged=True,
                    iterations=1,
                    marginal_error=float(
                        max(
                            np.max(np.abs(plan.sum(axis=1) - costs.source_mass)),
                            np.max(np.abs(plan.sum(axis=0) - costs.target_mass)),
                        )
                    ),
                    objective_change=0.0,
                    consecutive_stable_iterations=1,
                    reason="typed_greedy_matching",
                ),
            )
    residuals = None
    if transport.objective is not None and transport.diagnostics.feasible:
        residuals = (
            compute_residuals(
                transport=transport,
                task_graph=task_graph,
                response_graph=response_graph,
                validations=validations,
                omega_missing=parameters.omega_missing,
                omega_unsupported=parameters.omega_unsupported,
                minimum_support=parameters.minimum_support,
                normalizers=context.residual_normalizers,
                normalizer_floor=config.residuals.normalizer_floor,
            )
            if resolved.residual_channels
            else _zero_residuals()
        )
    return EvaluatedComponents(
        case=case,
        response=scoring_response,
        task_graph=task_graph,
        response_graph=response_graph,
        validations=validations,
        transport=transport,
        residuals=residuals,
    )


def _invalid_residuals() -> ResidualVector:
    return ResidualVector(
        components={
            channel: ResidualComponent(
                matched=0.0,
                missing=1.0,
                unsupported=1.0,
                raw=1.0,
                normalizer=1.0,
                normalized=1.0,
            )
            for channel in ResidualChannel
        }
    )


def _zero_residuals() -> ResidualVector:
    return ResidualVector(
        components={
            channel: ResidualComponent(
                matched=0.0,
                missing=0.0,
                unsupported=0.0,
                raw=0.0,
                normalizer=1.0,
                normalized=0.0,
            )
            for channel in ResidualChannel
        }
    )


def _transport_entropy(plan: np.ndarray) -> float:
    positive = plan[plan > 0.0]
    return float(-np.sum(positive * np.log(positive))) if positive.size else 0.0


def _motif_discrepancy(
    task_graph: TypedReasoningGraph,
    response_graph: TypedReasoningGraph,
) -> float:
    families = tuple(MotifFamily)

    def distribution(graph: TypedReasoningGraph) -> np.ndarray:
        counts = np.asarray(
            [sum(node.motif_family is family for node in graph.nodes) for family in families],
            dtype=np.float64,
        )
        return counts / counts.sum() if counts.sum() else counts

    return float(0.5 * np.abs(distribution(task_graph) - distribution(response_graph)).sum())


def _evaluation_row_from_components(
    *,
    case: SciCUCase,
    response: ResponseRecord,
    context: FormalEvaluationContext,
    resolved: AblationControls,
    components: EvaluatedComponents,
) -> FormalEvaluationRow:
    feasible = (
        components.transport.objective is not None
        and components.transport.diagnostics.feasible
        and components.residuals is not None
    )
    residual_vector = components.residuals if feasible else _invalid_residuals()
    endpoint = response.response_text.strip() == case.reference_answer.strip()
    if feasible:
        assert components.transport.objective is not None
        raw_objective = components.transport.objective
        normalized_objective = ObjectiveComponents(
            node=raw_objective.node / context.loss_normalizers["node"],
            structural=(
                raw_objective.structural / context.loss_normalizers["structure"]
            ),
            soft_constraint=(
                raw_objective.soft_constraint
                / context.loss_normalizers["constraint"]
            ),
            negative_entropy=raw_objective.negative_entropy,
            total=raw_objective.total,
        )
        score = build_score_bundle(
            endpoint_success=(endpoint,),
            transport_objective=normalized_objective,
            residuals=residual_vector,
            eta_node=context.parameters.eta_node,
            eta_structure=context.parameters.eta_structure,
            eta_constraint=context.parameters.eta_constraint,
            beta=np.asarray(
                context.config.residuals.residual_consistency_beta,
                dtype=np.float64,
            ),
        )
        pcs = score.pcs
        consistency = score.residual_consistency
        node_loss = score.node_loss
        structure_loss = score.structure_loss
        constraint_loss = score.constraint_loss
    else:
        pcs = consistency = node_loss = structure_loss = constraint_loss = 0.0
    state = (
        build_state_vector(
            pcs=pcs,
            residual_consistency=consistency,
            node_loss=node_loss,
            structure_loss=structure_loss,
            constraint_loss=constraint_loss,
            transport_entropy=_transport_entropy(components.transport.plan),
            motif_signature_discrepancy=_motif_discrepancy(
                components.task_graph,
                components.response_graph,
            ),
            residuals=residual_vector,
            task_graph_complexity=float(
                len(components.task_graph.nodes) + len(components.task_graph.edges)
            ),
            response_graph_complexity=float(
                len(components.response_graph.nodes) + len(components.response_graph.edges)
            ),
        ).values
        if resolved.state_field
        else np.zeros(15, dtype=np.float64)
    )
    return FormalEvaluationRow(
        task_id=case.task_id,
        model_id=response.model_id,
        replicate=response.replicate,
        endpoint_success=endpoint,
        tsr=float(endpoint),
        pcs=pcs,
        residual_consistency=consistency,
        node_loss=node_loss,
        structure_loss=structure_loss,
        constraint_loss=constraint_loss,
        residuals=residual_vector.as_array(),
        hard_feasible=feasible,
        transport_converged=components.transport.diagnostics.converged,
        state_vector=state,
        diagnostic_reason=components.transport.diagnostics.reason,
    )


def evaluate_scicu_response_with_components(
    *,
    case: SciCUCase,
    response: ResponseRecord,
    context: FormalEvaluationContext,
    controls: AblationControls | None = None,
) -> tuple[FormalEvaluationRow, EvaluatedComponents]:
    """Evaluate once and retain the paper-auditable intermediate components."""

    resolved = controls or AblationControls()
    components = evaluate_scicu_components(
        case=case,
        response=response,
        context=context,
        controls=resolved,
    )
    row = _evaluation_row_from_components(
        case=case,
        response=response,
        context=context,
        resolved=resolved,
        components=components,
    )
    return row, components


def evaluate_scicu_response(
    *,
    case: SciCUCase,
    response: ResponseRecord,
    context: FormalEvaluationContext,
    controls: AblationControls | None = None,
) -> FormalEvaluationRow:
    row, _components = evaluate_scicu_response_with_components(
        case=case,
        response=response,
        context=context,
        controls=controls,
    )
    return row
