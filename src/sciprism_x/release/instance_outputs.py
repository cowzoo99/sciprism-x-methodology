from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from sciprism_x.formal.evaluator import EvaluatedComponents
from sciprism_x.residuals.schema import ResidualChannel, ResidualComponent


@dataclass(frozen=True, slots=True)
class SerializedInstance:
    instance_key: str
    plan_key: str
    plan: np.ndarray
    task_graph_record: dict[str, object]
    response_graph_record: dict[str, object]
    validator_rows: tuple[dict[str, object], ...]
    transport_row: dict[str, object]
    residual_rows: tuple[dict[str, object], ...]
    link_row: dict[str, object]


def make_instance_key(task_id: str, model_id: str, replicate: int) -> str:
    if not task_id or not model_id or replicate < 0:
        raise ValueError("instance key requires task, model, and nonnegative replicate")
    return f"{task_id}::{model_id}::{replicate}"


def _invalid_component() -> ResidualComponent:
    # Matches formal.evaluator._invalid_residuals: infeasible transport is an
    # explicit unit residual, never an empty or silently omitted vector.
    return ResidualComponent(
        matched=0.0,
        missing=1.0,
        unsupported=1.0,
        raw=1.0,
        normalizer=1.0,
        normalized=1.0,
    )


def serialize_evaluated_components(
    components: EvaluatedComponents,
    *,
    model_id: str,
    replicate: int,
) -> SerializedInstance:
    task_graph = components.task_graph
    response_graph = components.response_graph
    task_count = len(task_graph.nodes)
    response_count = len(response_graph.nodes)
    if components.transport.plan.shape != (task_count, response_count):
        raise ValueError("transport plan shape does not match serialized graphs")
    if components.validations:
        if len(components.validations) != task_count or any(
            len(row) != response_count for row in components.validations
        ):
            raise ValueError("validator matrix shape does not match serialized graphs")
    elif task_count and response_count:
        raise ValueError("non-empty graph pair requires a validator matrix")
    if components.residuals is not None and set(components.residuals.components) != set(
        ResidualChannel
    ):
        raise ValueError("residual output must cover all six method channels")

    instance_key = make_instance_key(components.case.task_id, model_id, replicate)
    digest = hashlib.sha256(instance_key.encode("utf-8")).hexdigest()
    plan_key = f"plan_{digest[:24]}"
    task_graph_id = f"task::{components.case.task_id}"
    response_graph_id = f"response::{instance_key}"
    validator_rows = tuple(
        {
            "instance_key": instance_key,
            "task_node_id": task_node.node_id,
            "response_node_id": response_node.node_id,
            "status": result.status.value,
            "confidence": result.confidence,
            "reason_code": result.reason_code,
            "validator_id": result.validator_id,
            "evidence_spans_json": json.dumps(
                [span.model_dump(mode="json") for span in result.evidence_spans],
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
        }
        for task_node, row in zip(task_graph.nodes, components.validations, strict=True)
        for response_node, result in zip(response_graph.nodes, row, strict=True)
    )

    objective = components.transport.objective
    diagnostics = components.transport.diagnostics
    transport_row = {
        "instance_key": instance_key,
        "plan_key": plan_key,
        "task_node_count": task_count,
        "response_node_count": response_count,
        "objective_available": objective is not None,
        "objective_node": objective.node if objective is not None else 0.0,
        "objective_structural": objective.structural if objective is not None else 0.0,
        "objective_soft_constraint": (
            objective.soft_constraint if objective is not None else 0.0
        ),
        "objective_negative_entropy": (
            objective.negative_entropy if objective is not None else 0.0
        ),
        "objective_total": objective.total if objective is not None else 0.0,
        **asdict(diagnostics),
    }

    residual_source = (
        "computed" if components.residuals is not None else "infeasible_transport_unit_residual"
    )
    residual_rows = tuple(
        {
            "instance_key": instance_key,
            "channel": channel.value,
            "value_source": residual_source,
            **asdict(
                components.residuals.components[channel]
                if components.residuals is not None
                else _invalid_component()
            ),
        }
        for channel in ResidualChannel
    )
    link_row = {
        "instance_key": instance_key,
        "task_id": components.case.task_id,
        "model_id": model_id,
        "replicate": replicate,
        "task_graph_id": task_graph_id,
        "response_graph_id": response_graph_id,
        "plan_key": plan_key,
        "validator_pair_count": len(validator_rows),
        "residual_channel_count": len(residual_rows),
    }
    return SerializedInstance(
        instance_key=instance_key,
        plan_key=plan_key,
        plan=np.asarray(components.transport.plan, dtype=np.float64),
        task_graph_record={
            "graph_id": task_graph_id,
            "task_id": components.case.task_id,
            "graph": task_graph.model_dump(mode="json"),
        },
        response_graph_record={
            "graph_id": response_graph_id,
            "instance_key": instance_key,
            "graph": response_graph.model_dump(mode="json"),
        },
        validator_rows=validator_rows,
        transport_row=transport_row,
        residual_rows=residual_rows,
        link_row=link_row,
    )
