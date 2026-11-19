from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Any, ClassVar

import numpy as np

from .schema import (
    BaselineKind,
    BaselineResult,
    BaselineStatus,
    LLMJudgeInput,
)


@dataclass(frozen=True, slots=True)
class LLMJudgeRequest:
    case_id: str
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    instructions: str
    input: dict[str, str]
    response_schema: dict[str, Any]


Requester = Callable[[LLMJudgeRequest, str], Mapping[str, Any]]


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_QUANTITY = re.compile(
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?"
    r"(?:\s*(?:m/s|km/h|V/m|m\^2|m|s|kg|K|Pa|N|J|W|T))?",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|none|without|unsupported|false|opposite|contradict\w*)\b",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(?:uncertain|uncertainty|may|might|could|likely|unlikely|approximately|"
    r"estimate\w*|confidence|insufficient|unknown)\b",
    re.IGNORECASE,
)
_OVERCONFIDENCE = re.compile(
    r"\b(?:definitely|certainly|undoubtedly|guaranteed|always|never)\b",
    re.IGNORECASE,
)

PAPER_JUDGE_FIELDS = (
    "answer_quality",
    "evidence_support",
    "reasoning_consistency",
    "uncertainty_handling",
    "final_score",
    "brief_error_type",
)

PAPER_JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating a scientific model response.\n"
    "Task:\n{task}\n"
    "Reference context or target behavior:\n{reference_context}\n"
    "Model response:\n{response}\n"
    "Evaluate the response according to:\n"
    "answer correctness,\n"
    "evidence support,\n"
    "reasoning consistency,\n"
    "uncertainty handling,\n"
    "unsupported or overconfident claims.\n"
    "Return only a structured judgment with: answer_quality, evidence_support, "
    "reasoning_consistency, uncertainty_handling, final_score, and "
    "brief_error_type. Do not provide chain-of-thought."
)


def validate_paper_judgment(response: Mapping[str, Any]) -> None:
    if set(response) != set(PAPER_JUDGE_FIELDS):
        raise ValueError(
            "LLM judge response must contain the complete Supplementary Table S12 "
            "structured judgment"
        )
    for name in PAPER_JUDGE_FIELDS[:4]:
        value = response[name]
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        ordinal = isinstance(value, str) and bool(value.strip())
        if not ordinal and not (
            numeric and np.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"LLM judge diagnostic field must be a nonempty ordinal or [0, 1] "
                f"number: {name}"
            )
    final_score = response["final_score"]
    if (
        not isinstance(final_score, (int, float))
        or isinstance(final_score, bool)
        or not np.isfinite(float(final_score))
        or not 0.0 <= float(final_score) <= 1.0
    ):
        raise ValueError("LLM judge final_score must be a finite number in [0, 1]")
    error_type = response["brief_error_type"]
    if not isinstance(error_type, str) or not error_type.strip():
        raise ValueError("LLM judge brief_error_type must be a nonempty string")


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(text)}


def _coverage(required: set[str], observed: set[str]) -> float:
    return 1.0 if not required else len(required & observed) / len(required)


def _f1_overlap(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _quantities(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", match.group(0)).casefold()
        for match in _QUANTITY.finditer(text)
    }


@dataclass(frozen=True, slots=True)
class LocalInferredJudgeAdapter:
    """Deterministic local implementation of the published judge rubric."""

    pass_threshold: float
    fail_threshold: float
    component_weights: tuple[float, float, float, float, float]

    kind: ClassVar[BaselineKind] = BaselineKind.LLM_JUDGE
    input_type: ClassVar[type[LLMJudgeInput]] = LLMJudgeInput

    def __post_init__(self) -> None:
        for name, value in (
            ("pass_threshold", self.pass_threshold),
            ("fail_threshold", self.fail_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if any(weight < 0.0 for weight in self.component_weights) or abs(
            sum(self.component_weights) - 1.0
        ) > 1.0e-12:
            raise ValueError("component_weights must be nonnegative and sum to one")

    def evaluate(self, value: LLMJudgeInput) -> BaselineResult:
        if not isinstance(value, LLMJudgeInput):
            raise TypeError("LocalInferredJudgeAdapter requires LLMJudgeInput")

        response_tokens = _tokens(value.response)
        reference_tokens = _tokens(value.reference_context)
        task_tokens = _tokens(value.task)
        answer_overlap = _f1_overlap(response_tokens, reference_tokens)
        evidence_coverage = _coverage(reference_tokens, response_tokens)

        response_negated = bool(_NEGATION.search(value.response))
        reference_negated = bool(_NEGATION.search(value.reference_context))
        consistency = float(response_negated == reference_negated)

        units = _coverage(_quantities(value.reference_context), _quantities(value.response))
        uncertainty_required = bool(
            _UNCERTAINTY.search(f"{value.task} {value.reference_context}")
        )
        uncertainty_present = bool(_UNCERTAINTY.search(value.response))
        overconfident = bool(_OVERCONFIDENCE.search(value.response))
        if uncertainty_required:
            uncertainty = float(uncertainty_present and not overconfident)
        else:
            unsupported_tokens = response_tokens - reference_tokens - task_tokens
            uncertainty = float(not overconfident or not unsupported_tokens)

        components = np.asarray(
            (answer_overlap, evidence_coverage, consistency, units, uncertainty),
            dtype=np.float64,
        )
        score = float(
            np.clip(
                np.dot(components, np.asarray(self.component_weights, dtype=np.float64)),
                0.0,
                1.0,
            )
        )
        error_names = (
            "answer_quality",
            "evidence_support",
            "reasoning_consistency",
            "unit_consistency",
            "uncertainty_handling",
        )
        brief_error_type = error_names[int(np.argmin(components))]
        return BaselineResult(
            self.kind,
            BaselineStatus.AVAILABLE,
            "llm_judge_final_score",
            score,
            (
                ("answer_quality", f"{answer_overlap:.17g}"),
                ("evidence_support", f"{evidence_coverage:.17g}"),
                ("reasoning_consistency", f"{consistency:.17g}"),
                ("unit_consistency", f"{units:.17g}"),
                ("uncertainty_handling", f"{uncertainty:.17g}"),
                ("brief_error_type", brief_error_type),
                ("pass_threshold", f"{self.pass_threshold:.17g}"),
                ("fail_threshold", f"{self.fail_threshold:.17g}"),
                ("route", "local_inferred"),
            ),
        )


@dataclass(frozen=True, slots=True)
class LLMJudgeAdapter:
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    api_key: str | None
    requester: Requester | None

    kind: ClassVar[BaselineKind] = BaselineKind.LLM_JUDGE
    input_type: ClassVar[type[LLMJudgeInput]] = LLMJudgeInput

    def __post_init__(self) -> None:
        if self.model != "gpt-4o-2024-08-06":
            raise ValueError("method contract requires gpt-4o-2024-08-06")
        if self.temperature != 0.0 or self.top_p != 1.0 or self.max_output_tokens != 512:
            raise ValueError("LLM judge sampling controls differ from Supplementary Table S12")

    def build_request(self, value: LLMJudgeInput) -> LLMJudgeRequest:
        if not isinstance(value, LLMJudgeInput):
            raise TypeError("LLMJudgeAdapter requires LLMJudgeInput")
        return LLMJudgeRequest(
            case_id=value.case_id,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            instructions=PAPER_JUDGE_PROMPT_TEMPLATE.format(
                task=value.task,
                reference_context=value.reference_context,
                response=value.response,
            ),
            input={
                "task": value.task,
                "response": value.response,
                "reference_context": value.reference_context,
            },
            response_schema={
                "type": "object",
                "properties": {
                    name: {
                        "anyOf": [
                            {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            {"type": "string", "minLength": 1},
                        ]
                    }
                    for name in PAPER_JUDGE_FIELDS
                    if name not in {"final_score", "brief_error_type"}
                }
                | {
                    "final_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Scalar judgment of answer correctness, evidence support, "
                            "reasoning consistency, uncertainty handling, and unsupported "
                            "or overconfident claims."
                        ),
                    },
                    "brief_error_type": {"type": "string", "minLength": 1},
                },
                "required": list(PAPER_JUDGE_FIELDS),
                "additionalProperties": False,
            },
        )

    def evaluate(self, value: LLMJudgeInput) -> BaselineResult:
        request = self.build_request(value)
        if not self.api_key or self.requester is None:
            return BaselineResult(
                self.kind,
                BaselineStatus.UNAVAILABLE_EXTERNAL,
                "llm_judge_final_score",
                None,
                (("reason", "credentials_or_requester_unavailable"),),
            )
        response = self.requester(request, self.api_key)
        validate_paper_judgment(response)
        score = float(response["final_score"])
        return BaselineResult(
            self.kind,
            BaselineStatus.AVAILABLE,
            "llm_judge_final_score",
            score,
            tuple(
                (name, str(response[name]))
                for name in PAPER_JUDGE_FIELDS
                if name != "final_score"
            )
            + (("route", "exact_gpt4o"),),
        )
