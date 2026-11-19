from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .provenance import ParameterRecord, ProvenanceClass


class TrackIsolationError(ValueError):
    """Raised when Track A and Track B inputs are mixed."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeneratorConfig(StrictModel):
    model_ids: tuple[str, ...]
    temperature: float
    max_output_tokens: int
    responses_per_task: int
    evaluation_period: tuple[str, str]


class JudgeConfig(StrictModel):
    model_id: str
    dataset: str
    temperature: float
    top_p: float
    max_output_tokens: int


class SciCUEvalConfig(StrictModel):
    role: Literal["primary"]


class DBP15KConfig(StrictModel):
    role: Literal["auxiliary"]
    language_pairs: tuple[str, ...]
    seed_fraction: float
    full_candidate_pool: bool
    candidate_pruning: bool


class DatasetsConfig(StrictModel):
    scicueval: SciCUEvalConfig
    dbp15k: DBP15KConfig


class RegimesConfig(StrictModel):
    names: tuple[str, ...]
    nonstress_cases: int
    boundary_stress_cases: int


class PaperConfig(StrictModel):
    seeds: tuple[int, ...]
    hyperparameter_trials: int
    patience: int
    max_epochs: int
    bootstrap_resamples: int
    confidence_level: float
    two_sided_tests: bool
    correction: Literal["holm_bonferroni"]


class NodeCostWeightsConfig(StrictModel):
    motif: float = Field(ge=0.0)
    role: float = Field(ge=0.0)
    modality: float = Field(ge=0.0)
    evidence_anchor: float = Field(ge=0.0)
    variable_equation_anchor: float = Field(ge=0.0)
    scibert_cosine: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_unit_sum(self) -> "NodeCostWeightsConfig":
        if abs(sum(self.model_dump().values()) - 1.0) > 1.0e-12:
            raise ValueError("node-cost weights must sum to one")
        return self


class TransportConfig(StrictModel):
    dtype: Literal["float64"]
    loss_normalizer_percentile: float = Field(gt=0.0, le=1.0)
    loss_normalizer_floor: float = Field(gt=0.0)
    max_iterations: int
    marginal_tolerance: float
    objective_tolerance: float
    stable_iterations: int
    epsilon_grid: tuple[float, ...]
    lambda_structure_grid: tuple[float, ...]
    lambda_constraint_grid: tuple[float, ...]
    node_cost_weights: NodeCostWeightsConfig
    unresolved_soft_penalty: float = Field(ge=0.0, le=1.0)
    unavailable_soft_penalty: float = Field(ge=0.0, le=1.0)
    greedy_mass_tolerance: float = Field(gt=0.0)
    greedy_numerical_tolerance: float = Field(gt=0.0)
    greedy_lp_method: Literal["highs-ds"]
    greedy_lp_presolve: Literal[True]
    greedy_lp_primal_feasibility_tolerance: float = Field(gt=0.0)
    greedy_lp_dual_feasibility_tolerance: float = Field(gt=0.0)
    greedy_lp_edge_weight_strategy: Literal["dantzig"]


class ResidualConfig(StrictModel):
    gamma_grid: tuple[float, ...]
    missing_penalty_grid: tuple[float, ...]
    unsupported_penalty_grid: tuple[float, ...]
    normalizer_percentile: float
    normalizer_floor: float
    pcs_coefficient_grid: tuple[float, ...]
    minimum_support_threshold: float = Field(ge=0.0, le=1.0)
    residual_consistency_beta: tuple[float, float, float, float, float, float]


class SalienceWeightsConfig(StrictModel):
    inverse_position: float = Field(ge=0.0)
    motif_confidence: float = Field(ge=0.0)
    variable_reuse: float = Field(ge=0.0)
    graph_centrality: float = Field(ge=0.0)

    @model_validator(mode="after")
    def require_unit_sum(self) -> "SalienceWeightsConfig":
        if abs(sum(self.model_dump().values()) - 1.0) > 1.0e-12:
            raise ValueError("salience weights must sum to one")
        return self


class GraphConfig(StrictModel):
    short_fragment_token_threshold: int = Field(ge=1)
    explicit_cue_confidence: float = Field(ge=0.0, le=1.0)
    lexical_cue_confidence: float = Field(ge=0.0, le=1.0)
    learned_recognizer_threshold: float = Field(ge=0.0, le=1.0)
    learned_recognizer_margin: float = Field(ge=0.0, le=1.0)
    salience_weights: SalienceWeightsConfig


class ReconstructionProfileConfig(StrictModel):
    physical_error: float = Field(ge=0.0, le=1.0)
    logical_error: float = Field(ge=0.0, le=1.0)
    unit_error: float = Field(ge=0.0, le=1.0)
    causal_error: float = Field(ge=0.0, le=1.0)
    modal_evidence_error: float = Field(ge=0.0, le=1.0)
    uncertainty_error: float = Field(ge=0.0, le=1.0)
    endpoint_error: float = Field(ge=0.0, le=1.0)
    motif_omission: float = Field(ge=0.0, le=1.0)
    unsupported_content: float = Field(ge=0.0, le=1.0)
    length_ratio: float = Field(ge=0.25, le=2.0)
    evidence_synthesis: float = Field(ge=0.0, le=1.0)


class ResponseReconstructionConfig(StrictModel):
    fixture_profile: ReconstructionProfileConfig


class BaselineReconstructionConfig(StrictModel):
    endpoint_case_sensitive: Literal[False]
    endpoint_strip_whitespace: Literal[True]
    endpoint_collapse_whitespace: Literal[True]
    attribution_lowercase: Literal[True]
    attribution_token_pattern: str = Field(min_length=1)
    attribution_empty_overlap_score: float = Field(ge=0.0, le=1.0)
    taxonomy_patterns: tuple[tuple[str, tuple[str, ...]], ...]
    taxonomy_no_failure_score: float = Field(ge=0.0, le=1.0)
    taxonomy_failure_score: float = Field(ge=0.0, le=1.0)
    taxonomy_lowercase: Literal[True]
    uncertainty_maximum_entropy: float = Field(gt=0.0)
    support_clip_to_unit_interval: Literal[True]
    transport_score_transform: Literal["exp_negative_cost"]
    relation_multiset: Literal[True]
    ranking_descending: Literal[True]
    representation_dimensions: int = Field(gt=0)
    task_success_threshold: float = Field(ge=0.0, le=1.0)
    matched_context_rotations: int = Field(gt=0)
    local_judge_component_weights: tuple[float, float, float, float, float]
    dbp_validation_transport_epsilon: float = Field(gt=0.0)
    support_neighbors_grid: tuple[int, ...]
    support_distance_scale_grid: tuple[float, ...]
    uncertainty_mode_codes: tuple[int, ...]
    graph_edit_cost_grid: tuple[float, ...]
    graph_edit_timeout_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def require_valid_baseline_grid(self) -> "BaselineReconstructionConfig":
        if not self.taxonomy_patterns or any(
            not name or not patterns or any(not pattern for pattern in patterns)
            for name, patterns in self.taxonomy_patterns
        ):
            raise ValueError("baseline taxonomy requires named nonempty patterns")
        if len({name for name, _patterns in self.taxonomy_patterns}) != len(
            self.taxonomy_patterns
        ):
            raise ValueError("baseline taxonomy category names must be unique")
        if any(weight < 0.0 for weight in self.local_judge_component_weights) or abs(
            sum(self.local_judge_component_weights) - 1.0
        ) > 1.0e-12:
            raise ValueError("local judge weights must be nonnegative and sum to one")
        if not self.support_neighbors_grid or any(
            value <= 0 for value in self.support_neighbors_grid
        ):
            raise ValueError("baseline neighbor counts must be positive")
        if not self.support_distance_scale_grid or any(
            value <= 0.0 for value in self.support_distance_scale_grid
        ):
            raise ValueError("baseline distance scales must be positive")
        if set(self.uncertainty_mode_codes) != {0, 1, 2}:
            raise ValueError("uncertainty modes must cover confidence/entropy/product")
        if not self.graph_edit_cost_grid or any(
            value < 0.0 for value in self.graph_edit_cost_grid
        ):
            raise ValueError("graph-edit costs must be nonnegative")
        return self


class DBPReconstructionConfig(StrictModel):
    signature_neighborhood_hops: int = Field(gt=0)
    local_transport_hops: int = Field(gt=0)
    signature_cosine_weight: float = Field(ge=0.0, le=1.0)
    relation_overlap_weight: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_normalized_candidate_weights(self) -> "DBPReconstructionConfig":
        if abs(
            self.signature_cosine_weight + self.relation_overlap_weight - 1.0
        ) > 1.0e-12:
            raise ValueError("DBP candidate weights must sum to one")
        return self


class FrozenModelConfig(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class TfidfProxyConfig(StrictModel):
    lowercase: Literal[True]
    analyzer: Literal["word"]
    ngram_range: tuple[int, int]
    min_df: int = Field(gt=0)
    max_df: float = Field(gt=0.0, le=1.0)
    max_features: None
    norm: Literal["l2"]
    use_idf: Literal[True]
    smooth_idf: Literal[True]
    sublinear_tf: Literal[False]
    binary: Literal[False]
    strip_accents: None
    stop_words: None
    token_pattern: str = Field(min_length=1)
    dtype: Literal["float64"]

    @model_validator(mode="after")
    def require_valid_ngram_range(self) -> "TfidfProxyConfig":
        minimum, maximum = self.ngram_range
        if minimum < 1 or maximum < minimum:
            raise ValueError("TF-IDF ngram_range must be positive and ordered")
        return self


class ValidatorsConfig(StrictModel):
    symbolic_timeout_seconds: float = Field(gt=0.0)
    model_root: str = Field(min_length=1)
    scibert_pooling: Literal["attention_masked_mean"]
    scibert_max_length: int = Field(ge=1)
    bart_max_length: int = Field(ge=1)
    learned_pass_threshold: float = Field(ge=0.0, le=1.0)
    learned_fail_threshold: float = Field(ge=0.0, le=1.0)
    nli_proxy_feature_weights: tuple[float, float, float, float, float]
    nli_proxy_contradiction_count_scale: float = Field(gt=0.0)
    tfidf_proxy: TfidfProxyConfig
    scibert: FrozenModelConfig
    bart_mnli: FrozenModelConfig

    @model_validator(mode="after")
    def require_normalized_nli_proxy_weights(self) -> "ValidatorsConfig":
        if any(weight < 0.0 for weight in self.nli_proxy_feature_weights) or abs(
            sum(self.nli_proxy_feature_weights) - 1.0
        ) > 1.0e-12:
            raise ValueError("NLI proxy weights must be nonnegative and sum to one")
        return self


class StateFieldConfig(StrictModel):
    dimensions: tuple[int, int, int]
    energy_dimensions: tuple[int, int, int]
    activation: Literal["gelu"]
    gelu_approximate: Literal["none"]
    linear_bias: Literal[True]
    initialization: Literal["kaiming_uniform_linear_default"]
    dropout: float
    dropout_inplace: Literal[False]
    optimizer: Literal["adamw"]
    learning_rate_grid: tuple[float, ...]
    weight_decay_grid: tuple[float, ...]
    batch_size: int
    energy_loss_weight_grid: tuple[float, ...]
    energy_margin_grid: tuple[float, ...]
    early_stopping_min_delta: float = Field(ge=0.0)
    adamw_betas: tuple[float, float]
    adamw_eps: float = Field(gt=0.0)
    class_weighting: Literal["inverse_frequency_mean_one"]
    deterministic_shuffling: Literal[True]


class BoundaryConfig(StrictModel):
    mad_floor: float = Field(gt=0.0)
    mad_consistency_factor: Literal[1.0]
    standardized_radii: tuple[float, ...]
    symbolic_depth_steps: tuple[int, ...]
    context_composition_steps: tuple[int, ...]
    dbp_removal_ratios: tuple[float, ...]
    dbp_seed_fractions: tuple[float, ...]
    evidence_removal_rule: Literal["ceil_fraction_preserve_one"]
    dbp_edge_removal_rule: Literal["ceil_fraction_preserve_one"]
    seed_alignment_rounding: Literal["nearest_integer"]

    @model_validator(mode="after")
    def require_aligned_perturbation_levels(self) -> "BoundaryConfig":
        count = len(self.standardized_radii)
        if count == 0:
            raise ValueError("standardized_radii must not be empty")
        if len(self.symbolic_depth_steps) != count or len(self.context_composition_steps) != count:
            raise ValueError("SciCUEval perturbation levels must have equal lengths")
        if any(radius <= 0.0 for radius in self.standardized_radii):
            raise ValueError("standardized radii must be positive")
        if any(left >= right for left, right in zip(
            self.standardized_radii, self.standardized_radii[1:], strict=False
        )):
            raise ValueError("standardized radii must be strictly increasing")
        if any(step <= 0 for step in self.symbolic_depth_steps + self.context_composition_steps):
            raise ValueError("perturbation step counts must be positive")
        if any(not 0.0 < ratio < 1.0 for ratio in self.dbp_removal_ratios):
            raise ValueError("DBP removal ratios must be in (0, 1)")
        if any(not 0.0 < fraction < 0.30 for fraction in self.dbp_seed_fractions):
            raise ValueError("DBP seed fractions must be in (0, 0.30)")
        return self


class OpenSetConfig(StrictModel):
    cluster_counts: tuple[int, ...]
    cluster_selection: Literal["silhouette"]
    kmeans_init: Literal["k-means++"]
    kmeans_n_init: int = Field(gt=0)
    kmeans_max_iterations: int = Field(gt=0)
    kmeans_tolerance: float = Field(gt=0.0)
    kmeans_algorithm: Literal["lloyd"]
    unknownness_weights: tuple[float, float, float]
    novelty_normalization_quantile: float = Field(gt=0.0, le=1.0)
    updq_weights: tuple[float, float, float]
    separation_floor: float = Field(gt=0.0)
    calibration_bins: int = Field(gt=0)
    calibration_minimum_bin_samples: int = Field(ge=0)
    calibration_minimum_bin_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_valid_search_and_weights(self) -> "OpenSetConfig":
        if not self.cluster_counts or any(count < 2 for count in self.cluster_counts):
            raise ValueError("cluster counts must be at least two")
        if any(left >= right for left, right in zip(
            self.cluster_counts, self.cluster_counts[1:], strict=False
        )):
            raise ValueError("cluster counts must be strictly increasing")
        for name, weights in (
            ("unknownness", self.unknownness_weights),
            ("UPDQ", self.updq_weights),
        ):
            if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1.0e-12:
                raise ValueError(f"{name} weights must be nonnegative and sum to one")
        return self


class RunConfig(StrictModel):
    track: Literal["A", "B"]
    published_targets: str | None
    generators: GeneratorConfig
    judge: JudgeConfig
    datasets: DatasetsConfig
    regimes: RegimesConfig
    paper: PaperConfig
    transport: TransportConfig
    residuals: ResidualConfig
    graph: GraphConfig
    response_reconstruction: ResponseReconstructionConfig
    baseline_reconstruction: BaselineReconstructionConfig
    dbp_reconstruction: DBPReconstructionConfig
    validators: ValidatorsConfig
    state_field: StateFieldConfig
    boundary: BoundaryConfig
    open_set: OpenSetConfig

    @model_validator(mode="after")
    def enforce_track_isolation(self) -> "RunConfig":
        if self.track == "A" and self.published_targets is not None:
            raise TrackIsolationError("Track A cannot define published_targets")
        if self.track == "B" and self.published_targets is None:
            raise TrackIsolationError("Track B requires published_targets")
        return self

    def parameter_records(self) -> tuple[ParameterRecord, ...]:
        records: list[ParameterRecord] = []
        explicit_sections = ("generators", "judge", "datasets", "regimes", "paper")
        implementation_sections = (
            "transport",
            "residuals",
            "graph",
            "response_reconstruction",
            "baseline_reconstruction",
            "dbp_reconstruction",
            "validators",
            "state_field",
            "boundary",
            "open_set",
        )
        payload = self.model_dump(mode="json")
        for section in explicit_sections:
            records.extend(
                _flatten_records(section, payload[section], ProvenanceClass.PAPER_EXPLICIT)
            )
        for section in implementation_sections:
            records.extend(
                _flatten_records(
                    section,
                    payload[section],
                    ProvenanceClass.IMPLEMENTATION_SETTING,
                )
            )
        records.append(
            ParameterRecord(
                path="track",
                value=self.track,
                provenance=ProvenanceClass.IMPLEMENTATION_SETTING,
            )
        )
        records.append(
            ParameterRecord(
                path="published_targets",
                value=self.published_targets,
                provenance=ProvenanceClass.IMPLEMENTATION_SETTING,
            )
        )
        return tuple(records)


def _flatten_records(
    prefix: str,
    value: Any,
    provenance: ProvenanceClass,
) -> list[ParameterRecord]:
    if isinstance(value, dict):
        records: list[ParameterRecord] = []
        for key in sorted(value):
            records.extend(_flatten_records(f"{prefix}.{key}", value[key], provenance))
        return records
    return [ParameterRecord(path=prefix, value=value, provenance=provenance)]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_mapping(path: Path, active: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in active:
        raise ValueError(f"configuration include cycle: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    include_value = raw.pop("include", ())
    if isinstance(include_value, str):
        includes = (include_value,)
    else:
        includes = tuple(include_value)
    merged: dict[str, Any] = {}
    for include in includes:
        included = _load_mapping(resolved.parent / include, active + (resolved,))
        merged = _deep_merge(merged, included)
    return _deep_merge(merged, raw)


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    try:
        return RunConfig.model_validate(_load_mapping(config_path, ()))
    except ValidationError as error:
        for detail in error.errors(include_url=False):
            cause = detail.get("ctx", {}).get("error")
            if isinstance(cause, TrackIsolationError):
                raise cause from error
        raise
