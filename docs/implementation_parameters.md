# SciPRISM-X Implementation Parameters

## Parameter Classes and Execution Tracks

The core method contract is recorded in [`method_specification.md`](method_specification.md). This document fixes every additional numerical or procedural choice required by the executable implementation. Each value is explicit, validated, and versioned; third-party defaults are not used as hidden method parameters.

- Track A consumes supplied response manifests for all six generators and keeps evaluation and tuning inputs physically separate.
- Track B constructs auditable response records from official benchmark references and fitted generator profiles.
- `fixture_only` is a 12-case interface check; the formal allocation is the 600/2,400 protocol.
- Reference tables are isolated from per-instance computation and enter only the final comparison stage.

## Fixed Implementation Settings

The tables below list every `IMPLEMENTATION_SETTING` used by the code. The sensitivity column identifies which choices require a new validation selection when changed.

### Transport

| Full key | Adopted value | Selection rationale | Potential sensitivity |
|---|---:|---|---|
| `transport.dtype` | `float64` | Reduce numerical drift in iterative transport under small tolerances | Low; speed and memory change |
| `transport.loss_normalizer_percentile` | `.95` | Implement the main text's “validation-set scale” with a robust validation statistic | High; affects PCS and state features |
| `transport.loss_normalizer_floor` | `1e-12` | Prevent division by zero when a loss is always zero on validation | Sensitive only for degenerate channels |
| `transport.epsilon_grid` | `[.01,.03,.05,.10]` | Cover low-to-moderate entropic regularization with a compact grid | High; affects plan smoothness |
| `transport.lambda_constraint_grid` | `[.25,.50,1,2]` | Search a log-scale neighborhood for the constraint term | High; affects soft-violation mass |
| `transport.lambda_structure_grid` | `[.25,.50,1,2]` | Use the same search budget for the structure term | High; affects structure preservation |
| `transport.marginal_tolerance` | `1e-9` | Strict marginal validation under float64 | Medium; affects convergence decisions |
| `transport.max_iterations` | `1000` | Allow sufficient iterations for stabilized Sinkhorn | Medium; affects non-convergence rate |
| `transport.node_cost_weights.evidence_anchor` | `1/6` | Equal weights for six active node channels | Medium |
| `transport.node_cost_weights.modality` | `1/6` | Same as above | Medium |
| `transport.node_cost_weights.motif` | `1/6` | Same as above | High |
| `transport.node_cost_weights.role` | `1/6` | Same as above | Medium |
| `transport.node_cost_weights.scibert_cosine` | `1/6` | When frozen embeddings are unavailable, training-set TF-IDF 1-2-gram cosine supplies a deterministic non-empty proxy | High; revalidate when the frozen encoder route changes |
| `transport.node_cost_weights.variable_equation_anchor` | `1/6` | Equal weights for six active node channels | High |
| `transport.objective_tolerance` | `1e-12` | Strict objective stabilization under float64 | Medium |
| `transport.stable_iterations` | `5` | Avoid stopping after one accidentally small change | Low to medium |
| `transport.unavailable_soft_penalty` | `1.0` | Treat unavailable evidence more conservatively than unresolved evidence | High |
| `transport.unresolved_soft_penalty` | `.50` | Apply a moderate penalty rather than a hard rejection for unresolved evidence | High |
| `transport.greedy_mass_tolerance` / `transport.greedy_numerical_tolerance` | `1e-12` / `1e-9` | Separate mass-validity and zeroing thresholds explicitly for the transport ablation | Medium; transport ablation only |
| `transport.greedy_lp_method` / `transport.greedy_lp_presolve` / `transport.greedy_lp_primal_feasibility_tolerance` / `transport.greedy_lp_dual_feasibility_tolerance` / `transport.greedy_lp_edge_weight_strategy` | `highs-ds` / `true` / `1e-9` / `1e-9` / `dantzig` | Freeze every path-affecting scipy parameter for the typed-greedy feasibility-protection LP | Medium; transport ablation only |

### Residuals, Graphs, and Response Construction

| Full key | Adopted value | Selection rationale | Potential sensitivity |
|---|---:|---|---|
| `residuals.gamma_grid` | `[.10,.25,.50]` | Cover permissive through strict support thresholds | High |
| `residuals.minimum_support_threshold` | `.50` | Use the conservative grid endpoint for acceptance fixtures | High |
| `residuals.missing_penalty_grid` | `[.50,1,2]` | Symmetric weak/medium/strong missing penalties | High |
| `residuals.normalizer_floor` | `1e-12` | Prevent a zero denominator | Sensitive only for degenerate samples |
| `residuals.normalizer_percentile` | `.95` | Robustly suppress extremes while retaining the tail | Medium |
| `residuals.pcs_coefficient_grid` | `[.25,.50,1,2]` | Compact search sharing the 50-trial budget | High |
| `residuals.residual_consistency_beta` | `[-.5]×6` | Non-positive coefficients ensure violations cannot improve consistency | High |
| `residuals.unsupported_penalty_grid` | `[.50,1,2]` | Same scale as the missing penalty | High |
| `graph.explicit_cue_confidence` | `1.0` | Treat explicit symbol/unit/evidence cues as deterministic-rule evidence | Medium |
| `graph.lexical_cue_confidence` | `.8` | Retain uncertainty for lexical cues | Medium |
| `graph.learned_recognizer_threshold` | `.75` | Apply frozen scientific-encoder/motif-prototype cosine only to scientific units without an explicit cue; use a conservative threshold above the common midpoint | Very high; checkpoint and threshold are fixed implementation settings |
| `graph.learned_recognizer_margin` | `.05` | Require minimum separation between the top two motif-prototype families to avoid forced classification of weak structure | Very high; margin is a fixed implementation setting |
| `graph.salience_weights.graph_centrality` | `.25` | Equal weights for four salience components | Medium |
| `graph.salience_weights.inverse_position` | `.25` | Same as above | Medium |
| `graph.salience_weights.motif_confidence` | `.25` | Same as above | Medium |
| `graph.salience_weights.variable_reuse` | `.25` | Same as above | Medium |
| `graph.short_fragment_token_threshold` | `4` | Avoid treating a very short unanchored fragment as an independent motif | Medium |
| `response_reconstruction.fixture_profile.causal_error` | `0` | Keep reference causal relations in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.endpoint_error` | `0` | Keep endpoint answers in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.evidence_synthesis` | `1` | Preserve evidence synthesis in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.length_ratio` | `1` | Do not introduce a defect through length differences | Fixture-only, medium |
| `response_reconstruction.fixture_profile.logical_error` | `0` | Do not inject logical contradictions into acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.modal_evidence_error` | `0` | Preserve modal evidence in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.motif_omission` | `0` | Do not randomly remove motifs from acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.physical_error` | `0` | Preserve physical conditions in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.uncertainty_error` | `0` | Do not additionally damage uncertainty statements in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.unit_error` | `0` | Preserve units in acceptance responses | Fixture-only, high |
| `response_reconstruction.fixture_profile.unsupported_content` | `0` | Do not inject unsupported content into acceptance responses | Fixture-only, high |

### Baseline and DBP Parameters

| Full key | Adopted value | Selection rationale | Potential sensitivity |
|---|---:|---|---|
| `baseline_reconstruction.endpoint_case_sensitive` / `baseline_reconstruction.endpoint_strip_whitespace` / `baseline_reconstruction.endpoint_collapse_whitespace` | `false/true/true` | Freeze normalized answer agreement as exact equality after Unicode case folding, trimming, and collapsing repeated whitespace | Medium; normalization is a fixed implementation setting |
| `baseline_reconstruction.attribution_lowercase` / `baseline_reconstruction.attribution_token_pattern` / `baseline_reconstruction.attribution_empty_overlap_score` | `true/[A-Za-z0-9]+/0` | Use Jaccard over a frozen ASCII alphanumeric token set for attribution overlap; score an empty union as zero | High; tokenization and empty-input behavior are fixed implementation settings |
| `baseline_reconstruction.taxonomy_patterns` | Frozen phrase table for `unsupported/contradiction/unit/causal_direction/overconfidence` | Build a minimal auditable rule table from scientific failure types named by the paper | Very high; taxonomy and phrase table are fixed implementation settings |
| `baseline_reconstruction.taxonomy_no_failure_score` / `baseline_reconstruction.taxonomy_failure_score` / `baseline_reconstruction.taxonomy_lowercase` | `1/0/true` | Normalize taxonomy output so a high score means no failure, with case folding before matching | High; category-to-scalar conversion is a fixed implementation setting |
| `baseline_reconstruction.uncertainty_maximum_entropy` / `baseline_reconstruction.support_clip_to_unit_interval` | `1/true` | Inputs already use normalized entropy; constrain support novelty to the unit interval used by paper metrics | Medium; normalization is a fixed implementation setting |
| `baseline_reconstruction.transport_score_transform` / `baseline_reconstruction.relation_multiset` / `baseline_reconstruction.ranking_descending` | `exp_negative_cost/true/true` | Freeze transport-cost conversion to similarity, relation-multiset Jaccard, and descending ranking instead of using library defaults | High; baseline conversion details are fixed implementation settings |
| `baseline_reconstruction.representation_dimensions` | `256` | Fix hashed bag-of-words capacity and avoid third-party embedding defaults or runtime drift | High; baseline representation is a fixed implementation setting |
| `baseline_reconstruction.task_success_threshold` | `.50` | Use the midpoint when projecting a strongest-baseline scalar to TSR | Very high; endpoint projection is a fixed implementation setting |
| `baseline_reconstruction.matched_context_rotations` | `3` | Construct at most three content-preserving cyclic sentence-order rotations to approximate matched valid contexts | High; context mapping is a fixed implementation setting |
| `baseline_reconstruction.local_judge_component_weights` | `[.20]×5` | For the deterministic local judge route, equally weight five locally verifiable answer/evidence/reasoning/unit/uncertainty components | Very high; frozen external judge outputs may produce different values |
| `baseline_reconstruction.dbp_validation_transport_epsilon` | `.05` | DBP validation cases need a frozen plan for the TP contract before validation-only baseline selection; final cases use the validation-selected Sinkhorn epsilon | Medium; the validation plan does not enter the epsilon-trial objective |
| `baseline_reconstruction.support_neighbors_grid` / `baseline_reconstruction.support_distance_scale_grid` | `[1,3,5,10,20]/[.25,.5,1,2,4]` | Use a local logarithmic grid for generic open-set kNN/centroid, covering local-to-broad neighborhoods and fourfold distance scales | High; search space is a fixed implementation setting |
| `baseline_reconstruction.uncertainty_mode_codes` | `[0,1,2]` | Enumerate confidence, `1-entropy`, and their product as three paper-compatible uncertainty-only forms | High; uncertainty score is a fixed implementation setting |
| `baseline_reconstruction.graph_edit_cost_grid` / `baseline_reconstruction.graph_edit_timeout_seconds` | `[.5,1,2]/2s` | Use a symmetric local grid for node/relation insertion, deletion, and replacement costs, with a fixed per-instance solve limit | High; GED costs and timeout are fixed implementation settings |
| `dbp_reconstruction.signature_cosine_weight` | `.50` | Equal weight for the entity-signature signal in DBP full-pool ranking | High; fusion formula is a fixed implementation setting |
| `dbp_reconstruction.relation_overlap_weight` | `.50` | Equal weight for the incident-relation signal in DBP full-pool ranking | High; fusion formula is a fixed implementation setting |
| `dbp_reconstruction.signature_neighborhood_hops` | `2` | Explicitly include 1/2-hop neighborhood sizes in the entity signature | High; vector layout is a fixed implementation setting |
| `dbp_reconstruction.local_transport_hops` | `2` | Use the reference entity's 2-hop induced local graph for SCS/TP | High; local-graph radius is a fixed implementation setting |

### Validators and State Field

| Full key | Adopted value | Selection rationale | Potential sensitivity |
|---|---:|---|---|
| `validators.bart_mnli.model_id` | `facebook/bart-large-mnli` | Prefer a local frozen NLI model when rules cannot decide; use the deterministic proxy only when the asset manifest is absent | High |
| `validators.bart_mnli.revision` | `d7645e...b8ce` | Pin the external asset version | High; revision is pinned by this implementation |
| `validators.bart_max_length` | `512` | Fix tokenizer truncation explicitly and prevent library-default drift | Medium |
| `validators.learned_fail_threshold` | `.8` | Require high confidence for learned FAIL | High |
| `validators.learned_pass_threshold` | `.8` | Require equal confidence for learned PASS | High |
| `validators.nli_proxy_feature_weights` | `[.20]×5` | When frozen BART-MNLI is unavailable, equally weight five verifiable token/quantity/equation/anchor/non-contradiction features | High; proxy and frozen-checkpoint routes may differ |
| `validators.nli_proxy_contradiction_count_scale` | `2.0` | Two matched terms reach the maximum clipped contradiction proxy | High |
| `validators.model_root` | `data/raw/models` | Read only project-local assets; `scibert/manifest.json` and `bart_mnli/manifest.json` must match model ID, revision, and per-file SHA-256; no implicit online download | Low |
| `validators.scibert_max_length` | `512` | Fix tokenizer truncation explicitly | Medium |
| `validators.scibert.model_id` | `allenai/scibert_scivocab_uncased` | Prefer a local frozen scientific-text encoder; fit TF-IDF on the training split only when the asset manifest is absent | High |
| `validators.scibert_pooling` | `attention_masked_mean` | Exclude padding with the attention mask, mean-pool tokens, then L2-normalize for cosine | High; pooling is fixed by this implementation |
| `validators.scibert.revision` | `24f92d...2fc1` | Pin the external asset version | High; revision is pinned by this implementation |
| `validators.symbolic_timeout_seconds` | `2.0` | Finite deterministic budget for one symbolic validation | Medium |
| `validators.tfidf_proxy.lowercase` / `validators.tfidf_proxy.analyzer` / `validators.tfidf_proxy.ngram_range` / `validators.tfidf_proxy.min_df` / `validators.tfidf_proxy.max_df` / `validators.tfidf_proxy.max_features` / `validators.tfidf_proxy.norm` / `validators.tfidf_proxy.use_idf` / `validators.tfidf_proxy.smooth_idf` / `validators.tfidf_proxy.sublinear_tf` / `validators.tfidf_proxy.binary` / `validators.tfidf_proxy.strip_accents` / `validators.tfidf_proxy.stop_words` / `validators.tfidf_proxy.token_pattern` / `validators.tfidf_proxy.dtype` | Word analyzer, lowercase, `1-2` grams, `min_df=1`, `max_df=1`, no stop words or feature cap, L2, IDF+smooth, non-sublinear/non-binary, default Unicode word-token regex, float64 | Fully freeze sklearn behavior when SciBERT is unavailable, without reading test text or relying on third-party defaults | High; proxy and frozen-encoder routes may differ |
| `state_field.activation` | `gelu` | Smooth activation for a small MLP | Medium |
| `state_field.adamw_betas` | `[.9,.999]` | Explicit AdamW momentum | Medium |
| `state_field.adamw_eps` | `1e-8` | Explicit AdamW numerical term | Low |
| `state_field.batch_size` | `64` | Moderate batch size reproducible on CPU | Medium |
| `state_field.class_weighting` | `inverse_frequency_mean_one` | Reduce three-class imbalance while stabilizing overall scale | High |
| `state_field.deterministic_shuffling` | `true` | Seed fully determines batch order | Low |
| `state_field.dimensions` | `[15,64,16]` | Compact encoder for the printed 15-dimensional state | High |
| `state_field.dropout` | `0` | Remove unspecified stochastic regularization | Medium |
| `state_field.early_stopping_min_delta` | `0` | Count every strict validation improvement | Medium |
| `state_field.energy_dimensions` | `[31,32,1]` | Compact head over concatenated 16-dimensional embedding and 15-dimensional state | High |
| `state_field.energy_loss_weight_grid` | `[.25,.50,1]` | Search the relative classification/energy-loss scale | High |
| `state_field.energy_margin_grid` | `[.50,1,2]` | Log-scale neighborhood around a unit margin | High |
| `state_field.gelu_approximate` | `none` | Freeze exact GELU rather than a PyTorch default | Medium |
| `state_field.linear_bias` | `true` | Retain explicit bias in encoder, regime head, and energy head | High |
| `state_field.initialization` | `kaiming_uniform_linear_default` | Explicitly implement `a=sqrt(5)`, fan-in Kaiming uniform, and corresponding uniform bias while preserving the existing seeded Linear realization | High; initialization was not disclosed |
| `state_field.learning_rate_grid` | `[3e-4,1e-3,3e-3]` | Standard three-point log-scale neighborhood for a small MLP | High |
| `state_field.optimizer` | `adamw` | Explicit, stable, and supports independent weight decay | Medium |
| `state_field.dropout_inplace` | `false` | Freeze non-inplace dropout; no layer is created while `dropout=0` | Low |
| `state_field.weight_decay_grid` | `[0,1e-4,1e-3]` | Three-point search from no regularization to light regularization | Medium |

### Boundaries, Open Set, and Calibration

| Full key | Adopted value | Selection rationale | Potential sensitivity |
|---|---:|---|---|
| `boundary.context_composition_steps` | `[1,2,3]` | Discrete weak/medium/strong intervention | High |
| `boundary.dbp_edge_removal_rule` | `ceil_fraction_preserve_one` | Ensure a real intervention on small graphs without deleting all edges | Medium |
| `boundary.dbp_removal_ratios` | `[.05,.10,.20]` | Weak/medium/strong structural deletion | High |
| `boundary.dbp_seed_fractions` | `[.25,.20,.10]` | Progressively weaken the published `.30` baseline | High |
| `boundary.evidence_removal_rule` | `ceil_fraction_preserve_one` | Produce an auditable intervention in small samples while retaining one required evidence item | High |
| `boundary.mad_consistency_factor` | `1` | Use the validation MAD scale directly | Medium |
| `boundary.mad_floor` | `1e-12` | Prevent division by zero on a constant dimension | Sensitive only for degenerate dimensions |
| `boundary.seed_alignment_rounding` | `nearest_integer` | Map a fraction to a discrete seed count with minimum deviation | Medium |
| `boundary.standardized_radii` | `[.25,.50,1]` | Continuous weak/medium/strong standardized radii | High |
| `boundary.symbolic_depth_steps` | `[1,2,3]` | Discrete weak/medium/strong symbolic depth | High |
| `open_set.calibration_bins` | `10` | Match the fixed equal-width-bin protocol | Medium |
| `open_set.calibration_minimum_bin_fraction` | `.01` | Paper low-support flagging rule | Low; explicit in prose but absent from the primary explicit configuration |
| `open_set.calibration_minimum_bin_samples` | `20` | Paper low-support flagging rule | Low; same as above |
| `open_set.cluster_counts` | `[2..10]` | Small-to-moderate candidate failure-cluster counts | High |
| `open_set.cluster_selection` | `silhouette` | Validation-only selection without test labels | High |
| `open_set.kmeans_algorithm` | `lloyd` | Freeze the sklearn implementation path | Medium |
| `open_set.kmeans_init` | `k-means++` | Reduce the probability of poor initial centers | Medium |
| `open_set.kmeans_max_iterations` | `300` | Explicit iteration limit | Low to medium |
| `open_set.kmeans_n_init` | `20` | Reduce variance from one initialization | Medium |
| `open_set.kmeans_tolerance` | `1e-4` | Explicit convergence threshold | Low to medium |
| `open_set.novelty_normalization_quantile` | `.95` | Robust normalization of three novelty components | Medium |
| `open_set.separation_floor` | `1e-12` | Prevent zero within-cluster dispersion | Sensitive only for degenerate clusters |
| `open_set.unknownness_weights` | `[1/3,1/3,1/3]` | Equal weights for support, residual, and motif direction | High |
| `open_set.updq_weights` | `[1/3,1/3,1/3]` | The paper states equal unless otherwise specified | Medium |
| `track` | `A` or `B` | Keep supplied-response and fitted-profile paths separate | Very high; determines response-source semantics |

### Formal-Scale Runner

| Full key | Adopted value | Selection rationale | Potential sensitivity |
|---|---:|---|---|
| `formal.split_fractions` | `[0.70,0.15,0.15]` | If an official split is unavailable, assign original cases excluded from every formal allocation to train/validation/unused-test candidate pools using task-ID SHA-256 percentile buckets; the formal 600/2400 set is not internally split | High; affects tuning and generalization |
| `formal.smoke_nonstress_cases` | `6` | Fast tests still cover all three non-stress regimes | Smoke only |
| `formal.smoke_scicu_boundary_cases` | `15` | Exactly cover 5 perturbation families × 3 strengths | Smoke only |
| `formal.smoke_dbp_boundary_cases` | `36` | Exactly cover 3 language pairs × 4 perturbation families × 3 strengths | Smoke only |
| `formal.response_profile_entropy_weight` | `0.01` | Select the maximum-entropy identifiable solution when generator-stratified constraints are unavailable | High |
| `formal.stable_pcs_quantile` | `0.67` | Define the high-PCS region on validation | High |
| `formal.stable_residual_quantile` | `0.33` | Define the low-residual region on validation | High |
| `formal.invalid_pcs_quantile` | `0.33` | Define the low-PCS region on validation | High |
| `formal.invalid_residual_quantile` | `0.67` | Define the high-residual region on validation | High |
| `formal.proxy_pass_threshold` | `0.80` | Require high-confidence PASS even without a frozen model | High |
| `formal.proxy_fail_threshold` | `0.80` | Require high-confidence FAIL even without a frozen model | High |
| `formal.resume` | `true` | Resume a formal-scale run by content hash | Does not affect numerical values |

## Execution Policies and Derived Conventions

| Undisclosed or unrecoverable item | Current handling | Rationale | Impact/sensitivity |
|---|---|---|---|
| `workflow.fixture_grid_selection` | A 12-case acceptance run uses the first value from each frozen validation grid and writes it to `implementation_settings.json` | A fixture has no validation corpus for retuning, and hidden defaults are prohibited | High; formal experiments must select values on validation |
| `workflow.fixture_scope` | Explicitly acceptance-only; does not replace the formal 600/2,400 allocation | Prevent reporting a reduced acceptance run as a formal result | Does not affect the algorithm; determines conclusion boundary |
| `workflow.infeasible_transport_mapping` | Preserve the hard constraint and map infeasible transport to `INVALID`; `loss_defined=false`, PCS/consistency are 0, and normalized residuals are 1 | Do not soften constraints to manufacture evaluable results | High; affects INVALID fraction |
| Empty-list SciCUEval answers | Normalize three official MatTab `answer: []` entries to the string `[]` | Preserve the original “no matching item” empty-set semantics without inventing material IDs, while ensuring a non-empty downstream reference string | Low; affects only empty-set serialization |
| DBP15K nominal-strength mapping | Map `0.25/0.50/1.00` to edge/relation intervention ratios `0.05/0.10/0.20` and seed-retention ratios `0.25/0.20/0.10` | One-to-one mapping to the frozen weak/medium/strong intervention settings | High; affects boundary curves |
| DBP15K relation-perturbation materialization | Ambiguity maps selected edges in both graphs to one new relation; cross-graph weakening maps them to two different new relations | The published specification names the perturbation families but not the discrete triple transformation; these rules implement relation-merging ambiguity and cross-graph evidence mismatch without changing entities | High; discrete transformation is a fixed implementation setting |
| Invalid-perturbation exclusion and replacement | Check all three frozen radii jointly by `bsi_regime × family × trajectory` for SciCUEval and `language × family × trajectory` for DBP15K. If any radius is invalid, select the first all-radius-valid, non-duplicate case in SHA-256 order from the same frozen base/reference pool and save requested/rejected/resolved IDs | Implements Appendix C.3 exclusion before formal evaluation while preserving fixed paper sample counts and paired radii | Medium; exact replacement manifest was not disclosed |
| Track A response entry | `--response-manifest` must provide separate evaluation/tuning files for all six generators. The primary response matrix uses supplied content only and does not read published targets. When a boundary-original response is absent, internal paired responses use the neutral `response_reconstruction.fixture_profile` and disclose each field in `target_derivation`. `--paper-reference` connects only to post-statistics comparison | Keeps Track A independent from Track B `published_targets` while retaining complete BSI original-end pairing and comparison output | High; supplied boundary-original responses take precedence over the neutral profile |
| Six-generator profile stratification | All six models share one `maximum_entropy_pooled` profile because generator-level cells were not published | Do not invent model differences; use the Appendix worked case's six-channel residuals, endpoint, PCS, BSI, and secondary gaps as public constraints | Very high; refit per model when stratified results are available |
| Profile entropy regularization | Apply negative entropy with weight `0.01` to the first nine Bernoulli defect rates; exclude `length_ratio` and `evidence_synthesis` | Select the maximum-entropy, minimum-added-information identifiable solution under published constraints | High; affects fitted response distribution |
| Formal response seed | `derive_seed(paper_seed, "formal-response::{task_id}::{model_id}::{replicate}")` | Fully isolate task, model, and replicate randomness deterministically | Does not affect expectations; affects individual realization |
| Evaluator response anonymization | Motif/validator/transport internals receive an immutable copy with `model_id=anonymized_generator`; the original model ID remains only in outer paired-aggregation rows | Implements generator-identity isolation from §§3.2-3.3 and prevents source IDs from indirectly carrying model names | Does not affect scores; affects audit isolation |
| Transport entropy in the 15-dimensional state | `-sum(P_ij log P_ij)` over positive mass only | The paper names the feature but gives no discrete formula; use standard Shannon entropy | Medium |
| Motif-signature discrepancy | Total-variation distance between task/response six-family motif-frequency distributions | Bounded in `[0,1]` and directly expresses motif-composition difference | High; feature definition is a fixed implementation setting |
| Graph complexity | `number of nodes + number of edges` | The paper gives no normalization; use the smallest auditable graph-size statistic | High |
| CCS matched contexts | Using `baseline_reconstruction.matched_context_rotations=3`, create at most three content-preserving cyclic sentence-order rotations of the same valid reference context, recompute PCS for each, then calculate `1-mean(abs(delta PCS))` as in Appendix E.2 | The published formula does not fix paired context files; sentence-order rotations preserve the evidence set and create no new facts | High; context mapping is a fixed implementation setting |
| Validation objective for method coefficients | Minimize structural-validity Brier score over 50 joint configurations on validation; break ties by higher feasible rate and then earlier trial; freeze the selected configuration for test, boundary, and DBP paths | The published contract requires validation selection and freezing; Brier score is the fixed proper scoring rule used here | High; objective is a fixed implementation setting |
| Validation scale for transport loss | For every candidate transport trial, take the validation `.95` quantile separately for raw node, structure, and constraint losses with a `1e-12` floor; PCS and the 15-dimensional state use the frozen normalized losses | The main text requires graph-size and validation-set scaling but does not disclose the statistic. Unit transport marginals handle graph size; the `.95` quantile provides a robust scale | High |
| Residual normalizers | Take the validation `.95` quantile per raw residual channel for the selected method configuration, apply a `1e-12` floor, then freeze | Implements validation scaling and freezing while preventing test information from entering scale estimation | High |
| State-field trial count | `learning_rate × weight_decay × energy_loss_weight × energy_margin` yields 81 unique combinations. For each seed, draw 50 distinct configurations with the fixed seed, train each independently, and select by validation loss | Matches the published 50-trial budget per learned method and prohibits reusing duplicate configurations to inflate the trial count | Medium; the four grids are fixed implementation settings |
| Regime-quantile tie fallback | If validation quantiles cannot produce three classes, stably rank by `PCS - mean(residual)` and split into thirds | Ensure all three training classes exist; include task/model/replicate in the sort key for deterministic tie breaking | High |
| Open-set pseudo-group | `argmax(residual) × motif_dim + argmax(abs(motif_mismatch))`; build code-to-group mapping on validation only and map unseen codes to one frozen unseen group | The paper requires validation-derived pseudo-groups but gives no finer mapping | High |
| Open-set motif centers and regime outputs | Compute and freeze residual/motif centers only on validation. At formal scale, output UPDQ/SC separately for test candidates in all four regimes; label smoke explicitly as `pooled_smoke` | Prevent unknown/test self-centering leakage and match the regime-specific results table in the main text | High |
| Structural confidence for SC | Use PCS clipped to `[0,1]` as SciPRISM-X structural confidence; baselines use their own permitted scores | The paper provides the `1-ECE` formula but no separate confidence head or per-instance confidence file. PCS is the only frozen scalar directly representing structural validity | High |
| Open-set single-point degeneracy | When the joint validation features have only one distinct point, record `degenerate_single_cluster=true`, choose one cluster, set silhouette and separation to 0, and set stability to 1 when all five seeds have identical labels | Silhouette is mathematically undefined for one cluster. The bounded single-cluster convention preserves finite values without presenting degenerate samples as normal multi-cluster results | Sensitive only for degenerate input |
| Separation normalization | Divide each unknown seed's separation by the validation seed-wise `.95` separation quantile, clip to `[0,1]`, then average over five seeds | Raw separation is unbounded, while UPDQ requires the same scale as Purity/Stability | High |
| Cluster-count selection | Compute silhouette on validation independently for each of the five paper seeds, choose candidate k with the highest mean silhouette, and break ties toward smaller k | Do not use unknown/test data to select k | Medium |
| DBP entity signature | Seven degree/neighborhood structural quantities plus bidirectional relation frequencies; scale structural quantities by the maximum across both graphs, then L2-normalize the full vector | The paper states relation-frequency and 1/2-hop signatures but not the vector layout | High |
| DBP non-stress regime allocation | Allocate 200 cases per language pair. For domain transfer, treat the pair as a rotating held-out domain and sample by frozen hash. For structure split, select remaining pairs with the largest cross-graph log-degree and incident-relation-cardinality displacement. Hash-sample IID from the remaining pool; the three are disjoint | The published contract gives regime names, `n=600` per regime, and the original split but no DBP regime manifest. This rule keeps structure/domain subsets distinct from IID sampling | Very high; allocation manifest is a fixed implementation setting |
| DBP full-pool ranking score | `dbp_reconstruction.signature_cosine_weight × signature cosine + dbp_reconstruction.relation_overlap_weight × incident-relation Jaccard`, currently `.5/.5` | Equally weight the two published structural signals and stably rank the complete target-entity pool; weights are explicit provenance-bearing configuration | High |
| DBP local transport graph | 2-hop induced neighborhood of the reference entity with uniform mass on all nodes. Only official 30% seed alignments enter method-side soft admissibility; final reference pairs construct only the frozen SCS/TP evaluation mask | Ranking still uses the complete 15,000-candidate pool, and final correspondence truth cannot feed back into the transport objective | High |
| DBP SCS scalarization | `exp(-(Lnode + lambda_str Lstr + lambda_con Lcon))`, excluding the entropy regularizer from the reported score | The paper gives no independent SCS equation but states that its claim depends on constrained correspondence and that Hits@1/MRR are auxiliary ranking diagnostics. This matches `exp(-cost)` used by transport baselines | High |
| DBP structure-cost evaluation | Rewrite the literal fourfold sum exactly as `A²·a + B²·b - 2A·P·Bᵀ`, precompile adjacency relations as CSR, and verify small matrices elementwise against the literal formula to `1e-14` | Direct broadcasting would allocate about 124 GiB for a 619×208 local graph. The matrix identity changes neither objective, parameter, nor float64 precision | Does not affect the mathematical definition; changes executability and floating-point summation order |
| DBP experimental-unit primary key | Store `allocation_id/regime/family/strength` for every DBP result; include allocation ID in ablation and baseline case IDs | The same entity pair can recur across regimes or perturbation allocations; using only the pair would incorrectly merge paired samples | Does not affect one metric row; affects paired-statistics correctness |
| DBP boundary input | A boundary allocation must provide the corresponding `MaterializedDBPPerturbation`; fail immediately when missing | Prohibit silently generating a “boundary” score on the original graph | Does not affect values with complete input |
| Boundary base-pool reuse | Each SciCUEval `bsi_regime` draws base cases from its corresponding non-stress 600-case pool; boundary stress draws from the union of the three pools. DBP15K draws within language from the union of frozen reference-pair pools across the three non-stress regimes | Matches §3.10 “boundary cases constructed from the same base evaluation pool” and avoids sampling a separate easier/harder set | Medium; allocation IDs are generated by the fixed manifest procedure |
| SciCUEval four-regime BSI stratification | At formal scale, stratify 2,400 boundary-stress probe instances into four BSI reporting strata of 600 each, then distribute each equally across 5 perturbation families × 3 strengths (40 per cell). Each stratum independently forms right-censored boundary trajectories and pairs with its own baseline. The 15 smoke cases are labeled only `pooled_smoke` | The primary table reports BSI by regime, while `n=600/600/600/2400` does not uniquely determine probe-to-trajectory assignment. This design neither duplicates scores nor exceeds the formal 2,400-probe total | Very high; stratum manifest and probe allocation are fixed implementation settings |
| Response-source state | Track B constructs responses from official references and fitted profiles; route/profile details enter only internal manifests | Keep response provenance explicit and machine-readable | Numerical results are highly sensitive |
| Identifiability of profile parameters from published targets | Deterministically fit only paper-observable aggregates and save targets, weights, and traces | Multiple hidden parameter sets may produce the same aggregates | High; parameter uniqueness is not claimed |
| Worked-case target mapping | Build fixed acceptance targets from Appendix E.5 endpoint, PCS, BSI, motif statuses, and transport/residual fields | Check the data flow with a public case | Fixture-only; does not represent global fitting |
| Grid values for acceptance fixtures | In a 12-case run without a validation corpus, use the first value of each configuration grid and record the resolved configuration | Avoid hidden library defaults | High; formal experiments must select on validation |
| Infeasible hard mask | Map to `INVALID`, save diagnostics, and do not relax the constraint | Matches the semantics of a scientifically inadmissible match | May increase INVALID count |
| Structural-relation channel aggregation | Average relation-level structural costs | The paper does not specify a multi-relation aggregator | Medium to high |
| Node role | Derive from deterministic motif-family rules | Provides a complete role annotation from the frozen motif protocol | Medium |
| SciBERT node similarity | When frozen assets are absent, fit word 1-2-gram TF-IDF on the training split and output cosine; reweight six node-cost channels equally to `1/6` | Ensure the channel participates without reading test text | High |
| Learned-validator model revision | When frozen BART-MNLI files are absent, use the equal-weight five-feature deterministic NLI proxy with thresholds `0.80/0.80` | The paper gives no exact checkpoint/revision, while formal results retain no `UNAVAILABLE` value | High |
| Learned motif recognizer | When neither rule nor dictionary cue matches, compute cosine between frozen SciBERT (or training-split-only TF-IDF when absent) and six motif prototypes. Require the top score to reach `.75` and exceed the second family by `.05` | Implements the published rule-first order and prohibits final-test fine-tuning | Very high; corpus, labels, checkpoint, prototypes, and thresholds are fixed implementation settings |
| Specific baseline implementation version | When the published specification names only a family, use the smallest concept-compatible adapter with all behavior exposed in configuration | Avoids implicit third-party defaults and unverifiable repository assumptions | High |
| DBP15K comparison-system outputs | The deterministic concept-compatible adapters use the same `0_3` split under contract key `dbp15k_prior_system_outputs` | Makes structural and auxiliary ranking paths executable under one explicit interface | Very high; supplied per-instance comparison outputs can replace adapter outputs through the same interface |
| Strongest-baseline multi-metric projection | Use structure-aware score directly as PCS; form TSR with `baseline_reconstruction.task_success_threshold=.5`; form CCS as `1-|score-group_mean|` within the same original-task matched group; compute BSI crossing at the same perturbation radii using validation `.33/.67` score states; cluster baseline score only for UPDQ; compute SC as `1-ECE` for the strongest method named in published tables (uncertainty in the first three regimes, structure-aware at the boundary) | These derivations read only the baseline's own score, permitted validation support, and final reference labels, never SciPRISM-X residual/state/boundary/open-set outputs | Very high; per-instance multi-head baseline outputs can replace this projection through the same schema |
| LLM judge route | `--judge-response-file` can replay complete six-field structured outputs from fixed `gpt-4o-2024-08-06` by baseline `case_id`; the file SHA-256 enters baseline-stage input and `baseline_selection.json`. If absent, use deterministic local inference reading only task, response, and reference context, aggregate with `baseline_reconstruction.local_judge_component_weights=[.2]×5`, and save the actual route | Allows frozen judge outputs to replace the local route without leaking residual, mask, state, boundary, open-set, or generator identity into the baseline | High; local and frozen-external routes may differ |
| Local judge formula | Equal mean of normalized token F1, reference-token coverage, negation consistency, numeric/unit coverage, and uncertainty handling. Thresholds are explicitly `.80/.80`; diagnostics use the published answer/evidence/reasoning/uncertainty/error-type field names | Matches the published judge rubric. Unit consistency contributes only to local `final_score` and is not exposed as an extra structured field | High |
| Formal baseline parameters | Endpoint/attribution/taxonomy/relation/ranking have no trained parameters and run deterministically once. Support, uncertainty, GED, Sinkhorn, Gromov, and structure-aware methods use explicit grids and each receives 50 validation trials under Appendix D.3. SciCUEval uncertainty/GED/structure-aware minimize structural-validity Brier; kNN/centroid support minimizes open-set-novelty Brier; DBP maximizes alignment similarity on a seed-pool holdout. If fewer than 50 candidate combinations exist, reshuffle the candidate grid repeatedly with the same frozen RNG. All seven tunable baseline families must have a selection result; selected parameters, complete fixed/grid configuration, and DBP configuration are written to `baseline_selection.json` | Satisfies validation-only selection and the 50-trial learned-method budget while keeping fixed values, repeated candidates, and objectives explicit | Very high; grid, objective, and split are fixed implementation settings |
| DBP baseline validation split | Hold out `15%` from the official 30% seed-alignment pool by frozen SHA-256 order for shared validation; use remaining seeds as method-side evidence. Keep the full target pool unchanged and exclude final reference pairs from selection | Aligns with the validation fraction in the global `70/15/15` allocation and D.3 fairness controls | High |
| Neutral replacements for eight ablations | Removing motif disables the structure term; removing hard constraints opens the mask; removing soft penalties sets them to zero; residual/state/boundary/open-set use their corresponding neutral branches; removing transport uses typed greedy. Each policy changes one control and recomputes every downstream dependency | Uses the minimum single-module difference while preserving per-case pairing | Very high; ablation policies are fixed implementation settings |
| Typed-greedy feasibility protection | Sort by node cost with deterministic row/column tie breaking. At each allocation, use HiGHS dual simplex (primal/dual tolerance `1e-9`, Dantzig edge weight) to maximize the current edge while keeping remaining typed marginals feasible | Direct lowest-cost filling can incorrectly consume a scarce target under a feasible hard mask. The feasibility LP protects only marginals and does not optimize global total cost | Medium; specific to the no-transport ablation |
| Bootstrap resampling seed | Use the first paper seed, `42`, for each comparison. First average direction-adjusted differences for repeated responses within the same task/pair ID, then resample unique IDs 1,000 times | The paper gives five experimental seeds but no separate bootstrap seed. Use the first frozen seed and save it in every row | Low to medium |
| Reference-value isolation | Force `source=computed` for bootstrap/Holm rows; published table values cannot be inserted directly into per-instance metrics or statistical rows. If response files are missing, Track B constrains the pooled response profile with the Appendix worked case, then follows the ordinary response→evaluation→statistics path. `published_metrics.json` enters the difference report only after statistics; its full use is recorded in `implementation_contract.json` | Keeps fitted-profile constraints separate from computed metrics and statistics | Very high; Track A uses supplied responses directly |
| Formal stage cache key | SHA-256 over canonical JSON containing stage name, sorted input hashes, and implementation hash; content hash is SHA-256 over the relative-path-to-file-SHA-256 mapping excluding the receipt | Safely resume identical inputs; any upstream or implementation change creates a new stage; timestamps do not enter numerical cache keys | Does not affect numerical values |
| Formal-stage atomicity | Use a fixed `.partial` directory, `fsync` every file after the producer completes, write and `fsync` the receipt, then finalize with `os.replace`. On the next run, clean a failed partial only under the exact store root | Windows does not reliably support `fsync` on a read-only handle, so use an `r+b` handle without changing content | Does not affect numerical values; affects crash recovery |
| Formal split isolation | Treat the complete formal `600/2400` allocation as final test. From original cases never entering a formal allocation, draw train and validation with frozen SHA-256 `70/15/15` rules, balance each by domain×competency, and use 600 cases per pool at formal scale | The method requires validation-only selection and final-test freezing. The same 600-case scale as each non-stress regime is used, and every formal case is prohibited from fitting, thresholds, or support sets | Medium; split manifest is a fixed implementation setting |
| Formal artifact schema | Require 32 `computed/`, 7 `release/`, 3 `figures/`, 1 `paper_reference/`, and 1 `comparison/` artifacts. Include all per-instance graph/validator/transport/residual/state foreign keys, effective hyperparameters, learned modules, environment, protocol, and plot manifests in the final receipt. Hard-link/copy only files declared by verified stage receipts | Exposes tuning pools, formal final evaluation, the Deliverables 1-8 contract, external-input routes, effective parameters, and separation of computed and reference values | Does not affect numerical values; affects output completeness |
| Unperturbed BSI/UPDQ fixture | Do not fabricate it from constants or published targets; leave it not applicable when no required perturbation/unknown set exists | The metric requires additional experimental units | The corresponding column cannot be compared with paper table values |
| Plot fonts, colors, and line styles | Use a deterministic pure-SVG plotting layer, explicit `configs/plotting.yaml`, a computed-only input gate, and input/style/output hashes. The colorblind-friendly palette is the fixed default | Keeps appearance fully versioned and independent from numerical computation | Does not affect numerical values; affects appearance |
| PCS direction tension | Preserve the positive node term in the printed primary formula; `direction_audit` emits warnings only | Do not unilaterally “correct” the paper | Interpretation is highly sensitive |
| BSI naming/direction | Give explicit names when storing both raw fragility (lower is better) and a direction-adjusted downstream quantity | Avoid plot-direction confusion | Medium |
| Windows/CPU reference environment | CPython `3.11.9`, PyTorch `2.15.0`, float64, deterministic algorithms; no Anaconda | Current independently verifiable runtime environment | GPU/other BLAS implementations may differ slightly |

## Configuration Discipline

1. Every method-affecting value must be explicit in configuration or selected by a recorded validation procedure.
2. The formal workflow confines every parameter grid, cluster choice, threshold, prototype, and baseline-support set to independent train/validation artifacts.
3. The complete 600/2,400 allocation enters only final evaluation, statistics, and comparison.
4. Any change to an implementation setting requires a new configuration hash, validation-selection artifact, and downstream stage receipts.
5. Smoke scale changes only allocation counts; it retains the same algorithms, explicit hyperparameters, five state seeds, complete DBP candidate pool, ablation policies, and 1,000-resample statistical contract.
