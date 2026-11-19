# SciPRISM-X Method Specification

This document lists the complete SciPRISM-X method contract. Published settings retain the runtime provenance label `PAPER_EXPLICIT`; settings required to make an underspecified component executable are listed in [`implementation_parameters.md`](implementation_parameters.md) and retain the label `IMPLEMENTATION_SETTING`. Both classes are concrete, validated inputs rather than library defaults.

## 1. Response-Generation and Evaluation Roles

Sources: Manuscript §3.2 and Table 1; Appendix §D.2 and Table S12.

| Item | Published specification | Implementation key | Check |
|---|---|---|---|
| Response generator 1 | `gpt-4o-2024-08-06` | `generators.model_ids[0]` | Match |
| Response generator 2 | `claude-3.5-sonnet-20241022` | `generators.model_ids[1]` | Match |
| Response generator 3 | `gemini-1.5-pro-002` | `generators.model_ids[2]` | Match |
| Response generator 4 | `Llama-3.1-70B-Instruct` | `generators.model_ids[3]` | Match |
| Response generator 5 | `Qwen2.5-72B-Instruct` | `generators.model_ids[4]` | Match |
| Response generator 6 | `Mistral-Large-2411` | `generators.model_ids[5]` | Match |
| Generation temperature | `0.0` | `generators.temperature` | Match |
| Maximum output length | `2048` tokens | `generators.max_output_tokens` | Match |
| Responses per task | `3` | `generators.responses_per_task` | Match |
| Evaluation period | `2024-10` through `2025-01` | `generators.evaluation_period` | Match |
| Anonymization | Evaluators cannot see generator identity, provider, model family, access method, or decoding metadata | Workflow input isolation | Implemented; verification contract in `tests/baselines/` |
| LLM judge model | `gpt-4o-2024-08-06` | `judge.model_id` | Match |
| LLM judge dataset | SciCUEval only; not DBP15K | `judge.dataset=scicueval` | Match |
| LLM judge temperature | `0.0` | `judge.temperature` | Match |
| LLM judge top-p | `1.0` | `judge.top_p` | Match |
| LLM judge maximum output | `512` tokens | `judge.max_output_tokens` | Match |
| LLM judge information boundary | Receives only the task, response, and permitted reference context; cannot see generator identity or internal SciPRISM-X quantities | Baseline typed projection | Implemented; verification contract in `tests/baselines/` |
| LLM judge structured output | `answer_quality/evidence_support/reasoning_consistency/uncertainty_handling/final_score/brief_error_type`; only `final_score` is used as the scalar score | `PAPER_JUDGE_FIELDS` + `--judge-response-file` | Frozen outputs can be replayed by `case_id` and file hash; otherwise the configured deterministic local route is used |

## 2. Data, Roles, Samples, and Evaluation Regimes

Sources: Manuscript §3.1, §3.3, and Table 2; Appendix §C.1 and Tables S6-S7; §D.4 and Table S13.

| Item | Published specification | Implementation key | Check |
|---|---|---|---|
| SciCUEval | Primary scientific-reasoning benchmark | `datasets.scicueval.role=primary` | Match |
| DBP15K | Auxiliary structural-correspondence stress test | `datasets.dbp15k.role=auxiliary` | Match |
| DBP15K language pairs | `ZH-EN`, `JA-EN`, `FR-EN` | `datasets.dbp15k.language_pairs` | Match |
| DBP15K original seed fraction | `0.30` | `datasets.dbp15k.seed_fraction` | Match |
| DBP15K original split | Reuse the original `0_3` train/test split with identical entity IDs | Fail-closed loader for `0_3/sup_ent_ids` + `0_3/ref_ent_ids` | Static contract validates the 30% fraction, no seed/reference overlap or duplicates, and existence of every entity ID |
| DBP15K candidate set | Full candidate pool | `datasets.dbp15k.full_candidate_pool=true` | Match |
| DBP15K candidate pruning | Disabled | `datasets.dbp15k.candidate_pruning=false` | Match |
| Evaluation regimes | `iid`, `motif_or_structure`, `domain_transfer`, `boundary_stress` | `regimes.names` | Match |
| Non-stress sample count | `600` sampled cases per regime | `regimes.nonstress_cases` | Match |
| Boundary-stress sample count | `2400` perturbed cases | `regimes.boundary_stress_cases` | Match |
| SciCUEval formal response matrix | `(600×3 + 2400) × 6 generators × 3 replicates = 75,600` | `build_paper_manifest(...).scicu_response_count` | Defined by the formal allocation contract |
| DBP15K formal case matrix | `600×3 + 2400 = 4,200` | Formal DBP allocation manifest | Defined by the formal allocation contract |
| Reuse rule | Compatible methods use identical instances, splits, perturbations, and scoring protocols | Workflow/baseline isolation | Interface constraint implemented |

## 3. Method Structure and Validation Order

Sources: Manuscript §3.3-§3.8; Appendix §§A-B.

| Item | Published requirement | Implementation location | Check |
|---|---|---|---|
| Six motif families | symbolic transformation, unit/dimensional consistency, causal-mechanistic relation, evidence binding, modality grounding, uncertainty qualification | `src/sciprism_x/motifs` | Match |
| Six residual channels | physical, logical, unit/dimensional, causal, modal/evidential, uncertainty | `src/sciprism_x/residuals` | Match |
| Reference-graph inputs | task, reference/target, scientific context, table/figure/record/graph evidence | Motif graph builder | Match |
| Validation order | Explicit rules and scientific constraints first; use a learned recognizer only for implicit or weak expression | Frozen registry lists 8 deterministic validators followed by 1 entailment fallback; startup enforces the same runtime order | Match |
| Implicit motif recognition | Weak or implicit structure is routed through a learned recognizer; the published specification does not fix training details or thresholds | Only a unit without an explicit cue uses SciBERT/prototype cosine; threshold `.75`, margin `.05`; training-split-only TF-IDF is used if frozen assets are unavailable | Concrete implementation setting; fully disclosed in `learned_modules.json` and the implementation-parameters document |
| Validator thresholds | Thresholds are selected on validation and frozen; the published specification does not fix their values | Learned-entailment PASS/FAIL thresholds are both `.80`; symbolic timeout, route, model revision, and prohibited inputs are recorded in the registry/effective-parameter artifact | Concrete implementation setting |
| Transport | Constraint-aware entropic graph transport; node, structure, and constraint terms; stabilized Sinkhorn | `src/sciprism_x/transport` | Literal implementation; unit mass normalizes graph size, while three validation scales are frozen separately |
| Hard constraints | Inadmissible matches have zero mass; an infeasible problem must not be silently relaxed | Hard mask / INVALID path | Match |
| Residuals | Store matched violation, missing required structure, and unsupported response content by channel | Residual schema | Match |
| State vector | 15 features; stable / brittle / invalid | `src/sciprism_x/state_field` | Match |
| Tuning freeze | Recognizers, thresholds, masks, perturbation strength, regime thresholds, and open-set prototypes are selected on validation and then frozen | `method_selection.json` + state/open-set manifests | Match; the three loss scales, selected parameters, and six-channel residual scales are auditable for every trial |

The published specification leaves parsing thresholds, network widths, optimizer details, Sinkhorn tolerances, and several other numerical settings open. The fixed values used here are listed in the implementation-parameters document.

## 4. Metrics, Directions, and Formula Checks

Sources: Manuscript Table 2 and §§3.6-3.12; Appendix §E.1-E.2 and Tables S16-S17.

| Metric | Published definition/direction | Implementation check |
|---|---|---|
| TSR | Benchmark endpoint success; higher is better; secondary SciCUEval metric | Literal implementation |
| PCS | Verified motif-trace realization; higher is better; primary SciCUEval structural metric | Implements the printed formula without changing its sign |
| CCS | Cross-context structural consistency; higher is better | Literal implementation |
| BSI | Validity-preserving boundary fragility; lower raw value is better; right-censored at `rho_max` when there is no crossing | Literal implementation with computation details retained |
| UPDQ | Weighted Purity, Stability, and Separation; higher is better | Equal weights |
| SC | Structural calibration as `1 - ECE`; higher is better | Literal implementation |
| SCS | DBP15K frozen structural-correspondence reference; higher is better | Uses `exp(-cost)` over CAEGT node, relational, and constraint non-entropy costs; candidate heuristics are used only for auxiliary Hits@1/MRR |
| TP | Transport purity over frozen reference correspondence; higher is better | Literal implementation |
| Hits@1 / MRR | DBP15K ranking diagnostics; not a primary claim | Literal implementation |

Direction audits produce warnings only and never change a primary formula or result. In particular, the PCS node term retains the published positive sign even where its monotonic direction may be in tension with the accompanying prose.

## 5. Perturbation, Boundaries, Open Set, and Calibration

Sources: Appendix §C.2-C.3 and Table S8; §E.2 and Table S17; §E.3.

| Item | Published requirement | Implementation check |
|---|---|---|
| SciCUEval perturbation families | evidence removal, context composition shift, modality conflict, symbolic-depth increase, boundary-condition change | All represented |
| DBP15K perturbation families | neighborhood degradation, relation ambiguity injection, seed-alignment reduction, cross-graph evidence weakening | All represented |
| Validity | Perturbations must preserve task interpretability and target validity; exclude ill-formed cases, complete evidence removal, impossible artifacts, and target leakage | Jointly check all three radii before response generation; deterministically replace invalid trajectories from the same frozen base/reference pool and preserve a replacement audit |
| BSI norm | Scientifically scaled perturbation norm `delta^T W delta` | Materializes intervention at three configured normalized radii; the discrete mapping is a fixed implementation setting documented in the implementation-parameters file |
| BSI event | A valid perturbation changes reasoning regime or diagnostic state | Literal implementation |
| BSI regime output | Report all four regimes separately in the primary table; lower is better | The 2,400 formal boundary instances carry independent `bsi_regime`; each stratum draws only from its corresponding 600-case base evaluation pool, while boundary stress draws from the union of the three pools; the exact 600×4 probe stratification is a fixed implementation setting |
| UPDQ weights | Equal unless otherwise specified | `1/3, 1/3, 1/3` |
| Stability | Mean adjusted Rand index across seed-wise clusterings | Literal implementation |
| Separation | Inter-cluster prototype distance / within-cluster dispersion | Literal implementation |
| Low-support calibration bin | Flag, but do not delete, bins with fewer than `20` samples or less than `1%` of the evaluation set | Match |

## 6. Training, Randomness, and Statistics

Sources: Appendix §D.3; §E.3; Fig. S1 and Table S19.

| Item | Published specification | Implementation key | Check |
|---|---|---|---|
| Fixed random seeds | `42, 0, 1, 123, 1234` | `paper.seeds` | Match |
| Hyperparameter trials | `50` per learned method | `paper.hyperparameter_trials` | Match |
| `lambda_str/lambda_con/epsilon/eta_1:3/gamma/omega_miss/omega_unsup` | Selected on validation and frozen; actual values not reported | Each run's `computed/effective_hyperparameters.json` writes the selected values from `method_selection.json`; `gamma` explicitly maps to selected `minimum_support` rather than a fixture constant | Complete runtime values are auditable |
| Baseline hyperparameters | Validation-only; cannot read the final test | `baseline_reconstruction.*` grids + `computed/baseline_selection.json` | Fixed 50-trial selection; structure validity, open-set novelty, and DBP alignment objectives are recorded separately |
| Baseline/DBP implementation constants | Not fixed by the published specification | Endpoint normalization, attribution tokenizer, taxonomy rules/scalar direction, uncertainty/support normalization, transport/relation/ranking direction, plus `representation_dimensions=256`, `task_success_threshold=.5`, `matched_context_rotations=3`, `local_judge_component_weights=[.2]×5`; DBP signature/local graphs both use `2` hops and ranking weights `.5/.5` | All values enter `baseline_reconstruction.*`/`dbp_reconstruction.*`, are fully written to `implementation_contract.json`, and are required key by key by the static auditor |
| Early-stopping patience | `10` | `paper.patience` | Match |
| Maximum training epochs/equivalent iterations | `100` | `paper.max_epochs` | Match |
| Freezing and final-test isolation | Validation is used only for selection/thresholds; formal test labels do not feed back | Independent `tuning_responses.parquet`/`tuning_metrics.parquet`; the entire formal `600/2400` allocation is final-test only | Match; independent-pool size is a fixed implementation setting |
| Bootstrap resamples | `1000` | `paper.bootstrap_resamples` | Match |
| Confidence interval | `95%`, percentiles `2.5%/97.5%` | `paper.confidence_level` | Match |
| Test | Paired, two-sided | `paper.two_sided_tests=true` | Match |
| Resampling unit | SciCUEval task ID; DBP15K reference-pair ID | Bootstrap API | Match |
| p-value | Twice the smaller empirical tail probability around zero, capped at 1 | Bootstrap implementation | Match |
| Multiple comparison | Holm-Bonferroni over the complete comparison family for the same reported table/figure | `paper.correction` | Match |
| Deterministic baselines | Run once on the same fixed instances; use paired bootstrap when needed | Workflow/statistics | Match |

## 7. Baseline and Ablation Scope

Sources: Manuscript §3.10 and Tables 4-5; Appendix §D.1 and Table S11; §D.5 and Tables S14-S15.

Supported compatible-output families: endpoint, attribution/evidence overlap, fixed taxonomy, uncertainty/calibration, validation-support distance, graph edit, Sinkhorn, Gromov-style transport, structure-aware matching, relation-aware matching, LLM judge, and DBP15K Hits@1/MRR. Each adapter may read only its declared inputs; it cannot read SciPRISM-X residuals, masks, states, boundary/open-set labels, or hidden generator identity. Strongest-baseline derivations for BSI, UPDQ, and SC are stored in separate computed tables. Thresholds/support come only from an independent validation pool excluded from formal evaluation, and final scores use the same final-test perturbations and reference labels.

Ablations cover removal of motif structure, hard constraints, soft penalties, residual channels, reasoning-state field, boundary estimation, open-set categorization, and transport. They share splits, perturbations, seeds, applicable transport budgets, and metric computation.

## 8. Parameter Sources

- Published method values: `configs/paper_explicit.yaml`
- Fixed executable settings not uniquely specified by the publication: `configs/implementation_settings.yaml`
- Combined Track A/B configuration: `configs/track_a.yaml`, `configs/track_b.yaml`
- Fully resolved per-run values and provenance: configuration receipts in the run directory, `computed/method_selection.json`, and stage content hashes

Any third-party default that affects an experiment must appear explicitly in configuration; otherwise runtime configuration validation fails.
