# SciPRISM-X Implementation Contract Audit

## 1. Audit Scope

This audit covers static consistency across the published method specification, configuration, source code, Notebook, formal artifact contract, and release-package contract. It evaluates names, paths, schemas, parameter values, stage dependencies, and artifact relationships. Runtime tests, training, Notebook execution, experiment execution, plotting, and numerical-result generation are separate verification activities.

## 2. Method Chain

The formal entry point is `sciprism_x.formal.workflow`, with 17 content-hashed stages:

1. `sources`;
2. `allocations`;
3. `perturbation_inputs`;
4. `responses`;
5. `method_selection`;
6. `evaluation`;
7. `state_training`;
8. `boundaries`;
9. `open_set`;
10. `dbp15k`;
11. `baselines`;
12. `ablations`;
13. `statistics`;
14. `paper_comparison`;
15. `plotting_inputs`;
16. `release_contract`;
17. `figures`.

Reference values enter only during `paper_comparison`, after per-instance evaluation and statistics are complete. The formal 600/2,400 allocation is the final evaluation set. Method coefficients, normalizers, state thresholds, open-set support, and baseline parameters are selected only from independent train/validation artifacts.

## 3. Deliverables 1-8

| ID | Implementation contract |
|---|---|
| D1 | Allowlisted release-package builder; preserved benchmark IDs; secret, local-path, identity-token, symlink, and per-file SHA-256 checks |
| D2 | Motif, trace, validator, transport, six-channel residual, perturbation, calibration, statistics, and deterministic SVG plotting modules |
| D3 | `sample_manifest.parquet` with dataset, sample, parent sample, allocation, split, regime, stratum, perturbation, language, validity, and seed fields |
| D4 | Six-family motif dictionary, nine-validator registry, prohibited-input rules, and three-annotator annotation/adjudication protocol |
| D5 | `effective_hyperparameters.json` with lambda, epsilon, eta, gamma, omega, normalizers, validator/state thresholds, solver/search settings, perturbations, open-set settings, and statistics |
| D6 | `learned_modules.json` with architectures, inputs, labels, objectives, optimizers, frozen revisions, fallbacks, and checkpoint-selection records |
| D7 | `solver_environment.json` with lock hashes, software versions, anonymized hardware, five seeds, determinism flags, tolerances, iteration limits, and stopping rules |
| D8 | Per-instance graphs, validator evidence, transport plans/diagnostics, six residual channels, state assignments, baseline/ablation links, bootstrap rows, and Holm results |

## 4. Artifact Contract

A completed formal run contains 44 artifacts:

- `computed/`: 32;
- `release/`: 7;
- `figures/`: 3;
- `paper_reference/`: 1;
- `comparison/`: 1.

`per_instance_outputs.parquet.instance_key` links the response graph, validator rows, transport diagnostics, transport plan, six residual channels, and state assignment. It also retains pointers to the baseline, ablation, and component artifacts. The formal auditor checks sample counts, parent-sample relationships, per-instance uniqueness, six-channel coverage, finite values, plan keys, state foreign keys, effective-parameter keys, and learned-module keys.

## 5. Fixed Settings for Open Components

- Split manifest: deterministic task-ID SHA-256 buckets after excluding the complete formal allocation.
- Response routes: Track A uses supplied, hash-recorded files; Track B uses the configured fitted profile and ordinary response schema.
- Baseline selection: 50 trials on an independent validation set, with method-specific objectives.
- Learned motif recognizer: SciBERT/prototype cosine only when no explicit cue exists, threshold `.75`, family margin `.05`, and training-only TF-IDF fallback.
- Learned entailment: BART-MNLI PASS/FAIL thresholds `.80/.80`, with the frozen five-feature fallback.
- State modules: `g_theta=15-64-16`, `E_phi=31-32-1`, exact GELU, float64, AdamW, and validation early stopping.
- Regime thresholds, normalizers, and method coefficients: selected from validation and written to the run's method-selection artifacts.
- Plot appearance: deterministic SVG style from `configs/plotting.yaml`.

`gamma` is bound to the selected `minimum_support`. `residuals.minimum_support_threshold=.50` applies only to fixture entry points that have no validation corpus.

## 6. Contract Invariants

- Motif regexes and aliases load from the frozen dictionary.
- Runtime validator IDs, order, model revision, thresholds, and prohibited inputs match the frozen registry.
- Each response is evaluated once; the same evaluation object produces metrics and per-instance evidence.
- Release-package contract paths are relative to the package root.
- The 44-artifact audit must succeed before package construction.
- Package manifests contain recomputed size and SHA-256 values and reject symlinks or an existing destination.
- Emitted artifacts omit supplied-response and model-root absolute paths.
- Plotting rejects a missing ablation cell instead of filling it with zero.
- README, Notebook, method specification, method map, and implementation settings use the 17-stage/44-artifact contract.

## 7. Static Acceptance Rules

- Workflow and required-artifact specifications contain the same 44 unique paths.
- `FORMAL_STAGE_ORDER` contains the documented 17 stage names in order.
- Production code and configuration contain no unfinished placeholders or hard-coded computed result rows.
- The release allowlist rejects secret patterns, identity paths, and disallowed local paths.
- Runtime defaults point to `D:\SciPRISM-X-runs`; no project runtime default points to drive C.
- Notebook cells have no stored output and no execution count in the release tree.

## 8. Versioned Extension Points

Supplied response manifests, frozen recognizer/validator checkpoints, external judge outputs, DBP15K comparison-system rows, allocation manifests, and plotting styles all enter through versioned interfaces. Replacing one of these inputs changes its content hash and invalidates every dependent stage receipt while leaving the per-instance schema and statistical protocol unchanged.
