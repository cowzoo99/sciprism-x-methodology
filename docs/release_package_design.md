# SciPRISM-X Release-Package Design

## Objective and Boundaries

The release package binds the complete method to reviewable source code, configuration, protocol assets, per-instance outputs, and a formal artifact contract. The numerical pipeline computes task motif graphs, response traces, constraint-aware transport, six-channel residuals, the state field, boundary perturbations, open-set analysis, baselines, ablations, and statistics.

The code remains on drive E. The default artifact root for formal runs remains `D:\SciPRISM-X-runs`. The code must not write result values from paper tables into the computation path or generate fabricated per-instance scores.

## Design Decisions

Use contract-completion changes rather than rewriting the computation framework:

1. Convert the motif dictionary, validator order, and threshold interfaces currently embedded in source code into versioned protocol assets.
2. Return numerical results and already-computed structural components from the evaluation process together, avoiding repeated algorithm execution solely to save per-instance evidence.
3. Promote sampling, graphs, validator evidence, transport plans, residual components, state models, effective hyperparameters, environment details, and plotting into formal workflow artifacts.
4. Let the release-package builder collect only explicitly allowlisted files, preserve public benchmark instance IDs, and reject personal identity, secrets, disallowed absolute paths, and cache directories.

## Mapping Between Paper Symbols and Implementation

- `lambda_structure` corresponds to the paper's \(\lambda_{str}\).
- `lambda_constraint` corresponds to \(\lambda_{con}\).
- `epsilon` corresponds to the entropic regularization coefficient \(\varepsilon\).
- `eta_node`, `eta_structure`, and `eta_constraint` correspond to \(\eta_{1:3}\).
- `minimum_support` corresponds to \(\gamma\) in the residual equation: the minimum-support threshold used to identify an omitted required motif.
- `omega_missing` and `omega_unsupported` correspond to \(\omega_{miss}\) and \(\omega_{unsup}\).
- Formal-run values are selected by 50 validation trials and written to `effective_hyperparameters.json`. `minimum_support_threshold` is only an explicit default for fixture entry points and is not presented as the formally selected result.

## Deliverables 1-8 Mapping

### D1: Release Package and Availability

The allowlist-based builder and content auditor use `release_package.yaml`. The package contains source code, configuration, lock files, instance identifiers, the motif dictionary, validator registry, annotation guidelines, per-instance artifacts, plotting code, and file hashes. `availability.json` records review availability, public availability, and actual build time.

### D2: Method and Analysis Implementation

Existing computation modules remain the sole numerical implementation. Add a deterministic plotting module that directly consumes formal computed artifacts and outputs an aggregate-metric figure, an ablation-difference heatmap, and a figure manifest. The plotting layer does not read paper references as computation inputs.

### D3: Sample Identifiers and Sampling Procedure

`sample_manifest.parquet` records dataset, sample ID, original parent sample, allocation ID, split, regime, stratum, perturbation family, strength, language pair, seed, and validity for every row. The paper-scale static contract requires 600 standard cases in each of the three non-stress regimes for each benchmark and 2,400 boundary-stress cases. Stress cases must link to the same base evaluation pool.

### D4: Complete Motif Dictionary, Validators, and Annotation Guidelines

The motif dictionary covers all six Table S1 families and stores every regular-expression cue, alias, confidence source, and edge-construction rule used by the implementation. The validator registry covers eight deterministic validators, the learned entailment fallback, rule-first short-circuit order, input isolation, return statuses, and threshold sources. The annotation guidelines implement three annotators, independent annotation, disagreement adjudication, generator-identity blinding, and freezing rules.

### D5: Actual Hyperparameters and Thresholds

`effective_hyperparameters.json` stores selected method coefficients, loss and residual normalizers, validator thresholds, state-regime thresholds, solver stopping rules, perturbation strengths, open-set settings, and all search spaces. Every value comes from an actual configuration or runtime selection artifact; null values and library defaults are prohibited.

### D6: Complete Specifications for Learned Modules

`learned_modules.json` stores the architecture, inputs, trained or frozen status, label source, objective, initialization, optimizer, checkpoint/revision, and actual route for \(g_\theta\), \(E_\phi\), the SciBERT encoder, the BART-MNLI validator, and deterministic fallbacks. Formal artifacts also include the state model, training manifest, and checkpoint hashes.

### D7: Solver, Versions, and Runtime Settings

`solver_environment.json` stores Python and dependency versions, lock-file hashes, OS, CPU, GPU, memory, five random seeds, PyTorch determinism flags, Sinkhorn tolerance, maximum iterations, consecutive-stability stopping rule, 50 search trials, patience 10, and a maximum of 100 epochs. Hardware information is collected during the formal run without recording personal identity.

### D8: Per-Instance Outputs and Computation Rules

Formal artifacts add task graphs, response graphs, validator outputs with evidence spans, transport plans and diagnostics, six-channel residual components, state assignments, and `per_instance_outputs.parquet`. The latter uses a stable instance key to link primary metrics, baselines, ablations, and all structural evidence. Statistical artifacts continue to retain paired bootstrap, 1,000 resamples, two-sided tests, and Holm-Bonferroni correction.

## Fail-Closed Rules

- Reject a formal package when motif families, validator IDs, or protocol-file versions are incomplete.
- Reject the audit when formal-scale sample counts, parent-sample links, or split isolation violate the contract.
- Reject the audit when a per-instance record lacks a graph, validator, transport, or residual foreign key.
- Reject the audit when formally selected parameters, regime thresholds, model routes, software versions, or file hashes are missing.
- Reject export when the release package contains secrets, personal identity, cache directories, or disallowed local-machine paths.
- Plotting consumes only the `computed` namespace; paper references may enter only the final `comparison` namespace.

## Static Acceptance Criteria

Every item in Deliverables 1-8 maps to a protocol asset, production-code path, formal artifact, and audit rule. No unresolved fields, implicit third-party method defaults, or hard-coded computed result rows are permitted. Runtime verification is performed separately from this static package contract.
