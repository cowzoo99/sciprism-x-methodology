# SciPRISM-X Release-Package Implementation

The `sciprism_x.release` package provides protocol loading, per-instance serialization, effective-configuration manifests, deterministic plotting, artifact auditing, and allowlisted export. It is called by the formal workflow only where evaluation results are materialized and where the completed run is closed.

## 1. Protocol Assets

| Path | Responsibility |
|---|---|
| `src/sciprism_x/release/protocol.py` | Strict protocol models, package-resource loading, and asset hashes |
| `src/sciprism_x/release/assets/motif_dictionary.yaml` | Six motif families, aliases, regular-expression cues, prototype text, and edge rules |
| `src/sciprism_x/release/assets/validator_registry.yaml` | Nine validators, rule-first order, thresholds, permitted inputs, and frozen model revisions |
| `tests/release/test_protocol.py` | Family/validator set, order, regex, threshold, and hash contracts |

## 2. Per-Instance Components

`src/sciprism_x/release/instance_outputs.py` serializes task and response graphs, validator evidence spans, transport-plan indices, solver diagnostics, six residual channels, state assignments, and primary metrics under a stable `instance_key`. `src/sciprism_x/formal/evaluator.py` returns the metric row and structural evidence from a single evaluation call, preventing duplicate calculations from diverging.

The formal workflow emits:

- `task_graphs.jsonl`;
- `response_graphs.jsonl`;
- `validator_outputs.parquet`;
- `transport_plans.npz`;
- `transport_diagnostics.parquet`;
- `residual_components.parquet`;
- `per_instance_outputs.parquet`.

## 3. Effective Configuration and Environment

`src/sciprism_x/release/manifests.py` generates:

- `effective_hyperparameters.json` for lambda, epsilon, eta, gamma, omega, normalizers, validator/state thresholds, solver settings, perturbations, open-set settings, statistics, and search spaces;
- `learned_modules.json` for architectures, inputs, label sources, objectives, initializers, optimizers, revisions, actual routes, and checkpoint metadata;
- `sample_manifest.parquet` for every allocation and perturbation row;
- `solver_environment.json` for dependency-lock hashes, software, anonymized hardware, seeds, and determinism flags;
- `protocol_manifest.json` for versioned release assets.

## 4. Plotting

`src/sciprism_x/release/plotting.py` reads only computed artifacts and the explicit style in `configs/plotting.yaml`. It produces the aggregate-metric SVG, ablation-difference heatmap, and figure manifest. Every figure manifest records input, style, and output hashes.

## 5. Package Builder and Auditor

`scripts/inspect_release_contract.py` checks the Deliverables 1-8 artifact contract and per-instance foreign keys. `scripts/build_release_package.py` then exports only the paths allowed by `configs/release_package.yaml`.

The builder:

- preserves public benchmark IDs;
- excludes Git metadata, virtual environments, caches, runtime staging, and temporary directories;
- scans text for secret patterns, personal identity, and disallowed local paths;
- rejects symlinks, an existing target, and targets inside the project or run tree;
- recomputes file size and SHA-256 for the final package manifest.

## 6. Documentation Closure

- `docs/annotation_guidelines.md` defines annotation units, independent review, adjudication, blinding, threshold freezing, and exception handling.
- `docs/release_package_matrix.md` maps Deliverables 1-8 to source code, configuration, artifacts, and audit rules.
- `docs/method_map.md` maps published method sections to production modules and verification files.
- `docs/implementation_audit.md` records the static 17-stage/44-artifact contract.

## 7. Verification Surface

The release-package checks are located in `tests/release/`. They cover protocol loading, per-instance serialization, manifest completeness, deterministic SVG generation, allowlist behavior, package hashes, path sanitization, and formal-run audit failures. Runtime verification is invoked through the repository's standard pytest command.
