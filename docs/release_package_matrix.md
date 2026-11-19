# Deliverables 1-8: Release-Package Code and Artifact Matrix

| ID | Source code or configuration | Formal artifacts | Static audit checks |
|---|---|---|---|
| D1 | `release/package.py`, `scripts/build_release_package.py`, `configs/release_package.yaml` | `release/availability.json`, package `package_manifest.json` | Allowlisted copy; exclude git, venv, cache, runtime, and stage directories; preserve benchmark IDs; scan for secrets, personal identity, and local-machine paths |
| D2 | `motifs/`, `transport/`, `residuals/`, `state_field/`, `perturbations/`, `formal/`, `release/plotting.py` | `computed/*`, `figures/*` | Reference values enter only `comparison`; plots consume only the `computed` namespace |
| D3 | `formal/allocation.py`, `release/manifests.py` | `computed/sample_manifest.parquet` | At formal scale, each benchmark has 600 cases in each of the three non-stress regimes and 2,400 boundary cases; stress samples include parent-sample links |
| D4 | `release/assets/motif_dictionary.yaml`, `release/assets/validator_registry.yaml`, `docs/annotation_guidelines.md` | `release/motif_dictionary.yaml`, `release/validator_registry.yaml`, `release/annotation_guidelines.md` | Six motif families; nine-validator order; three-annotator review, adjudication, blinding, and freezing |
| D5 | `formal/method_selection.py`, `release/manifests.py` | `computed/effective_hyperparameters.json` | lambda, epsilon, eta, gamma, omega, validator thresholds, regime thresholds, normalizers, and search spaces all have concrete values |
| D6 | `formal/proxies.py`, `state_field/model.py`, `state_field/train.py`, `release/manifests.py` | `computed/learned_modules.json`, `computed/state_models.pt`, `computed/state_training_manifest.json` | Complete specifications for g_theta, E_phi, prototype recognizer, SciBERT, BART-MNLI, fallbacks, labels, and objectives |
| D7 | `transport/solver.py`, `reproducibility.py`, `release/manifests.py`, lock files | `computed/solver_environment.json` | Five seeds, float64, tolerances, maximum iterations, stopping rules, 50 trials, patience 10, 100 epochs, preprocessing, software, and anonymized hardware |
| D8 | `release/instance_outputs.py`, `formal/evaluator.py`, `formal/workflow.py`, `formal/statistics.py` | graphs, validator outputs, transport plans, residual components, state assignments, per-instance outputs, bootstrap, and significance | Closed instance-key/foreign-key graph; all six residual channels; explicit baseline and ablation join keys; fixed Holm-Bonferroni procedure |

## Formal-Artifact Foreign Keys

`per_instance_outputs.parquet.instance_key` is the primary key for each SciCUEval response. It links:

- `response_graphs.jsonl.instance_key`;
- `validator_outputs.parquet.instance_key`;
- `transport_diagnostics.parquet.instance_key`;
- `residual_components.parquet.instance_key`;
- `plan_key` in `transport_plans.npz`;
- `baseline_case_id` in `baselines.parquet.case_id`;
- `ablation_case_id` in `ablations.parquet.case_id`.

Task graphs are deduplicated by `task_graph_id`. Public benchmark instance IDs preserve the sampling manifest and are not personal identity information, so they are not irreversibly hashed.
