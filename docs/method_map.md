# SciPRISM-X Method-to-Code Map

## 1. Input-Material Fingerprints

| Material | SHA-256 | Purpose |
|---|---|---|
| `Manuscript.docx` | `9611FA08BB2DAAF52A90DC9CC2262DC38EE799E26F1938E5FC5BC95E6E066934` | Methods, equations, Tables 1-5, and experimental/result protocols |
| `Appendix.docx` | `822DDCC81F767FE5148786127C09750D50484540BB322B5CC164D9620943AEC2` | Tables S1-S19, perturbations, baselines, statistics, worked example, and reproducibility details |

These documents define the published method contract and section-level citations. Runtime behavior is controlled by the versioned configuration, protocol assets, and validation-selection artifacts in this repository.

## 2. Official Data Sources and Pinned Versions

| Data | Pinned source | Verification |
|---|---|---|
| SciCUEval | `HICAI-ZJU/SciCUEval@423484f99541beed575b28645661afed86dd4ff9` | Per-file SHA-256 for 40 files; inventory contains `11,343` records |
| DBP15K | `nju-websoft/JAPE@73729a2df0547d79ad07c2758d27c9873930adec` | `dbp15k.tar.gz` SHA-256 `0ba26f55da9b49a67bdb8d9586c6b1882c9b673a7088e3091eb32feac017a2d7` |

Machine-readable sources of truth are `data/manifests/source_registry.json` (40 SciCUEval files plus one DBP15K archive) and `data/manifests/dataset_inventory.json`. Each downloaded item is hash-verified. A missing file is reported as `ABSENT`; substitute data must not be generated.

## 3. Method Mapping

| Code area | Published source location | Primary responsibility | Verification tests |
|---|---|---|---|
| `src/sciprism_x/config.py`, `provenance.py`, `reproducibility.py`, `tuning.py` | §3.13; Appendix D.3, E.3, Table S19 | Strict configuration, parameter provenance, seeds, validation-only tuning | `tests/test_config.py`, `tests/test_reproducibility.py`, `tests/test_tuning.py` |
| `src/sciprism_x/data/` | §3.1; Appendix C.1, Tables S6-S7, D.4/S13 | Official acquisition, hashes, inventory, fixed splits | `tests/data/` |
| `src/sciprism_x/responses/` | §3.2/Table 1; Appendix E.5 | Ordinary response schema, supplied-response routing, profile fitting, deterministic construction | `tests/responses/` |
| `src/sciprism_x/motifs/` | §§3.3-3.4; Appendix A | Scientific units, six motif families, typed edges, salience | `tests/motifs/` |
| `src/sciprism_x/validators/` | §3.4; Appendix B | Physical/logical/unit/causal/evidence/uncertainty validators with rule-first order | `tests/validators/` |
| `src/sciprism_x/transport/` | §3.5 printed transport equations; Appendix B | Node/structure/constraint costs, hard mask, stabilized fused transport | `tests/transport/` |
| `src/sciprism_x/residuals/` | §3.6 printed residual/PCS formulas; Appendix E.1-E.2 | Six-channel residuals, TSR/PCS/consistency, direction audit | `tests/residuals/` |
| `src/sciprism_x/state_field/` | §§3.7-3.8; Appendix B/E | 15-feature state, energy, stable/brittle/invalid | `tests/state_field/` |
| `src/sciprism_x/perturbations/` | §3.9; Appendix C.2-C.3, Table S8 | SciCUEval/DBP15K validity-preserving perturbations and boundaries | `tests/perturbations/` |
| `src/sciprism_x/open_set/` | §§3.8-3.9; Appendix E.2, Table S17 | Unknownness, K-means grouping, UPDQ, SC | `tests/open_set/` |
| `src/sciprism_x/baselines/` | §3.10, Tables 4-5; Appendix D.1-D.2, Tables S11-S12 | Compatible baseline adapters, LLM judge, input isolation | `tests/baselines/` |
| `src/sciprism_x/metrics/` | §3.11, Table 2; Appendix E.1-E.2, Tables S16-S17 | TSR/PCS/CCS/BSI/UPDQ/SC/SCS/TP/Hits@1/MRR | `tests/metrics/` |
| `src/sciprism_x/statistics/` | §3.12; Appendix E.3, Fig. S1 | Paired bootstrap, 95% interval, two-sided p, Holm | `tests/statistics/` |
| `src/sciprism_x/results/` | §3.13; Appendix Table S19 | Atomic writes, schemas, hashes, receipts | `tests/results/` |
| `src/sciprism_x/formal/` | §§3.1-3.13; Tables 1-5; Appendix C-E, Tables S6-S19 | Formal-scale configuration/allocation, public-data parsing, three response replicates, validation-only method selection and normalization, per-case evaluation, five-seed state field, perturbation boundaries, open set, full-candidate DBP15K ranking, compatible baselines, ablations, statistics, and 17-stage orchestration | `tests/formal/` |
| `src/sciprism_x/release/`, `configs/release_package.yaml`, `configs/plotting.yaml` | Main-text Code availability; Appendix E.6/Table S19; Deliverables 1-8 | Complete motif/validator protocols, per-instance components, effective parameters, learned modules, environment, deterministic plotting, allowlisted release package, and static auditor | `tests/release/` |
| `src/sciprism_x/workflow.py` | Complete §3.1-3.13 pipeline | Track A/B isolation, end-to-end orchestration, artifact manifest | `tests/test_workflow_isolation.py`, `tests/integration/` |
| `notebooks/SciPRISM_X_Methodology.ipynb` | Cross-entry point for the complete method | Top-to-bottom execution, parameter display, result loading, plotting entry point | `tests/notebook/test_notebook.py` |

## 4. Complete Core-Module Inventory

This list prevents an implemented module from being omitted from the cross-map. Every path is constrained by the corresponding method location above.

```text
src/sciprism_x/__init__.py
src/sciprism_x/config.py
src/sciprism_x/provenance.py
src/sciprism_x/reproducibility.py
src/sciprism_x/tuning.py
src/sciprism_x/workflow.py
src/sciprism_x/data/__init__.py
src/sciprism_x/data/acquisition.py
src/sciprism_x/data/schema.py
src/sciprism_x/data/splits.py
src/sciprism_x/responses/__init__.py
src/sciprism_x/responses/adapters.py
src/sciprism_x/responses/fit.py
src/sciprism_x/responses/reconstruct.py
src/sciprism_x/responses/schema.py
src/sciprism_x/motifs/__init__.py
src/sciprism_x/motifs/extract.py
src/sciprism_x/motifs/salience.py
src/sciprism_x/motifs/schema.py
src/sciprism_x/motifs/segment.py
src/sciprism_x/validators/__init__.py
src/sciprism_x/validators/deterministic.py
src/sciprism_x/validators/learned.py
src/sciprism_x/validators/registry.py
src/sciprism_x/validators/schema.py
src/sciprism_x/transport/__init__.py
src/sciprism_x/transport/costs.py
src/sciprism_x/transport/schema.py
src/sciprism_x/transport/solver.py
src/sciprism_x/residuals/__init__.py
src/sciprism_x/residuals/compute.py
src/sciprism_x/residuals/schema.py
src/sciprism_x/residuals/scores.py
src/sciprism_x/state_field/__init__.py
src/sciprism_x/state_field/features.py
src/sciprism_x/state_field/model.py
src/sciprism_x/state_field/train.py
src/sciprism_x/perturbations/__init__.py
src/sciprism_x/perturbations/boundary.py
src/sciprism_x/perturbations/dbp15k.py
src/sciprism_x/perturbations/scicu.py
src/sciprism_x/open_set/__init__.py
src/sciprism_x/open_set/calibration.py
src/sciprism_x/open_set/grouping.py
src/sciprism_x/baselines/__init__.py
src/sciprism_x/baselines/graph.py
src/sciprism_x/baselines/llm_judge.py
src/sciprism_x/baselines/registry.py
src/sciprism_x/baselines/schema.py
src/sciprism_x/baselines/scientific.py
src/sciprism_x/metrics/__init__.py
src/sciprism_x/metrics/scientific.py
src/sciprism_x/metrics/structural.py
src/sciprism_x/statistics/__init__.py
src/sciprism_x/statistics/bootstrap.py
src/sciprism_x/statistics/multiple_testing.py
src/sciprism_x/results/__init__.py
src/sciprism_x/results/schema.py
src/sciprism_x/results/writer.py
src/sciprism_x/formal/__init__.py
src/sciprism_x/formal/ablations.py
src/sciprism_x/formal/allocation.py
src/sciprism_x/formal/baselines.py
src/sciprism_x/formal/boundaries.py
src/sciprism_x/formal/config.py
src/sciprism_x/formal/datasets.py
src/sciprism_x/formal/dbp.py
src/sciprism_x/formal/evaluator.py
src/sciprism_x/formal/manifest.py
src/sciprism_x/formal/method_selection.py
src/sciprism_x/formal/open_set.py
src/sciprism_x/formal/perturbed_cases.py
src/sciprism_x/formal/proxies.py
src/sciprism_x/formal/responses.py
src/sciprism_x/formal/results.py
src/sciprism_x/formal/schema.py
src/sciprism_x/formal/stages.py
src/sciprism_x/formal/state_training.py
src/sciprism_x/formal/statistics.py
src/sciprism_x/formal/workflow.py
src/sciprism_x/release/__init__.py
src/sciprism_x/release/protocol.py
src/sciprism_x/release/instance_outputs.py
src/sciprism_x/release/manifests.py
src/sciprism_x/release/plotting.py
src/sciprism_x/release/package.py
src/sciprism_x/release/assets/motif_dictionary.yaml
src/sciprism_x/release/assets/validator_registry.yaml
scripts/inspect_formal_run.py
scripts/inspect_release_contract.py
scripts/build_release_package.py
```

## 5. Key Function-Level Evidence Chain

| Function/interface | Input evidence | Output evidence |
|---|---|---|
| `acquire_declared_sources`, `verify_source` | Official commit URL + SHA registry | Verified/absent acquisition records |
| `make_fixed_splits` | Appendix C/D split and leakage policy | Stable task/pair-keyed splits |
| `resolve_generator_profiles`, `resolve_track_a_profiles`, `acquire_formal_responses` | Table 1 generator roles + Track B published aggregate targets/E.5 worked case, or Track A evaluation/tuning supplied files for six generators | Auditable profile/acquisition trace + ordinary response rows; Track A neutral boundary-original counterfactuals are disclosed separately from the primary response matrix |
| `segment_scientific_units`, `extract_motifs`, `compute_salience` | §3.3-3.4, Appendix A | Span-preserving motif graphs |
| Validator registry | §3.4, Appendix B | PASS/FAIL/UNRESOLVED/UNAVAILABLE with rule-first provenance |
| `build_node_cost`, `solve_fused_transport` | §3.5 equations | Hard-masked transport plan + diagnostics |
| `compute_residuals`, score functions | §3.6 and E.2 formulas | Channel components + literal scores + direction warnings |
| `select_method_parameters` | §§3.5-3.6 validation-selected/frozen contract | 50-trial validation audit independent of the formal 600/2400 allocation, frozen transport/residual/PCS coefficients, three transport-loss validation scales, and six residual-channel normalizers |
| State feature/model/train interfaces | §§3.7-3.8 | 15-feature vector + regime/energy outputs |
| Perturbation generators, `find_boundary` | Appendix Table S8/S17 | Audited interventions + crossing/censoring + independent BSI strata for four regimes |
| `cluster_unknowns`, `compute_updq`, `compute_sc` | Appendix E.2/S17/E.3 | Validation-frozen centers/support, five-seed failure groups by regime, and quality/calibration diagnostics |
| Baseline registry and typed projections | Tables 4-5/S11/S12 | Comparator-only allowed views |
| `select_formal_baseline_parameters`, `build_formal_baseline_registry`, `run_formal_baselines` | §§3.10-3.11, Tables 4-5; Appendix D.1-D.3, Tables S11-S12 | Configured 50-trial validation selection + hash-frozen six-field GPT-4o judge replay or deterministic local route + every dataset-compatible numeric baseline row + isolated BSI/UPDQ/SC strongest-baseline derivations |
| `resolve_ablation_policy`, `run_formal_ablations`, `typed_greedy_minimum_cost_matching` | §3.10; Appendix D.5, Tables S14-S15 | Eight single-module controls, paired per-case recomputation, and typed greedy replacement without transport |
| `compute_formal_statistics`, `build_discrepancy_report` | §3.12; Appendix E.3, Fig. S1 | 1,000 paired bootstrap resamples over current per-case results, full-family Holm correction, and a reference-difference report read only after completion |
| `StageStore.run`, `assemble_formal_results` | §3.13; Appendix Table S19 | Input/implementation-hash-compatible resume, per-file verification, atomic finalization, and physical separation of computed/reference/comparison data |
| `run_formal_experiment`, `FORMAL_STAGE_ORDER` | §§3.1-3.13; Appendix C-E | Unified 17-stage CLI/Notebook entry from source verification through release contract and deterministic figures; separate Track A response manifests and Track B pooled-profile inputs; reference values enter comparison only after statistics; independent train/validation pools are physically separate from the formal final-test allocation |
| `serialize_evaluated_components`, manifest builders, release auditors | Main-text Code availability; Appendix E.6/Table S19; Deliverables 1-8 | Closed per-instance graph/validator/plan/residual/state foreign keys, effective hyperparameters, learned modules, environment, protocols, plot hashes, and release-package integrity |
| `build_paper_manifest`, `inspect_formal_run` | Tables 1-5; Appendix C-E and S19 | Exact manifests for 75,600 SciCUEval responses and 4,200 DBP cases, plus final hash, finiteness, provenance, baseline, ablation, and statistical-integrity audits |
| `paired_bootstrap`, `holm_bonferroni` | §3.12, Appendix E.3 | Paired intervals, raw/adjusted p, family membership |
| `AtomicRunWriter`, `run_track_a`, `run_track_b` | §3.13/Table S19 | Hashed immutable run directory and `RunReceipt` |

## 6. External-Input Interfaces

- Track A accepts versioned evaluation and tuning response manifests for all six generators, with file hashes included in stage receipts.
- The learned recognizer and entailment validator accept pinned frozen checkpoints; deterministic proxy routes remain fully specified when those assets are not selected.
- `--judge-response-file` accepts frozen six-field GPT-4o judgments keyed by baseline `case_id`; otherwise the deterministic local route is recorded.
- DBP15K comparison-system outputs can enter through the `dbp15k_prior_system_outputs` contract without changing the downstream metric schema.
- `data/manifests/published_targets.json` is Track B fitting evidence. Track A computation cannot read it as a profile or response input; reference values enter only through the post-statistics comparison interface.
