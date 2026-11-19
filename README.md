# SciPRISM-X Methodology and Reference Implementation

SciPRISM-X is a structural evaluation methodology for scientific reasoning systems. It compares a task motif graph with a response-trace graph through constraint-aware entropic transport, converts the alignment into six interpretable residual channels, and maps the resulting evidence to stable, brittle, or invalid response states. Boundary perturbations and open-set analysis complement the standard evaluation by measuring how response structure changes near regime boundaries.

This repository provides the complete Python/Jupyter implementation, fixed configuration, validator protocols, baseline adapters, statistical analysis, plotting layer, and release-package contract.

## Method Pipeline

1. Parse each task into a typed motif graph with entities, relations, quantities, constraints, equations, assumptions, and solution dependencies.
2. Convert each generated response into an ordered response-trace graph.
3. Build node, structural, and constraint costs, apply hard feasibility masks, and solve the entropically regularized transport problem.
4. Compute six residual channels: physical, logical, unit/dimensional, causal, modal/evidential, and uncertainty residuals.
5. Assemble the 15-feature state vector and classify each response as stable, brittle, or invalid.
6. Evaluate boundary sensitivity, open-set calibration, scientific baselines, graph baselines, LLM-judge baselines, and eight ablations.
7. Aggregate results with paired bootstrap intervals, two-sided tests, and Holm-Bonferroni correction.

The formal SciCUEval protocol contains 600 standard instances in each of three non-stress regimes and 2,400 boundary-perturbation instances. Six generators produce three responses per task, yielding 75,600 formal response records. The DBP15K auxiliary protocol uses the full candidate pool with the configured 30% seed alignment.

## Repository Layout

```text
configs/                    Fixed method, execution, plotting, and release settings
data/manifests/             Dataset inventory, source registry, and published targets
docs/                       Method specification, implementation settings, and audit maps
notebooks/                  Jupyter entry point for the complete workflow
scripts/                    Formal-run and release-contract inspection tools
src/sciprism_x/             Method implementation
tests/                      Unit, integration, formal-contract, and notebook checks
```

The primary implementation modules are:

- `motifs/`: task segmentation, motif extraction, salience, and graph schemas;
- `responses/`: response acquisition, trace construction, fitting, and provenance;
- `validators/`: deterministic validators and learned recognizers;
- `transport/`: cost construction, feasibility masks, and Sinkhorn solver;
- `residuals/`: six-channel residual computation;
- `state_field/`: 15-feature construction and three-state classification;
- `perturbations/` and `open_set/`: boundary and open-set analysis;
- `baselines/`, `metrics/`, and `statistics/`: comparisons and inference;
- `formal/`: the 17-stage experiment workflow;
- `release/`: protocol assets, per-instance outputs, manifests, plotting, and package auditing.

## Environment

- Windows / PowerShell
- CPython `3.11.9`: `D:\\py\\Python3\\python.exe`
- Direct and transitive dependencies locked by `uv.lock` and `requirements.lock.txt`
- Runtime outputs stored under `D:\\SciPRISM-X-runs` by the commands below

Create the isolated environment:

```powershell
D:\\py\\Python3\\python.exe -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install uv==0.13.8
.\\.venv\\Scripts\\uv.exe sync --locked --all-groups --no-editable --link-mode copy --python D:\\py\\Python3\\python.exe
.\\.venv\\Scripts\\python.exe -m ipykernel install --prefix .venv --name sciprism-x --display-name "SciPRISM-X (Python 3.11.9)"
```

Validate the dependency lock:

```powershell
.\\.venv\\Scripts\\uv.exe lock --check
```

## Formal Workflow

The default `paper` scale executes the complete 600/2,400 allocation, six generators, three responses per task, five random seeds, the full DBP15K candidate pool, all baselines, eight ablations, 1,000 bootstrap resamples, and Holm correction:

```powershell
$env:PYTHONPATH = 'src'
.\\.venv\\Scripts\\python.exe -m sciprism_x.formal.workflow --config configs\\track_b.yaml --formal-config configs\\formal_execution.yaml --scale paper --output D:\\SciPRISM-X-runs\\formal-paper --resume
```

The `smoke` scale follows the same method stages with reduced case counts for end-to-end acceptance:

```powershell
$env:PYTHONPATH = 'src'
.\\.venv\\Scripts\\python.exe -m sciprism_x.formal.workflow --config configs\\track_b.yaml --formal-config configs\\formal_execution.yaml --scale smoke --output D:\\SciPRISM-X-runs\\formal-smoke --resume
```

Each of the 17 stages records a content-hash receipt. `method_selection` selects and freezes method coefficients, the three transport-loss scales, and the six residual-channel scales on an independent validation pool. State-field training uses the independent train/validation allocation, while the 600/2,400 formal allocation remains the final evaluation set. `--resume` restores a stage only when its input, upstream-content, and implementation hashes match.

The final artifact contract contains 32 `computed/`, 7 `release/`, 3 `figures/`, 1 `paper_reference/`, and 1 `comparison/` artifact. Track A accepts independent evaluation and tuning response manifests for all six generators. Track B accepts pooled response profiles, optional frozen GPT-4o judgments, and the configured response-reconstruction route. Field-level provenance is written into the generated contracts.

Inspect the latest completed formal run:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\inspect_formal_run.py --root D:\\SciPRISM-X-runs\\formal-paper --latest-complete
```

## Notebook

Run the method from the project root:

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'
$env:SCIPRISM_SCALE = 'paper'
$env:SCIPRISM_OUTPUT = 'D:\\SciPRISM-X-runs\\formal-paper'
.\\.venv\\Scripts\\jupyter-execute.exe notebooks\\SciPRISM_X_Methodology.ipynb --inplace --timeout=-1 --kernel_name=sciprism-x
```

The Notebook uses the same formal workflow as the CLI and exposes the effective parameters, stage receipts, computed tables, comparison tables, figure manifest, and SHA-256 manifest. Set `SCIPRISM_SCALE=smoke` for the reduced acceptance scale.

## Release Package

After a formal run completes, audit the per-instance foreign keys and required artifacts, then build an allowlisted release package. The builder refuses to overwrite an existing target directory.

```powershell
$env:PYTHONPATH = 'src'
$run = Get-ChildItem -LiteralPath 'D:\\SciPRISM-X-runs\\formal-paper' -Directory | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
.\\.venv\\Scripts\\python.exe scripts\\inspect_release_contract.py $run.FullName --scale paper
.\\.venv\\Scripts\\python.exe scripts\\build_release_package.py --run $run.FullName --target 'D:\\SciPRISM-X-runs\\sciprism-x-release-v1'
```

## Verification

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
```

The verification suite covers fixed seeds, source hashes, split isolation, rule-first validation, transport feasibility and convergence, formula checks, hand-calculated metrics, paired bootstrap, Holm correction, baseline-input isolation, Track A/B separation, formal artifact contracts, Notebook structure, and deterministic artifact hashes.

## Method Documentation

- [Method specification](docs/method_specification.md)
- [Implementation parameters](docs/implementation_parameters.md)
- [Method-to-code map](docs/method_map.md)
- [Release-package design](docs/release_package_design.md)
- [Release-package matrix](docs/release_package_matrix.md)
- [Release-package implementation](docs/release_package_implementation.md)
- [Annotation and adjudication guidelines](docs/annotation_guidelines.md)
- [Implementation audit](docs/implementation_audit.md)

The configuration sources of truth are `configs/paper_explicit.yaml`, `configs/implementation_settings.yaml`, and `configs/formal_execution.yaml`. The effective configuration selected on validation data is emitted as `computed/method_selection.json`.
