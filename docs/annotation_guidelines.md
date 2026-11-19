# SciPRISM-X Structural-Reference Annotation Guidelines

## 1. Purpose and Scope

These guidelines govern construction and review of the verified structural reference on the SciCUEval task side. The annotation target is observable and verifiable scientific structure, not a model's hidden reasoning, writing style, or topic labels. Annotations are used for task motif graphs, dependency edges, and validator audits. They must not be changed based on final-test results, generator identity, baseline scores, or final SciPRISM-X scores.

## 2. Annotation Inputs

Annotators may see only the following for each task:

- task prompt;
- reference answer or target behavior;
- permitted reference context;
- task-supplied evidence such as tables, images, graph structures, or records;
- frozen motif dictionary, dependency-edge definitions, and validator documentation.

The following information must remain hidden: response-generator names, endpoint-correctness labels, final-test summaries, baseline results, boundary labels, open-set categories, and SciPRISM-X scores.

## 3. Basic Annotation Unit

Annotate text by clause or reasoning step, equations by transformation block, and tables, images, and graph structures by locatable artifact anchor. Merge a short fragment with an adjacent unit by default, but preserve it as an independent unit if it contains a scientific variable, equation, unit, uncertainty cue, causal cue, or modality anchor.

Record the following for each unit:

- `unit_id`;
- original text and character start/end positions;
- motif family;
- variables, equations, units, and evidence anchors;
- confidence;
- dependency edges;
- validator status, reason code, and evidence spans;
- annotator ID and adjudication status.

## 4. Motif-Family Assignment

| Family | Inclusion criterion | Exclusion criterion | Typical validator |
|---|---|---|---|
| Symbolic transformation | An equation transformation, variable substitution, or derived expression is necessary to complete the task | A number or symbol appears without an operational role | algebraic consistency |
| Unit dimensional consistency | Unit conversion or dimensional compatibility is required | Units are only background description and do not affect the reasoning | unit or dimensional parser |
| Causal mechanistic relation | A directed causal, mechanistic, or dependency relation is required | Mere co-occurrence or correlation is described | causal direction or relation checker |
| Evidence binding | A conclusion must be supported by a specific evidence item | Generic citation without locatable evidence | evidence grounding or entailment |
| Modality grounding | A textual conclusion must be anchored to a table, figure, image, record, or graph | An artifact is named but does not support the conclusion | modality-anchor validator |
| Uncertainty qualification | Evidence strength requires confidence, limitations, assumptions, or a caveat | Purely polite or stylistic hedging | uncertainty-discipline checker |

Retain a motif candidate only when all three conditions hold: it is a reusable scientific reasoning operation, it can be anchored to observable evidence, and it can be checked by a deterministic rule or a validation-frozen recognizer. Exclude topics, styles, and unobservable latent reasoning states.

## 5. Dependency Edges

- `prerequisite`: a task-side prerequisite.
- `temporal_succession`: an observable response-side step order.
- `symbolic_dependence`: dependence between expressions, variables, or derived quantities.
- `variable_reuse`: two units reuse the same scientific variable; record as bidirectional.
- `causal_direction`: a directed cause-effect relation.
- `justificatory_support`: one unit provides justification for another.
- `evidence_support`: a support relation between a claim and evidence.
- `local_to_global`: a local observation supports an overall conclusion.
- `cross_modal_grounding`: an anchoring relation between text and a table, image, graph, or record.

Do not add a scientific dependency merely because two units are adjacent. Record adjacency/order edges and scientific-relation edges separately.

## 6. Validator Annotation

Validators follow rule-first order. Use deterministic validators first for explicit equations, units, dimensions, table/figure/graph anchors, and boundary conditions. A frozen entailment fallback may be used only when every deterministic validator returns `UNRESOLVED`.

Every invocation must record one of the following:

- `PASS`: satisfies the validator's explicit rule;
- `FAIL`: violates an explicit rule;
- `UNRESOLVED`: evidence is insufficient or the structure is ambiguous;
- `UNAVAILABLE_EXTERNAL`: a frozen model file is unavailable or fails verification.

A low-confidence or unresolved result must not be labeled as a high-confidence channel violation. An evidence span must cite an actual input location; a post hoc explanation is not a substitute.

## 7. Three-Annotator Review and Adjudication

Three trained annotators independently review the paper-specified stratified audit subset without seeing one another's results. Stratification covers at least domain, competency, motif family, and modality. The review compares motif retention, family, span, edge, validator status, and evidence span.

Any disagreement enters adjudication:

1. Preserve all three original judgments and disagreement fields.
2. The adjudicator decides only from task content, permitted reference context, and the frozen guidelines.
3. Record the final value, rationale, and overridden original values.
4. The adjudicated version becomes the frozen structural standard.
5. Do not reopen adjudication because of final-test trends or method rankings.

## 8. Exceptions and Missing Evidence

- Evidence cannot be located: mark `UNRESOLVED`; do not invent evidence.
- Artifact is missing: record `UNAVAILABLE`; do not fabricate content from general task knowledge.
- Multiple families apply: create separate motif nodes and record evidence for each.
- Boundary condition is ambiguous: preserve the original span and send it to adjudication; do not infer it backward from the endpoint answer.
- Candidate is absent from the dictionary: record it as an audit note, but do not extend the dictionary during the final test.

## 9. Freezing and Versioning

The motif dictionary, parsing rules, recognizer prototypes and thresholds, validator order and thresholds, edge rules, perturbation strengths, regime thresholds, and open-set support must be frozen before the final test. Every protocol change must produce a new version and hash; never overwrite a version already used to compute results.
