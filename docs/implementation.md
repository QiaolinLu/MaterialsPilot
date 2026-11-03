# Implementation and evaluation

This release contains the structure-generation pipeline, its knowledge and retrieval
assets, and 400 project-defined construction tasks. The four task families are
defects, adsorption, grain boundaries and interfaces, with 100 tasks each. Dataset
and per-task hashes are recorded in
[`benchmark_four_tasks_400_manifest.json`](../configs/benchmark_four_tasks_400_manifest.json).
These tasks are not official JARVIS or Open Catalyst Project evaluation splits.

## Generation and retrieval

The default model is `deepseek/deepseek-v3.2` through OpenRouter, with reasoning
disabled, temperature 0.3 and a 4096-token output budget. The agent retrieves
knowledge, generates Python, executes it, and receives deterministic physical and
task feedback. It retains the complete code/feedback history and accumulated
knowledge, stops when no execution or critic issue remains or after five attempts,
and returns the final attempt. There is no score-based rollback, extra planning
model call or automatic coordinate relaxation in this loop.

Knowledge is organized into **L1 foundation**, **L2 structures** and **L3 applications**.
L1 includes the former L0 material. Mat2Vec elemental vectors and CGCNN structural
embeddings use the bundled pretrained assets. Three scientific rankings—composition,
structure and available numerical properties—are combined using reciprocal rank
fusion. Repair rules are a separately identified supplement, not a fourth scientific
ranking. Unsupported property targets are skipped; absent reference structures and
properties are not synthesized into evidence. Asset provenance and licenses are in
[`data/retrieval_assets`](../data/retrieval_assets).

The bundled corpus has limited prototype and property coverage. It does not include
the full set of reference structures or properties needed for every task. Knowledge
entries distinguish task requirements from illustrative dimensions and empirical
estimates. Construction examples preserve cell conventions, periodic images and
local geometry. The cubic CSL helper uses the requested angle's decimal precision
to select a full-precision enumerated angle; it does not change evaluation tolerances
or substitute another Sigma, axis or boundary plane.

## Physical feedback and metrics

The critic combines periodic geometry, chemical consistency, reference-relative
CHGNet energy and task constraints. CHGNet uses energy-only inference. The default
budget is 300 atoms per MLIP prediction; absent references or predictions outside
that budget are explicitly unmeasured. They are not measured physical failures.

Saved structures are evaluated independently with protocol `paper-aligned-1.0.2`:

| Metric | Definition |
|---|---|
| SR | Fraction with confirmed structural validity. |
| TSR | Fraction with confirmed structural validity and every required hard task constraint satisfied. |
| CSR | Mean of per-task flat hard-constraint scores over all completed task outcomes. |
| AvgScore | Aggregate quality score with the structural-validity gate. |

Invalid or failed outputs remain in the denominator. Unknown constraints stay
explicitly unknown; primary scores use conservative contributions and report
measurement coverage and evidence bounds. Those bounds are not confidence intervals.
Missing quality measurements can reduce AvgScore without demonstrating physical
instability. The supplementary **relaxed TSR** requires structural validity, at
least one observed hard constraint and all observed hard constraints passing; it
excludes unknown task constraints and is **not the paper's TSR** or an acceptance rule.

The implementation follows the paper's equations and execution loop with declared
choices for assets, tolerances, coefficients and reference selection where the
paper does not fully specify them. Run settings are in
[`configs/benchmark_400.json`](../configs/benchmark_400.json), with calculations and tolerances implemented in
[`src/evaluation`](../src/evaluation) and [`src/agents/paper_physics.py`](../src/agents/paper_physics.py).
No published performance number is asserted for this packaged revision.

## Runs and provenance

Each run freezes source, knowledge, assets, data, configuration and dependency
versions. Disjoint shards append results, usage and infrastructure-failure journals;
resuming preserves completed attempts, including failures. API and MLIP concurrency
are independent. Transport or memory failures have a bounded retry policy and are
not silently converted into ordinary model failures.

OpenRouter routing is automatic within the configured parameter-support filter.
The default profile excludes GMICloud, requires reasoning to be disabled and rejects
responses reporting positive reasoning-token usage. It imposes a 180-second total
API deadline; actual provider, model, reported tokens and cost are logged. Returned
usage may not account for charges from interrupted requests.

Generated Python executes in a temporary subprocess with a timeout and retains the
local user's permissions. Process separation is not an operating-system security
boundary for untrusted code.

Some task descriptions and structured fields contain ambiguities. The dataset is
preserved as supplied; generation must not silently reinterpret those ambiguities
or relax task constraints to raise scores. Scientific assumptions, the fixed
denominator and missing-evidence treatment should accompany reported results.

Third-party notices accompany the assets and derived knowledge. See
[`NOTICE.txt`](../NOTICE.txt) and the per-entry source metadata. This package does
not assign a new blanket license to inherited code or data.
