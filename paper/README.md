# Manuscript draft status

`conference_101719.tex` is a working draft about a fixed-mask, 49-weight source-illumination pilot. It distinguishes the measured source-only experiments from proposed future joint SMO--Neural-ILT and hotspot-classification work. The revised PDF has not been validated.

## Completed event-search comparison

The manuscript now centers on the bounded event-versus-grid search described in [SOURCE_EVENT_RESULTS.md](../docs/SOURCE_EVENT_RESULTS.md). Two separately pinned protocols completed 15 pairs each: the median paired grid/event search-wall ratios were 9.582 without cache and 9.598 with isolated physical-float32 caching. Both preserved the same hard quality, with no additional PV-band improvement over the protected incumbent. The four captured JSON evidence files are linked from the results note.

The cache-enabled server suite passed 233 tests. Muse completed supplemental implementation, numerical manuscript, and editorial closure reviews. Required exact Opus review remains quota-blocked; the integrated LaTeX compiler still fails while preparing its Windows helper, so the revised PDF is not validated. These runtime repeats use the same seven FIT layouts and are not independent generalization evidence.

## Frozen FIT evidence

**Schema 6** is recorded in `schema6_a7f5104_report.json` (SHA-256 `1b830b964b9e8ccb84f4e8306e7419a2be221e687222ded57a4171894747289c`). All five seeds completed 204 scheduled steps with four diagnostic snapshots, used distinct feasible starts with no fallback, and produced zero qualified checkpoints. Each had three accepted updates total and 201 stationary iterations. The minimum pairwise start distance was L1 = 0.37452050414042987; runtime was 642.929195 seconds. The last new-three FIT checkpoint tied reference PV-band and worst-dose means at 161.333333 and 93.333333 pixels, while nominal L2 rose from 35.333333 to 35.666667. Calibration remained closed; final3 was never indexed or evaluated.

**Schema 7** is recorded in `schema7_a5c9d3b_report.json` (SHA-256 `ec993a4a2b5fb566816c05cad863c84ac0b525d0fc36d6dc3da3fe2d6798ece5`). All five seeds completed 204 steps and four diagnostic snapshots using distinct feasible starts without fallback; each had one accepted update and 203 stationary iterations. The runs produced zero qualified checkpoints and took 586.786348 seconds total. Their final float32 source hash was common; float64 vectors differed by at most L1 = 2.6945676592782242e-14. New-three reference means were PV-band 161.333333, nominal L2 35.333333, and worst-dose L2 93.333333. Last-checkpoint means were 162, 35, and 94.333333, respectively: one aggregate nominal mismatch improved, while aggregate PV-band count increased by two and worst-dose L2 by three. Calibration remained closed; final3 was never indexed or evaluated.

The schema-6 beta-800 transition-to-final surrogate changes are run-specific: old-four mean soft loss decreased by 7.372981e-6 and new-three mean increased by 1.343352e-6. They are not hard-metric outcomes, do not establish global incompatibility, and do not explain schema-7 reference-to-final metrics, which use a different baseline. Repeated seeds on these fixed layouts are computational restarts, not independent generalization evidence.

**Schema 8** is recorded in `schema8_db12cd8_report.json` (SHA-256 `b3a2ba2d20a9a19584d35d15ebc5e942b9a78bbed85fb4f4fbe58df7a8d543c0`). All five distinct feasible starts completed 204 steps and four critical-softcount plus four smooth-PV diagnostic snapshots without fallback. Each seed had three accepted updates total and 201 stationary iterations. All five reached the same final FIT weight SHA-256, `480bca7f8ceceb0eeb15ea9d8e5b12f620e4eec395261349c3411e10aeea903b`; runtime was 683.377778 seconds. The new-three reference means were PV-band 161.333333, nominal L2 35.333333, and worst-dose L2 93.333333. Every last checkpoint had 161.333333, 35.666667, and 93.333333, respectively. Thus PV-band and worst-dose tied the reference, while nominal L2 increased by one aggregate mismatch. No checkpoint qualified; the strict PV-band improvement and nominal non-regression gates failed. Calibration remained closed and final3 was never indexed or evaluated.

## Protocol and schema-8 result

The pilot uses the seven fixed FIT layouts, scalar Abbe fixed-basis source intensity, and dose-weighted sigmoid resist. Its 49 source points are the radius-0.9 disk on a 9x9 pupil grid, including the center and points inside radius 0.3. The 0.3 inner radius defines the initial annular illumination profile, not a restriction on optimization support. A conditional bound in the draft shows that the fixed-focus physical continuous-resist heatmap with steepness 50 stays below 0.15 for all nonnegative aerial intensities under these settings. This says nothing about binary PV-band maps, critical-softcount diagnostics, schema-8's separate optimizer surrogate, or focus-varying data; no hotspot-classification metric is measured.

Schema 8 tested a static equal-layout mean over the new three FIT layouts of `sigmoid(beta*(1.02*aerial - 0.225)) - sigmoid(beta*(0.98*aerial - 0.225))`, with the existing beta continuation 200/400/800 and no target-fidelity penalty. It retained the source physics, feasible starts, original-four hard guards, all hard FIT gates, and common all-seven critical-softcount beta-800 checkpoint ranking. The external plan at `D:\Codex\Lithography\work\server_benchmark\robust-source-quality-20261005-766872\direct_pv_objective_candidate_plan.json` (SHA-256 `de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127`) is consumed and must not be rerun. The frozen schema-8 report pins the consumed schema-7 FIT report. This was a FIT-only run; calibration remained closed and final3 was never indexed or evaluated.

The hard-metric outcome did not qualify: PV-band and worst-dose means tied the fixed reference, while nominal L2 increased by one aggregate pixel. The measured run does not establish that a qualifying source is infeasible. A separate direct hard-PV mixed-integer diagnostic is only a read-only design proposal; it has not been implemented or run.

The October 7 schema-5 attempt is historical: all five jittered starts failed feasibility and fell back to the seed-17 reference. Its frozen report SHA-256 is `263c1bb7e0de8bfcfff133e58a0b1c417f78cb565b9b1c9d4c3fa995cb766320`. It is not an independent replicate.

## Validation and review status

The subsequent completed FIT-only segment diagnostic is documented in
[SOURCE_SEGMENT_RESULTS.md](../docs/SOURCE_SEGMENT_RESULTS.md). It evaluated
all 5,140 slots in 514.449706 seconds and found 342 passing slots (282 distinct
float32 sources). The best new-three PV-band mean decreased from 161.333333
to 160.666667 pixels, with mean nominal/worst-dose errors unchanged. Finite
ribbons improved nominal error by one while line ends worsened by one; this
is a mean-fidelity result, not non-regression on every layout. Calibration
stayed closed by design and final3 was never indexed or evaluated. The
diagnostic report SHA-256 is
`a66908b8a8b902bdcc1846a35c55ec4828080b9fba2499cbb6ebacdbff08d3b4`.
Its consumed plan SHA-256 is
`5c38f94ff79b7f4e2738664b211d4b0c1c0692f95ecb1ab56548736e50b1170d`.

The diagnostic implementation passed 201 unique tests locally in 88.930
seconds and on the server in 13.533 seconds. Muse completed read-only code
and manuscript reviews. Grok's diagnostic code review reached its 10-minute
timeout without a final review, with partial output preserved. Opus remains
pending under the exact-model quota restriction below. Review status does
not turn this small development-set gain into a quality or generalization claim.

- After schema-8 implementation and the two-test follow-up, `python -B -m unittest discover -s tests -p test_source_coverage.py` passed 51 tests and `python -B -m unittest discover -s tests -p test_source_robustness.py` passed 20 tests; the full local suite passed 174 tests with zero skips in 75.535 seconds. That run included the real smooth-PV gradient/Armijo descent at all three beta values and synthetic schema-7 FIT-lineage validation. The root's subsequent final independent unique harness passed 181/181 tests with zero skips in 76.519 seconds; the server harness passed 181/181 in 13.346 seconds.
- The consumed schema-8 plan hash and closed schema-7 FIT-lineage validator passed. The schema-7 and schema-8 attempts are immutable; the schema-8 report records no qualifying checkpoint and is validated without using calibration metrics.
- Grok 4.7 and Muse completed read-only reviews of the schema-8 implementation, focused test follow-up and documentation corrections, with no code blockers. These completed schema-8 reviews do not imply completion of the later diagnostic review. The required Claude Opus 5.5 High review remains pending because the exact model is unavailable under the account usage limit until November 3, 2026. No substitute review is represented as completed.
- The built-in LaTeX compile attempt failed with a Windows helper/setup error. This does not establish a TeX source error; no PDF has been validated.

## Unresolved author and publication details

- The authors report an accepted prior SBCCI paper on Neural-ILT/PV-band heatmap fine-tuning. Its title, author list, proceedings name, year, pages, and DOI/URL need confirmation before adding a citation.
- The third-author email was corrected from the supplied `.b` typo to `agostini@inf.ufpel.edu.br`, as listed on the [official UFPel page](https://wp.ufpel.edu.br/notcc/propostas/avaliacoes/propostas-2023-2/).
- Author names, order, and affiliations were preserved; the email correction is recorded above. The original Downloads source remains untouched.

The draft does not claim that joint SMO and Neural-ILT were implemented, that LithoBench or MaskOpt were evaluated, that generalization was demonstrated, or that the measured experiments improved process-window performance.
