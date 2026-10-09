# Manuscript draft status

`conference_101719.tex` is a working draft about a fixed-mask, 49-weight source-illumination pilot. It distinguishes the measured source-only experiments from proposed future joint SMO--Neural-ILT and hotspot-classification work. The revised PDF has not been validated.

## Frozen FIT evidence

**Schema 6** is recorded in `schema6_a7f5104_report.json` (SHA-256 `1b830b964b9e8ccb84f4e8306e7419a2be221e687222ded57a4171894747289c`). All five seeds completed 204 scheduled steps with four diagnostic snapshots, used distinct feasible starts with no fallback, and produced zero qualified checkpoints. Each had three accepted updates total and 201 stationary iterations. The minimum pairwise start distance was L1 = 0.37452050414042987; runtime was 642.929195 seconds. The last new-three FIT checkpoint tied reference PV-band and worst-dose means at 161.333333 and 93.333333 pixels, while nominal L2 rose from 35.333333 to 35.666667. Calibration remained closed; final3 was never indexed or evaluated.

**Schema 7** is recorded in `schema7_a5c9d3b_report.json` (SHA-256 `ec993a4a2b5fb566816c05cad863c84ac0b525d0fc36d6dc3da3fe2d6798ece5`). All five seeds completed 204 steps and four diagnostic snapshots using distinct feasible starts without fallback; each had one accepted update and 203 stationary iterations. The runs produced zero qualified checkpoints and took 586.786348 seconds total. Their final float32 source hash was common; float64 vectors differed by at most L1 = 2.6945676592782242e-14. New-three reference means were PV-band 161.333333, nominal L2 35.333333, and worst-dose L2 93.333333. Last-checkpoint means were 162, 35, and 94.333333, respectively: one aggregate nominal mismatch improved, while aggregate PV-band count increased by two and worst-dose L2 by three. Calibration remained closed; final3 was never indexed or evaluated.

The schema-6 beta-800 transition-to-final surrogate changes are run-specific: old-four mean soft loss decreased by 7.372981e-6 and new-three mean increased by 1.343352e-6. They are not hard-metric outcomes, do not establish global incompatibility, and do not explain schema-7 reference-to-final metrics, which use a different baseline. Repeated seeds on these fixed layouts are computational restarts, not independent generalization evidence.

## Protocol and prospective schema 8

The pilot uses the seven fixed FIT layouts, scalar Abbe fixed-basis source intensity, and dose-weighted sigmoid resist. Its 49 source points are the radius-0.9 disk on a 9x9 pupil grid, including the center and points inside radius 0.3. The 0.3 inner radius defines the initial annular illumination profile, not a restriction on optimization support. A conditional bound in the draft shows that the fixed-focus physical continuous-resist heatmap with steepness 50 stays below 0.15 for all nonnegative aerial intensities under these settings. This says nothing about binary PV-band maps, critical-softcount diagnostics, schema-8's separate optimizer surrogate, or focus-varying data; no hotspot-classification metric is measured.

Schema 8 is implemented but has not been benchmarked. It uses a static equal-layout mean over the new three FIT layouts of `sigmoid(beta*(1.02*aerial - 0.225)) - sigmoid(beta*(0.98*aerial - 0.225))`, with the existing beta continuation 200/400/800 and no target-fidelity penalty. It retains the source physics, feasible starts, original-four hard guards, all hard FIT gates, and common all-seven critical-softcount beta-800 checkpoint ranking. The prospective plan is external at `D:\Codex\Lithography\work\server_benchmark\robust-source-quality-20261005-766872\direct_pv_objective_candidate_plan.json`, SHA-256 `de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127`. It pins the consumed schema-7 FIT report above. No schema-8 production run, calibration scoring, or final3 evaluation has occurred.

The October 7 schema-5 attempt is historical: all five jittered starts failed feasibility and fell back to the seed-17 reference. Its frozen report SHA-256 is `263c1bb7e0de8bfcfff133e58a0b1c417f78cb565b9b1c9d4c3fa995cb766320`. It is not an independent replicate.

## Validation and review status

- After the two-test follow-up, `python -B -m unittest discover -s tests -p test_source_coverage.py` passed 51 tests and `python -B -m unittest discover -s tests -p test_source_robustness.py` passed 20 tests. Full `python -B -m unittest discover -s tests` passed 174 tests with zero skips in 75.535 seconds. This includes the real smooth-PV gradient/Armijo descent at all three beta values and synthetic schema-7 FIT-lineage validation. The root's independent unique harness had passed 179/179 tests in 75.648 seconds before this follow-up; that earlier count is not a validation count for the current tree.
- The schema-8 prospective plan hash and schema-7 closed failed FIT lineage validator passed. The schema-7 attempt is immutable and its frozen report is validated without using calibration metrics.
- Grok 4.7 and Muse completed read-only reviews of the schema-8 implementation and found no code blockers. This follow-up adds tests and documentation only and has not received a separate review. The required Claude Opus 5.5 High review remains pending because the exact model is unavailable under the account usage limit until November 3, 2026. No substitute review is represented as completed.
- The built-in LaTeX compile attempt failed with a Windows helper/setup error. This does not establish a TeX source error; no PDF has been validated.

## Unresolved author and publication details

- The authors report an accepted prior SBCCI paper on Neural-ILT/PV-band heatmap fine-tuning. Its title, author list, proceedings name, year, pages, and DOI/URL need confirmation before adding a citation.
- The third-author email is retained exactly from the supplied source as `agostini@inf.ufpel.edu.b`. Its `.b` ending needs author confirmation and was not guessed or corrected.
- Author names, order, affiliations, and supplied contact strings were preserved. The original Downloads source remains untouched.

The draft does not claim that joint SMO and Neural-ILT were implemented, that LithoBench or MaskOpt were evaluated, that generalization was demonstrated, or that the measured experiments improved process-window performance.
