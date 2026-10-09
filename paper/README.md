# Manuscript draft status

`conference_101719.tex` is a working draft about an implemented fixed-mask, 49-weight source-illumination pilot. It does not report the proposed future joint SMO--Neural-ILT system. No PDF has been validated from this revised source.

## Frozen FIT evidence

The primary evidence is the completed October 9 schema-6 run:

- Report: `schema6_a7f5104_report.json`
- Report SHA-256: `1b830b964b9e8ccb84f4e8306e7419a2be221e687222ded57a4171894747289c`
- Status: `no_fit_qualified_checkpoint`
- Five registered seeds (17, 29, 43, 71, 101) each completed all 204 scheduled steps, used a distinct feasible start with no fallback, and recorded four diagnostic snapshots. The minimum pairwise start distance was L1 = 0.37452050414042987.
- Each trajectory recorded three accepted updates total, one at each beta, and 201 stationary steps. All five reached the same final FIT source. Total runtime was 642.929195 seconds.
- There were zero qualified checkpoints. Calibration remained closed, and final3 layouts were never indexed or evaluated.
- Original four FIT reference means: PV-band 234.5, nominal L2 0, worst-dose L2 150.25.
- New three FIT reference means: PV-band 161.333333, nominal L2 35.333333, worst-dose L2 93.333333.
- Last FIT checkpoint on those same new three layouts: PV-band 161.333333, nominal L2 35.666667, worst-dose L2 93.333333. All five runs had this result. PV-band and worst-dose means tied the reference; nominal L2 increased by one aggregate mismatch across the three layouts. The checkpoint failed the strict PV-band-improvement and no-worse-nominal-L2 gates.
- At the final beta-800 phase, the observed original-four mean soft loss decreased by 7.372981e-6 from phase boundary to final checkpoint while the new-three mean increased by 1.343352e-6. These are run-specific differentiable-surrogate changes, not a hard-metric result, an impossibility claim, or evidence about generalization.

The report is FIT-only evidence for the manuscript. Calibration metrics were not used, and final3 was never indexed or evaluated.

## Protocol and scope

The experiment uses seven fixed FIT layouts, source-only optimization, scalar Abbe fixed-basis source intensity, and a dose-weighted sigmoid resist. The 49 supported source points are the radius-0.9 disk on a 9x9 pupil grid, including the center and points below radius 0.3. The 0.3 inner radius defines the initial annular illumination profile; it does not restrict optimization support. The pilot does not demonstrate focus variation, joint mask/source optimization, hotspot classification, or performance on LithoBench or MaskOpt.

The October 7 schema-5 attempt is historical: all five jittered starts failed the feasibility audit and fell back to the same seed-17 reference. Its frozen report SHA-256 was `263c1bb7e0de8bfcfff133e58a0b1c417f78cb565b9b1c9d4c3fa995cb766320`. It is not an independent replicate of the October 9 schema-6 attempt.

Schema 7 implements a prospective static new-three-only soft-objective ablation while retaining the same source physics, feasible-start protocol, original-four hard guards, all-layout hard qualification rules, and common all-seven checkpoint ranking. It has not been benchmarked and has no measured results. Its prospective plan SHA-256 is `94f1078a622a5abdb3646b2696e8a8485e9d128de13226f97a6c20e1eb128e6b`; its schema-6 predecessor report is pinned by the hash above. The schema-6 external plan SHA-256 is `c6dffeca8919dc8b0299ca6d35b05131b7bc67b8b61d45aba86377c7199915bc`.

## Validation and review status

- The local Python unittest suite passed 166 tests with zero skips in 75.660 seconds after the schema-7 implementation and before the manuscript-only edits. The root's independent full harness passed 173/173 tests with zero skips in 75.842 seconds before the focused nonzero-gap schema-7 Armijo regression test was added. After adding that regression, `python -B -m unittest discover -s tests -p test_source_coverage.py` passed all 46 source-coverage tests. The schema-7 prospective plan and frozen schema-6 report passed their validators. No production benchmark was run for schema 7.
- Grok 4.7 and Muse completed read-only reviews of the schema-7 code and found no code blockers. These reviews do not replace the required Claude Opus review.
- The required Claude Opus 5.5 High review remains pending because the exact model is unavailable under the account usage limit until November 3, 2026. No substitute review is represented as completed.
- The built-in LaTeX compile attempt failed with a Windows helper/setup error; this does not establish a TeX source error. No PDF has been validated from this draft.

## Unresolved author and publication details

- The authors report an accepted prior SBCCI paper on Neural-ILT/PV-band heatmap fine-tuning. Its title, author list, proceedings name, year, pages, and DOI/URL need confirmation before adding a citation.
- The third-author email is preserved exactly from the supplied source as `agostini@inf.ufpel.edu.b`. Its `.b` ending looks incomplete and needs author confirmation; it was not guessed or corrected.
- The author names, order, affiliations, and supplied contact strings were retained. The original Downloads source file was left untouched.

The revised draft removes unsupported claims that joint SMO and Neural-ILT were implemented, that LithoBench/MaskOpt were evaluated, that generalization was demonstrated, or that the method improved process-window performance.
