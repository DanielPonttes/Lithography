# Manuscript draft status

conference_101719.tex is a working draft that documents the implemented fixed-mask, 49-weight source-illumination pilot. It is not a report of the proposed future joint SMO--Neural-ILT system.

## Scope and evidence

The draft is based on the frozen October 7 coverage report:

- Report SHA-256: 263c1bb7e0de8bfcfff133e58a0b1c417f78cb565b9b1c9d4c3fa995cb766320
- Status: no_fit_qualified_checkpoint
- Seeds 17, 29, 43, 71, and 101 each completed 204 iterations. Every jittered start failed the feasibility audit and fell back to the same seed-17 reference. Each trajectory recorded three accepted updates (one at each beta) and 201 stationary steps.
- No checkpoint qualified; calibration stayed closed; final3 layouts were never indexed or evaluated.
- Original four FIT reference means: PV-band 234.5, nominal L2 0, worst-dose L2 150.25.
- New three FIT reference means: PV-band 161.333333, nominal L2 35.333333, worst-dose L2 93.333333.
- Last FIT checkpoint on those same new three layouts: PV-band 161.333333, nominal L2 35.666667, worst-dose L2 93.333333. All five seeds had this same result. PV-band and worst-dose means tied the reference, while nominal L2 increased by one aggregate mismatch across the three layouts. The checkpoint failed the strict PV-band-improvement and no-worse-nominal-L2 gates.

The experiment uses seven fixed FIT layouts, source-only optimization, scalar Abbe fixed-basis source intensity, and a dose-weighted sigmoid resist. The 49 supported source points are the radius-0.9 disk on a 9x9 pupil grid, including the center and points below radius 0.3. The 0.3 inner radius defines the initial annular illumination profile, not a restriction on optimization support. The pilot does not demonstrate focus variation, joint mask/source optimization, hotspot classification, or performance on LithoBench or MaskOpt.

## Planned work and unresolved items

- The schema-6 distinct feasible LMO-start protocol is implemented as a Luna prototype and pinned in a prospective external plan (SHA-256 `c6dffeca8919dc8b0299ca6d35b05131b7bc67b8b61d45aba86377c7199915bc`). It awaits the required remaining review and a controlled benchmark run; it has not been benchmarked and the draft makes no performance claim.
- A joint SMO--Neural-ILT system, soft-heatmap hotspot targets, and external validation on official LithoBench and MaskOpt data remain proposals.
- After the superseded-plan filename correction, the parent test harness reported 169 of 169 tests passed, zero skips, in 78.102 seconds. The source-coverage module suite also passed all 42 tests, and the prospective plan validated at its pinned SHA-256.
- Muse approved the final code review with no blockers. The final Grok code review timed out; its earlier strategy review completed. The required Claude Opus 5.5 High review remains pending because the exact model is unavailable under the account usage limit until November 3, 2026; no substitute review is represented as completed.
- The authors report an accepted prior SBCCI paper on Neural-ILT/PV-band heatmap fine-tuning. The exact title, author list, proceedings name, year, pages, and DOI/URL need confirmation before adding a citation.
- The third author email was preserved from the supplied source as agostini@inf.ufpel.edu.b; its .b ending looks incomplete and needs confirmation. It was not guessed or corrected.
- The PDF previously associated with this manuscript is generic IEEE template output, not a verified compilation of this revised source. The built-in compiler call failed with a Windows helper/setup error, so no PDF has been validated. Regenerate and review a PDF from the revised .tex before circulation.

## Provenance

The supplied starting TeX remains unchanged at C:\Users\danie\Downloads\conference_101719 (2).tex. This revised draft replaces unsupported claims that joint SMO and Neural-ILT were implemented, that LithoBench/MaskOpt were evaluated, that generalization was demonstrated, or that the method improved process-window performance.
