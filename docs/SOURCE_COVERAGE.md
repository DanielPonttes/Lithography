# FIT source coverage experiment

This source-only experiment optimizes the same 49 supported light-source weights against fixed FIT masks. Schema 5 is the consumed historical attempt; schema 6 changed only feasible-start construction and added diagnostics; schema 7 tested a static new-three critical-softcount objective; schema 8 is a prospective static new-three smooth-PV objective ablation. All schemas keep masks fixed and preserve the source constraints and hard FIT qualification. None uses MRC, dynamic hotspot weights, calibration in the loss or ranking, or final3 data.

## Objective and solver

Schemas 5 and 6 optimize the existing target-aware critical-corner softcount averaged equally over all seven FIT layouts. The fixed beta schedule is 200, 400, and 800 with 68 Frank–Wolfe steps at each beta. Five registered starts use seeds 17, 29, 43, 71, and 101. The consumed schema-5 attempt used deterministic simplex jitter and, after guard failure, fell back to the seed-17 reference for all five seeds. That run completed 204 steps per seed without a qualified checkpoint; repeated starts therefore did not provide independent evidence.

The 49 source samples are the disk support of the pixelated source, defined by radius at most `sigma_outer = 0.9`; this includes the center and points inside `sigma_inner = 0.3`. The inner sigma parameter defines the initial annular illumination profile, not a restriction on which of the 49 weights may be optimized.

Schema 6 replaces only start construction and adds diagnostics. It keeps the same masks, guards, physics, objective, seed list, step schedule, qualification gates, teacher, and geometry. The fixed FIT mask and target hashes must match the consumed attempt before new bases are prepared.

For each seed, schema 6 draws up to eight 49-dimensional standard-normal directions from `numpy.default_rng(seed)`, normalizes each by its L2 norm, and minimizes that direction over the guarded source domain with the existing LMO. It tests convex mixtures with the seed-17 reference in the fixed alpha order 0.5, 0.25, 0.125. It accepts the first candidate that passes the guarded-domain check, the full original nominal-q polytope, and the float32 critical-guard audit, and whose source-vector hash is new and L1 distance exceeds 1e-4 from the reference and every earlier accepted start. This search uses feasibility and registered distinctness only; image metrics and loss values do not rank starts. If no candidate passes within eight LMO directions or the 60-second per-seed initialization budget, the seed fails closed without reference fallback.

Initialization and diagnostics count inside the existing 360-second seed and 1,800-second total budgets. Schema 6 records per-layout softcount loss, 49-vector gradient, and gradient norm at the initial beta 200 point, the beta 400 and 800 transitions, and the final beta 800 point. At each snapshot the arithmetic mean of the seven layout values and gradients must match the aggregate objective within absolute tolerance 1e-10. These diagnostics do not affect updates, checkpoint qualification, or ranking.

## Schema-6 result and schema-7 objective ablation

The schema-6 five-seed run completed all 204 steps for every seed with four complete per-layout diagnostic snapshots per seed. Its feasible-start search accepted five distinct starts, passed the registered audits, and did not fall back. The seeds nevertheless reached the same final source-vector SHA256. Each had one accepted update in each beta phase and 67 stationary-tolerance iterations per phase; no checkpoint qualified, and calibration remained closed.

At beta 800, comparing the transition snapshot with the final snapshot isolates progress within the same beta. The old-four mean soft loss changed by −7.3729811e−6, contributing −4.2131320e−6 after the fixed 4/7 objective weight. The new-three mean changed by +1.3433525e−6, contributing +5.7572250e−7 after its 3/7 weight. The all-seven objective therefore fell by 3.6374095e−6 while the new-three surrogate rose slightly. This is an observed tradeoff in this FIT-only run, not a claim of global impossibility or generalization.


Schema 7 changes only the Frank–Wolfe and line-search objective to the arithmetic mean of the three fixed new-layout softcount losses, with static weight 1/3 each. It reuses the registered schema-6 feasible-start procedure and the same five seeds, source physics, beta schedule, budgets, guards, fixed masks, and all hard FIT gates. The four original FIT layouts remain in the guarded source domain and hard qualification; they are omitted only from the optimization loss and gradient. Per-layout diagnostics still report all seven losses and gradients and separately verify parity between the new-three per-layout mean and the new-three optimization objective. The common checkpoint rank continues to use the all-seven softcount mean at beta 800, so objective training and checkpoint ranking are reported separately.

The completed schema-7 run is FIT-only and did not qualify any checkpoint. All five seeds completed 204 steps and four diagnostic snapshots without fallback; each had one accepted update and 203 stationary-tolerance iterations. The five starts were distinct. All five final sources had the same float32 hash; their float64 vectors differed only at the last bits (maximum pairwise L1 distance 2.6945676592782242e-14). Runtime was 586.786348 seconds. Calibration remained closed and final3 was never indexed or evaluated. The new-three last-checkpoint means were PV-band 162, nominal L2 35, and worst-dose L2 94.333333, versus reference means 161.333333, 35.333333, and 93.333333. Nominal L2 improved by one aggregate mismatch, while PV-band and worst-dose L2 each worsened; the required strict PV-band reduction and no-worse hard gates therefore failed. The frozen report SHA256 is `ec993a4a2b5fb566816c05cad863c84ac0b525d0fc36d6dc3da3fe2d6798ece5`.

Schema 8 is a prospective direct smooth-PV objective ablation. For each of the three new FIT layouts it optimizes the pixel mean
`sigmoid(beta*(1.02*aerial - 0.225)) - sigmoid(beta*(0.98*aerial - 0.225))`,
with equal static weight across the three layouts and the same beta continuation 200, 400, 800. It introduces no target-fidelity penalty. The existing four-layout guards, all hard FIT gates, feasible starts, budgets, steps, and all-seven critical-softcount beta-800 checkpoint ranking remain unchanged. Diagnostics preserve the all-seven critical-softcount comparisons and separately check value and gradient parity for the new-three smooth-PV objective. The schema-8 plan is prospective, has not been run, and makes no measured-performance claim; its local plan SHA256 is `de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127`.

The schema-7 hard qualification remains unchanged: every seed must pass the original-four FIT gate and the strict new-three gate, which requires a strict new-three mean hard-PV reduction and no regression in nominal or worst-dose L2 versus the fixed reference. All five seeds must qualify before calibration can open. The schema-6 attempt is immutable; schema 7 carries its plan and failed FIT-only report hashes as explicit supersession lineage. This is an ablation plan, not a reported improvement.

## Consumed schema-5 attempt

The schema-5 run is closed and consumed. Each of the five seeds completed 204 steps, with three accepted updates total (one in each beta phase) and 201 stationary Frank–Wolfe iterations distributed across the schedule. There were zero qualified checkpoints. Every seed fell back to the same seed-17 reference. Its original FIT mean was PV-band 234.5, nominal L2 0, and worst-dose L2 150.25. The three new FIT layouts at the fixed reference had mean PV-band 161.333333, nominal L2 35.333333, and worst-dose L2 93.333333. The last FIT checkpoint had PV-band 161.333333 and worst-dose L2 93.333333, while nominal L2 was 35.666667, one aggregate nominal pixel above the reference. It did not qualify. These repeated starts are one computational experiment, not independent observations.

Hard-metric checkpoints are recorded after steps 25, 50, 68, 75, 100, 125, 136, 150, 175, 200, and 204. Every iteration stores its loss gradient, LMO vertex and objective, post-step 49-vector and SHA256, and step status. The current vector, partial history, and checkpoints are atomically written during the run. A seed has a 360-second solver budget; the five seeds share an 1,800-second solver budget. Original-data validation and identity checks before each seed are recorded outside that solver budget. Solver iterations and progress writes count against it.

## Prospective FIT geometries

The geometry functions are local to `source_coverage.py`; they do not call the legacy layout factory or calibration/held-out generators. The fixed 128 × 128 masks are defined by these half-open rectangles and contact centers:

- `fit_coverage_finite_ribbons_v1`: horizontal `[y0:y1, x0:x1]` rectangles `(9,21,8,112)`, `(29,43,21,119)`, `(58,71,4,94)`, `(81,97,28,124)`, `(106,118,14,87)`; vertical rectangles `(y0,y1,x0,x1) = (21,54,16,28)` and `(52,78,93,107)`.
- `fit_coverage_asymmetric_line_ends_v1`: vertical bars `(x0,width,y0,y1) = (12,12,10,48)`, `(40,14,24,75)`, `(76,13,8,56)`, `(103,12,48,113)`; joined end jog rectangles `[61:75,40:67]`, `[61:88,53:67]`, and `[91:105,80:112]`.
- `fit_coverage_irregular_contacts_v1`: eight square contacts `(cx,cy,width) = (18,20,20)`, `(47,21,22)`, `(80,18,20)`, `(111,22,22)`, `(16,89,22)`, `(47,87,20)`, `(79,92,24)`, `(112,88,20)`. Their two rows are non-collinear and remain separate under 8-connectivity.

The original synthetic teacher uses the frozen 193 nm, NA 1.35, 4 nm pixel, 128 × 128 raster, and threshold/resist constants. New targets are generated once, after the attempt marker is consumed and before optimization. Geometry parameters are fixed in code before target generation; they are not corrected in response to scores. Duplicate mask/target hashes are compared only against the four permitted original FIT layouts and the three new layouts.

## Guard domain and qualification

The full original nominal-q polytope is checked for each candidate. The LMO uses exact sparse pruning: simplex vertex bounds certify redundant rows; a same-pixel LP-anchor critical protection at epsilon `2e-6` certifies stronger nominal margins. Counts and digests record both pruning maps. Reference critical guards that overlap those same anchor protections are also removed by exact same-row dominance. There is no row quantization or approximate merging.

The original LP-anchor critical protections and original FIT qualification remain fixed. Reference and new-layout guard floors are positive and capped by half their reference margin. New guards use deterministic row-major samples of at most 128 target-positive and 128 target-negative pixels per new layout, retaining only reference-correct pixels. The seed-17 reference must be feasible in the augmented domain; LP-anchor feasibility under added reference/new guards is recorded as diagnostic metadata. Pixelwise float32 audits use the same contraction order as the legacy hard-print path and check anchor/reference critical-correct pixels plus sampled new guards. Schema-6, -7, and -8 starts must also pass the full original nominal-q audit, which is deliberately not pruned.

A checkpoint qualifies only if it preserves the original FIT gates: mean hard PV-band at most 234.5, mean nominal L2 zero, and mean worst-dose L2 at most 150.25. It must also strictly reduce the new-three mean hard PV-band versus the fixed seed-17 reference, keep the new-three nominal and worst-dose L2 means no higher than that reference, pass the full original polytope and guard checks, pass the float32 guard audit, and have no blank positive target at any dose.

Qualified checkpoints rank by new-three mean hard PV-band, new-three mean worst-dose L2, new-three mean nominal L2, the common all-seven critical-softcount objective evaluated at beta 800, L1 distance to the original LP anchor, then registered checkpoint order. Training loss at beta 200/400/800 is reported separately. Calibration never breaks ties.

## Inputs, identity, and one-use boundary

The runner parses each JSON from the same bytes whose SHA256 it records and deserializes the dataset from the already-hashed bytes with `weights_only=True`. It indexes only `payload["fit"]`; final3 is never indexed or evaluated. FIT mask/target hashes, ordered IDs, pixel size and raster are checked against diagnostic metadata. Every original float32 basis is checked against its diagnostic hash, shape, dtype, and direct simulator/basis parity. Schema 6 rechecks the consumed schema-5 plan and closed failed-FIT report. Schema 7 additionally rechecks the frozen schema-6 plan and failed FIT-only report by pinned SHA256, validating their attempt boundary without using calibration metrics. Schema 8 rechecks both the schema-6 and schema-7 attempts.

The historical lineage parity comes from the pinned predecessor report and is validated separately from fresh direct parity checks. The read-only prerequisite-manifest validator verifies all 14 lineage artifacts. Their byte hashes, along with the plan, data, diagnostic, predecessor report/plan, and clean committed source identity, are rechecked before the marker and around each seed/calibration boundary. Schema 8 additionally pins and validates the consumed, closed schema-7 FIT report; the validator does not use calibration metrics.

Preflight mode only validates metadata and hashes; it never deserializes the dataset, prepares optics, or consumes the marker. Run mode completes all original-FIT deterministic checks before creating the marker. The marker is created with exclusive file creation immediately before new FIT teacher-target generation. If that generation or any later stage fails, the marker remains consumed and the report records the failure; there is no retry under the same plan.

Calibration is opened once only after all five complete, qualified FIT-only vectors are frozen and identities are rechecked. It must match the four registered calibration IDs, 4 nm pixels, 128 × 128 raster, diagnostic mask/target and basis hashes, and fresh direct parity. Its gates remain mean PV-band ≤239.4, mean nominal L2 ≤56.175, mean worst-dose L2 ≤157.2375, every seed PV-band <268, and no blank positive target at any dose. Calibration is gate-only, not optimizer input, and a post-open failure is recorded as `opened_then_failed_no_retry`.

## Execution

Use the reviewed, clean committed source tree on the registered CUDA host. Keep the plan and prerequisite manifest in the same parent directory; write the report outside the repository.

```powershell
python scripts/optimize_source_coverage.py `
  --mode preflight `
  --plan-file "<frozen-plan.json>" `
  --expected-plan-sha256 7b4cbc414bc2b2a6ee72fac3a633e27adb2bcb2b299b1adb79dbfa9cbc3fc6d2 `
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44

python scripts/optimize_source_coverage.py `
  --mode run `
  --plan-file "<frozen-plan.json>" `
  --expected-plan-sha256 7b4cbc414bc2b2a6ee72fac3a633e27adb2bcb2b299b1adb79dbfa9cbc3fc6d2 `
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44 `
  --output-root "<absolute-output-directory-outside-repository>"
```

The first commands document the consumed schema-5 attempt; do not rerun that plan. The schema-6 and schema-7 plans and runs are also consumed and must not be rerun. The current prospective attempt is schema 8. After its code is reviewed and committed cleanly on the CUDA host, place the external plan in the same parent directory as the prerequisite manifest:

```powershell
python scripts/optimize_source_coverage.py `
  --mode preflight `
  --plan-file "/home/daniel/experiments/robust-source-quality-20261005-766872/direct_pv_objective_candidate_plan.json" `
  --expected-plan-sha256 de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127 `
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44

python scripts/optimize_source_coverage.py `
  --mode run `
  --plan-file "/home/daniel/experiments/robust-source-quality-20261005-766872/direct_pv_objective_candidate_plan.json" `
  --expected-plan-sha256 de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127 `
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44 `
  --output-root "<absolute-output-directory-outside-repository>"
```

The unit tests use only small synthetic arrays and fixture metadata. They do not load the dataset, generate teacher targets, prepare optical bases, or score calibration.

## Limits of interpretation

Frank–Wolfe optimizes a nonconvex surrogate with registered computational restarts; neither the reported gap nor a qualified checkpoint is a global-optimum certificate. The five seeds are computational restarts, not independent generalization trials. Schema 5 recorded five identical reference fallbacks and no qualified checkpoint. Schemas 6 and 7 completed with distinct feasible starts but had no qualified checkpoints. Schema 8 is implemented as a prospective objective ablation and has not been benchmarked. The reused calibration set is development data and does not demonstrate generalization. No result from this experiment is valid unless the complete identity and unchanged hard FIT gates pass.
