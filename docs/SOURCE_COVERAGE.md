# FIT source coverage experiment

This prospective experiment optimizes the same 49 supported light-source weights against seven fixed FIT layouts: the four original FIT layouts and three independently generated coverage layouts. It keeps masks fixed during optimization and uses static equal weighting across layouts. It does not use MRC, dynamic hotspot weights, calibration in the loss or ranking, or held-out data. The frozen schema-5 plan is stored outside the repository and pinned by SHA256 in the runner.

## Objective and solver

The objective is the existing target-aware critical-corner softcount averaged equally over all seven FIT layouts. The fixed beta schedule is 200, 400, and 800 with 68 Frank–Wolfe steps at each beta. Five registered starts use seeds 17, 29, 43, 71, and 101. Each starts from deterministic simplex jitter of the frozen seed-17 reference; if the jitter violates the guard domain, the exact reference is used and the fallback is recorded. This is a fixed 204-step procedure per seed, with no claim of global optimality.

Hard-metric checkpoints are recorded after steps 25, 50, 68, 75, 100, 125, 136, 150, 175, 200, and 204. Every iteration stores its loss gradient, LMO vertex and objective, post-step 49-vector and SHA256, and step status. The current vector, partial history, and checkpoints are atomically written during the run. A seed has a 360-second solver budget; the five seeds share an 1,800-second solver budget. Original-data validation and identity checks before each seed are recorded outside that solver budget. Solver iterations and progress writes count against it.

## Prospective FIT geometries

The geometry functions are local to `source_coverage.py`; they do not call the legacy layout factory or calibration/held-out generators. The fixed 128 × 128 masks are defined by these half-open rectangles and contact centers:

- `fit_coverage_finite_ribbons_v1`: horizontal `[y0:y1, x0:x1]` rectangles `(9,21,8,112)`, `(29,43,21,119)`, `(58,71,4,94)`, `(81,97,28,124)`, `(106,118,14,87)`; vertical rectangles `(y0,y1,x0,x1) = (21,54,16,28)` and `(52,78,93,107)`.
- `fit_coverage_asymmetric_line_ends_v1`: vertical bars `(x0,width,y0,y1) = (12,12,10,48)`, `(40,14,24,75)`, `(76,13,8,56)`, `(103,12,48,113)`; joined end jog rectangles `[61:75,40:67]`, `[61:88,53:67]`, and `[91:105,80:112]`.
- `fit_coverage_irregular_contacts_v1`: eight square contacts `(cx,cy,width) = (18,20,20)`, `(47,21,22)`, `(80,18,20)`, `(111,22,22)`, `(16,89,22)`, `(47,87,20)`, `(79,92,24)`, `(112,88,20)`. Their two rows are non-collinear and remain separate under 8-connectivity.

The original synthetic teacher uses the frozen 193 nm, NA 1.35, 4 nm pixel, 128 × 128 raster, and threshold/resist constants. New targets are generated once, after the attempt marker is consumed and before optimization. Geometry parameters are fixed in code before target generation; they are not corrected in response to scores. Duplicate mask/target hashes are compared only against the four permitted original FIT layouts and the three new layouts.

## Guard domain and qualification

The full original nominal-q polytope is checked for each candidate. The LMO uses exact sparse pruning: simplex vertex bounds certify redundant rows; a same-pixel LP-anchor critical protection at epsilon `2e-6` certifies stronger nominal margins. Counts and digests record both pruning maps. Reference critical guards that overlap those same anchor protections are also removed by exact same-row dominance. There is no row quantization or approximate merging.

The original LP-anchor critical protections and original FIT qualification remain fixed. Reference and new-layout guard floors are positive and capped by half their reference margin. New guards use deterministic row-major samples of at most 128 target-positive and 128 target-negative pixels per new layout, retaining only reference-correct pixels. The seed-17 reference must be feasible in the augmented domain; LP-anchor feasibility under added reference/new guards is recorded as diagnostic metadata. Pixelwise float32 audits use the same contraction order as the legacy hard-print path and check anchor/reference critical-correct pixels plus sampled new guards.

A checkpoint qualifies only if it preserves the original FIT gates: mean hard PV-band at most 234.5, mean nominal L2 zero, and mean worst-dose L2 at most 150.25. It must also strictly reduce the new-three mean hard PV-band versus the fixed seed-17 reference, keep the new-three nominal and worst-dose L2 means no higher than that reference, pass the full original polytope and guard checks, pass the float32 guard audit, and have no blank positive target at any dose.

Qualified checkpoints rank by new-three mean hard PV-band, new-three mean worst-dose L2, new-three mean nominal L2, a common soft objective evaluated at beta 800, L1 distance to the original LP anchor, then registered checkpoint order. Training loss at beta 200/400/800 is reported separately. Calibration never breaks ties.

## Inputs, identity, and one-use boundary

The runner parses each JSON from the same bytes whose SHA256 it records and deserializes the dataset from the already-hashed bytes with `weights_only=True`. It indexes only `payload["fit"]`; the held-out split is never selected or evaluated. FIT mask/target hashes, ordered IDs, pixel size and raster are checked against diagnostic metadata. Every original float32 basis is checked against its diagnostic hash, shape, dtype, and direct simulator/basis parity.

The historical lineage parity comes from the pinned predecessor report and is validated separately from fresh direct parity checks. The read-only prerequisite-manifest validator verifies all 14 lineage artifacts. Their byte hashes, along with the plan, data, diagnostic, predecessor report/plan, and clean committed source identity, are rechecked before the marker and around each seed/calibration boundary.

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

The unit tests use only small synthetic arrays and fixture metadata. They do not load the dataset, generate teacher targets, prepare optical bases, or score calibration.

## Limits of interpretation

Frank–Wolfe optimizes a nonconvex surrogate with registered computational restarts; neither the reported gap nor a qualified checkpoint is a global-optimum certificate. The five seeds are restarts, not independent generalization trials; guard-infeasible jitter can make them converge from the same seed-17 warm start. The reused calibration set is development data and does not demonstrate generalization. No result from this experiment is valid unless the complete identity and original gates pass.
