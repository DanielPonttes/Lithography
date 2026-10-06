# Robust dose-corner fitting with a shared source

This experiment adds one target-aware quality objective for the existing scalar-Abbe, source-only model. It changes the 49 active illumination weights; masks, targets, optics, and process corners stay fixed. It does not implement MRC, mask updates, dynamic hotspot weights, or SOCS parity.

The motivation is process-window optimization across process conditions. RMO-ILT frames this as multi-objective mask optimization and uses uniform gradients across its objectives ([ICCAD 2024 paper](https://personal.hkust-gz.edu.cn/yuzhema/papers/ICCAD2024-PV-ILT.pdf)). Earlier source-mask optimization work also studies illumination distributions against integrated process window ([IBM publication](https://research.ibm.com/publications/global-optimization-of-the-illumination-distribution-to-maximize-integrated-process-window)). This implementation is a new source-only adaptation; it is neither a reproduction of RMO-ILT nor a transfer of its guarantees.

## Objective and feasible region

For each fixed fit pixel `i`, let `t_i` be its binary target, `y_i = 2 t_i - 1`, `A_i` the 49 basis intensities, `w` the source weights, and `T = 0.225`. The intensity is `I_i(w) = A_i w`. The dose corners are `{0.98, 1.00, 1.02}`.

The fixed nominal LP region remains:

```text
w >= 0
sum(w) = 1
y_i * (I_i(w) - T) >= rho * m_LP  for every pixel in all four fit layouts
```

`rho` is locked to the selected entry in the frozen schema-2 candidate plan (initially `0.5`; later entries are `0.05` and `0.01`). A different value requires selecting its registered candidate index and meeting the prior-run sequence rules. Every candidate must also have zero nominal hard-print L2 on the original hash-verified float32 basis. The data and diagnostic paths must exactly match the plan, and the runner enforces checkpoint interval `25` and solver limit `60` seconds.

The new loss penalizes the weakest signed dose-corner margin at each fit pixel:

```text
L(w) = mean_i [ max(0, rho*m_LP - min_{d in {0.98,1.02}} y_i * (d*I_i(w) - T))^2 ] / m_LP^2
```

The nominal corner is already protected by the LP region. Optical basis intensities are checked as finite and exactly nonnegative; source weights are constrained nonnegative. Therefore `I_i(w) >= 0`. For a target-positive pixel, the limiting dose is always `0.98`; for a target-negative pixel, it is always `1.02`. The implementation uses this label-specific analytic form rather than differentiating through a minimum at a tie. It has the same loss on the declared feasible region and is a convex, continuously differentiable squared hinge in `w`.

Dividing by the positive constant `m_LP^2` makes the loss dimensionless without changing its minimizers. The source remains a single global distribution shared by the four fit layouts, so this objective may still expose limits in how well one source can represent distinct geometries.

## Optimization and diagnostics

The runner reuses the existing float64 LP region and HiGHS linear minimization oracle, with 49 source variables. Frank–Wolfe uses a monotone Armijo backtracking search and at most 204 iterations per seed. Its floating-point gap is an approximate numerical stopping diagnostic for this convex continuous objective over the nominal fit polytope. It is not a rigorous global certificate and does not certify hard PV-band reduction or calibration performance.

The fixed seeds are `17, 29, 43, 71, 101`. Each fit checkpoint is scored only on the four fit layouts. Selection retains the existing order: lower hard PV-band, lower worst-dose L2, lower common `beta=800` smooth-PV score, then earlier checkpoint. The LP anchor is included. A candidate also needs a passing polytope check and zero nominal float32 fit L2.

At the anchor and fit checkpoints, diagnostics include, by fit layout and dose:

- signed-margin minimum and pixel quantiles;
- hard false-positive and false-negative counts, L2, and PV-band;
- objective values, raw source-gradient norms/cosines, and simplex-tangent-projected norms/cosines for nominal fidelity, each dose-corner fidelity, the smooth-PV proxy, and the robust hinge. The tangent projection removes only the simplex-normal component; it does not account for active LP constraints.

Margins and hard prints are evaluated from the registered float32 basis; the optimization promotes that exact hash-verified basis to float64. Negative optical basis values are rejected rather than clamped or recomputed.

The four synthetic calibration layouts are scored only after all five seeds complete and every seed selects a fit-qualified checkpoint. Calibration receives no gradients. The unchanged development gate requires all of:

- mean PV-band `<= 239.4` pixels;
- mean nominal L2 `<= 56.175` pixels;
- mean worst-dose L2 `<= 157.2375` pixels;
- each seed's PV-band strictly below the LP-anchor mean of `268` pixels;
- no positive-target layout printing zero pixels at any dose;
- all five registered seeds complete and qualify on fit.

A timeout, solver failure, failed line search, or missing fit-qualified checkpoint closes calibration and cannot produce a passing gate. The final-three layouts remain closed and are never indexed or evaluated. These reused calibration layouts are development evidence, not independent generalization results.

An anchor-only feasibility preflight found a nonzero objective (`mLP=0.0002215098124816`, `rho=0.5` hinge `2.8996481688`) and a feasible Frank–Wolfe descent direction. In a separate fit-only first-step diagnostic, the hinge fell from `2.89965` to `2.04871` while hard PV-band rose from `332.75` to `365` pixels and worst-dose L2 rose from `215.75` to `280`. These diagnostics show why hinge loss is not the selection metric: every checkpoint, including the LP anchor, is selected by the frozen hard fit rule above. They do not show a quality gain; only the preregistered five-seed run and unchanged calibration gate can assess the next experiment.

## Running

The runner requires CUDA and defaults to an RTX 5090 check. It requires the reviewed schema-2 `candidate_plan.json`, whose SHA, selected candidate index/rho, iterations, stopping rule, gates, code, data, and runtime are frozen in `protocol.json` before gradient diagnostics. A real benchmark also requires a clean committed source tree at or after the plan's base commit. Source files, basis/data inputs, Python, NumPy, SciPy, and Torch versions are recorded. Calibration remains closed until all five fit-only selections are complete and qualified.

```bash
python3 scripts/optimize_source_robust_corners.py \
  --dataset-file /path/to/previous-datasets.pt \
  --diagnostic-file /path/to/previous_diagnostic.json \
  --candidate-plan /path/to/candidate_plan.json \
  --output-root /path/to/robust-source-corners/runs \
  --candidate-index 0 \
  --rho 0.5 \
  --iterations 204
```

For a timeout or interruption, resume the same run with the same plan and inputs. Omit `--output-root` and use its run directory:

```bash
python3 scripts/optimize_source_robust_corners.py \
  --dataset-file /path/to/previous-datasets.pt \
  --diagnostic-file /path/to/previous_diagnostic.json \
  --candidate-plan /path/to/candidate_plan.json \
  --resume-run /path/to/robust-source-corners/runs/<run-id>
```

`run_state.json` is the checksummed canonical state; `results.json` and `progress.json` are repairable display sidecars. Accepted Frank–Wolfe steps persist the current 49 weights, next step, candidate order, seed record, and frozen identity atomically. Resume validates the run identity, checksum, phase, fit selections, and checkpoint metadata before repairing sidecars or weight files. Rejected identity/checksum/phase validation does not rewrite existing run artifacts. Completed-run resume is read-only, and the CLI reads the canonical state even if a display sidecar is stale. Checkpoints load with Torch's weights-only mode. Timeout during fit resumes the same seed and remaining budget; calibration progress is resumable only after all five fit selections pass. The lock prevents simultaneous writers. A forced process termination can leave `.run.lock`; inspect the host process and lock owner before an operator removes a stale lock.

A candidate at `rho=0.05` or `0.01` needs the previous candidate run passed through `--prior-run`; advancing is allowed only if all five seeds completed and the previous unchanged development gate failed. A timed-out run should be resumed. If a numerical failure is retried as a new attempt, use `--prior-run` for that same-rho attempt and preserve both directories. Do not advance after a timeout or numerical failure. A completed run with a rejected gate is a completed experiment, not an accepted quality improvement. Compare the saved calibration means with the fixed gate; do not widen it to obtain a pass.

## Limits on hotspot identification claims

For the current sigmoid-50 resist and the declared 0.98/1.02 dose pair, the maximum pixelwise soft corner difference over intensity is about `0.1129` (near intensity `0.2285`). A heatmap cutoff of `0.15` therefore produces no positive pixels in this model. That numerical bound describes this scalar-Abbe configuration; it does not establish a physically meaningful hotspot threshold. Hard PV-band counts and per-corner FP/FN measure print stability against the synthetic target, not the accuracy of hotspot detection. F1, recall, or precision for identified hotspots require independent per-pixel or per-region hotspot labels that were not used to define the masks, objective, or threshold.

