# Target aware critical corner softcount

This is a source only experiment over the existing 49 illumination weights. Fit
masks and targets remain fixed. It optimizes the source inside the existing
nominal LP region and retains the float32 nominal hard fit check.

For nonnegative aerial intensity \(I_i(w)\), the target signed margins at the
two dose corners are

\[
m_i(w) =
\begin{cases}
0.98 I_i(w)-0.225, & y_i=1,\\
0.225-1.02 I_i(w), & y_i=0.
\end{cases}
\]

The fit loss is the equal layout mean of each layout's mean pixel
\(\sigma(-\beta m_i(w))\). It has no LP margin offset or normalization.
The limiting corner follows from the nonnegative intensity and monotone dose
response: low dose for target positive pixels and high dose for target negative
pixels. The implementation rejects negative optical basis values rather than
clamping them.

This surrogate is smooth and nonconvex. The Frank Wolfe gap is recorded as a
floating point stationarity diagnostic for the current continuous objective;
it is not a global optimality certificate or a guarantee about hard PV,
calibration, or generalization. Each seed starts from its LP anchor and a
seeded 5 percent feasible vertex mix. The scheduled beta blocks are 200, 400,
and 800 with 68 accepted-step opportunities each. Checkpoints are saved after
local steps 25, 50, and 68. Early stationarity saves a block candidate and
continues at the next beta; timeout resumes the same block and step. A terminal
solver, numerical, verification, or no progress failure closes calibration.
Each attempt records at most 205 outer LMO invocations per seed, including
initialization; solver subattempts are recorded separately. A timed out outer
LMO consumes one of the 205 outer-call slots even when its returned subattempt
list is empty. Timeouts resume the same attempt. A new same-rho attempt is
allowed only after a terminal pre-calibration failure.

Each outer LMO event keeps its raw solver status and outcome: it is first saved
as in progress, then as completed, timeout retry, or interrupted retry. A
process interruption after the call was registered but before its solver result
was committed is normalized to an interrupted retry with an explicitly unknown
subattempt count on resume. On resume,
the optimizer keeps that event and registers a new outer call for the same
initialization or beta/local step. Any number of timed out or interrupted calls
may precede the single completed result for that point; all consume the same
205-call attempt budget. A solver failure returned after the run deadline is
treated as a timeout retry while preserving `solver_failure` as the raw solver
status. A solver failure before the deadline remains terminal.

Checkpoint selection stays fit only: lowest hard PV band count, then lowest
worst dose L2, then the common beta 800 smooth PV score, then earliest order.
Every checkpoint must remain feasible and have zero nominal float32 fit L2.
All five seed selections are frozen before calibration scoring. The fixed
development gate remains mean PV band at most 239.4, mean nominal L2 at most
56.175, mean worst dose L2 at most 157.2375, every seed PV strictly below 268,
and no positive target blank at any dose. The calibration images are reused
development data, not independent generalization or detector F1 evidence.
Final three masks remain closed and are neither indexed nor evaluated.

## Required hinge sequence

The prospective schema 3 plan preregisters this objective after the three
schema 2 worst dose hinge candidates (rho 0.5, 0.05, and 0.01) each complete all
five fit selections and fail the same frozen gate. The runner requires the
parent produced `hinge_family_manifest.json` and verifies its candidate order,
protocol and canonical state SHA256 values, identity hashes, source commit,
dataset and diagnostic file hashes, the full diagnostic input identity,
float32/float64 basis parity identity, closed final three status, gate failure,
and five qualified seeds before computing any softcount gradients. The manifest
path and every run directory must be absolute. The manifest SHA256 and exact
objective spec are included in the new run identity. A hinge run cannot be used
as a softcount resume or as its preceding rho run.

For each listed hinge candidate, the runner also walks and verifies its prior
run chain. A candidate may point to an earlier terminal pre-calibration failure
at the same rho when a preserved retry was needed; each retry must increment the
attempt number and match the actual protocol, canonical state, source/data
identity, candidate settings, and recorded artifact hashes. The chain must end
at candidate zero attempt one, with no missing parent, cycle, duplicate listed
run, or stale parent summary. After skipping same-rho retry attempts, each rho
must descend from the exact completed run directory and protocol/state hashes
listed for the preceding candidate. Advancing to a new rho still requires that
listed candidate to have completed all five seeds and failed its frozen gate.

The manifest has `schema_version: 1`, status
`completed_hinge_family_gate_failed`, objective ID
`target_aware_worst_dose_squared_hinge_v1`, the registered hinge plan SHA256,
base commit, dataset SHA256, and three `runs` entries in rho order. Each entry
contains `candidate_index`, `rho`, absolute `run_directory`, `status:
complete`, `gate_passed: false`, the five `{seed, complete, fit_qualified}`
summaries, `protocol_sha256`, `run_state_sha256`, and `identity_sha256`.

## Invocation

Candidate index 0 starts rho 0.5 only after the hinge manifest exists and
validates. Indices 1 and 2 require `--prior-run` pointing to the preceding
completed softcount candidate whose frozen gate failed. A timeout resumes the
same attempt with `--resume-run` and the same candidate plan. A new same rho
attempt is allowed only after a terminal pre calibration training failure and
must preserve the failed attempt. The plan locks the 204 step budget,
25 step checkpoints, and 60 second solver limit.

Example first candidate:

```bash
python3.12 -u -B scripts/optimize_source_robust_corners.py \
  --dataset-file /home/daniel/experiments/constrained-source-20260929-cb2e6c/previous-datasets.pt \
  --diagnostic-file /home/daniel/experiments/constrained-source-20260929-cb2e6c/diagnostic.json \
  --candidate-plan /home/daniel/experiments/robust-source-quality-20261005-766872/softcount_candidate_plan.json \
  --output-root /home/daniel/experiments/robust-source-quality-20261005-766872/softcount-runs \
  --candidate-index 0 --rho 0.5 --iterations 204 \
  --checkpoint-interval 25 --solver-time-limit 60
```

This is a new source only adaptation, not a reproduction of mask optimization
or a transfer of another method's guarantees. The earlier mask based
multiobjective literature motivates checking process corners, but does not
validate this source only surrogate or predict a quality improvement.
