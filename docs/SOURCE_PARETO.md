# Pareto protected-corner MILP

This is a source-only diagnostic for the fixed 49-weight annular source and
fixed FIT masks. It minimizes the number of originally incorrect anchor pixels
that cannot be repaired to a fixed epsilon buffer. The integer objective is a
conservative **epsilon-buffered unrepaired anchor-error count**, not an exact
hard-PV objective. Calibration is reused development data and never enters the
objective, selection, or tie-breaking. `final3` is never indexed or evaluated.

## Model and exact row reduction

For a target-positive FIT pixel the limiting process corner is dose `0.98`; for
a target-negative pixel it is `1.02`. With `s = +1` for target one and `s = -1`
for target zero, its signed critical margin is

```text
m_i(w) = s_i * (d_i * B_i @ w - 0.225)
```

The source weights stay on the simplex. The original nominal FIT polytope has
the fixed floor `q = rho * mLP`, with `mLP = 0.0002215098124816199`,
`rho = 0.01`, and `epsilon = 2e-6`. A binary `z_i` is assigned to each of the
1,331 anchor-error pixels; the model minimizes `sum(z_i)` subject to
`m_i(w) + M_i*z_i >= epsilon`. The tight deactivation value is
`M_i = max(0, epsilon - L_i)`, where `L_i` is the stronger of the simplex and
nominal-floor lower bounds.

The full nominal polytope is first checked exactly against the ordered FIT
bases and targets: matrix, labels, right-hand side, equality, bounds, and margin
floor must match. For the four registered 128x128 FIT layouts, this is 65,536
nominal rows. Only then are the 64,205 rows for originally correct
critical-corner pixels omitted, leaving exactly 1,331 nominal rows for the
anchor errors. The registered model is expected to have roughly 23,335 total
rows after protected-row vertex pruning; each run records its actual count.
Each originally correct pixel either keeps its critical-corner row, which is
stronger than the nominal row, or has a certified simplex lower bound proving
that the critical margin already exceeds `epsilon`. In either case its
nominal signed margin exceeds `q`:

```text
target 1: .98*A - .225 >= epsilon  =>  A - .225 >= (epsilon + .02*.225)/.98
target 0: .225 - 1.02*A >= epsilon => .225 - A >= (epsilon + .02*.225)/1.02
```

Both implications have margin greater than the registered `q`. The 1,331
nominal rows corresponding to anchor errors remain in the MILP. No row is
quantized or approximately deduplicated. Model metadata records the original,
pruned, and retained nominal row counts, plus pixel mappings for binaries,
kept protected rows, and both protected-row pruning proofs. The float32 hard
print partition and float64 MILP mapping must match exactly before solving.

After every solve, qualification still calls the full `poly.verify` over the
original nominal polytope. It also checks the original float32 sigmoid-50
prints, zero nominal hard FIT errors, all originally correct critical pixels,
the complete source-weight-plus-binary incumbent vector, integrality,
residuals, objective, and its per-seed float64 little-endian SHA256.

## Frozen inputs and one-use attempt

The schema-4 prospective candidate plan and schema-1 source-family manifest
are pinned by SHA256 in `source_pareto.py`. The runner validates the three
hinge and three softcount runs, their state checksums and lineage, data and
code identities, and all frozen FIT qualifications. A softcount run must name
the exact listed hinge manifest path and SHA256 in
`shared.prerequisite_manifest`. Legacy hinge rows use the validated manifest
header's `code_commit`; they do not need a per-row `git_head` field.

The runner loads only `payload["fit"]`. It prepares FIT bases only and never
reads calibration masks, targets, or bases while building or solving the MILP.
Plan, dataset, diagnostic, prerequisite manifest and lineage artifacts, source
hashes, and Git provenance are rechecked before and after every solver call and
before calibration access. JSON identity checks hash and parse the same bytes.
Every visited lineage ancestor and retry contributes its protocol and run-state
files to the identity snapshot; canonical duplicate paths are deduplicated only
when their recorded SHA256 values agree. The dataset must still match both its
initial preflight file hash and the dataset hash recorded in the diagnostic.
The candidate plan and prerequisite manifest must reside in the same directory,
so copying only a plan cannot create a fresh adjacent attempt-marker namespace.
At runtime, the frozen gate values are compared directly with the constrained
FIT and calibration gate constants, including the distinct 239.4 development
band gate and 268 per-seed control.

An exclusive marker named
`source_pareto_attempt_<candidate-plan-sha256>.json` is created beside that
plan, regardless of the output directory. It is consumed with `O_EXCL`
immediately before the first solver call and is never removed after
consumption, including on interruption or failure. A second run with the same
plan cannot reset its five-seed, 1,800-second solver budget; a new prospective
plan is required for another attempt. Every run writes solver status, elapsed
time, budget use, and the complete incumbent vector and per-seed SHA256 as soon
as the solver returns, before identity checks, audit, or FIT qualification.
Audit and qualification then update that same seed record. A later qualifier
failure therefore keeps the incumbent and solver telemetry and leaves
calibration closed.

Calibration opens once only after all five registered seeds (17, 29, 43, 71,
101) are optimal with exactly zero MIP gap, all FIT checks pass, and at least
one candidate improves FIT PV over the LP anchor. Before calibration opens,
failures leave its status `closed`. After the access boundary, failures remain
`opened_then_failed; no_retry`. A FIT plateau remains closed and makes no
global-impossibility claim.

## Solver and output

Production requires Python 3.12, SciPy 1.17.1 with its bundled private HiGHS
1.12.0 API, CUDA for registered optical-basis preparation, and a clean
committed checkout descended from the plan's base commit. There is no external
`highspy` fallback. Each seed uses feasibility tolerances `1e-9`, zero relative
and absolute MIP gaps, four threads, its registered random seed, and at most
360 seconds; the shared five-seed solver budget is 1,800 seconds. The runner
checks option set and readback status, model and run statuses, warm-start
acceptance, runtime telemetry, dual bound, gap, and incumbent residuals.

Run from the server checkout with absolute paths and an output directory
outside the source tree:

```bash
/usr/bin/python3.12 scripts/diagnose_source_pareto.py \
  --dataset-file /absolute/path/previous-datasets.pt \
  --diagnostic-file /absolute/path/diagnostic.json \
  --candidate-plan /absolute/path/pareto_candidate_plan.json \
  --output-root /absolute/path/pareto-results
```

Each invocation creates a UUID-named result directory and atomically writes
`diagnostic.json` with the plan and input hashes, model dimensions and row
reduction counts, solver telemetry, per-seed incumbent vector and hash, FIT
qualification, and calibration state. A time-limit incumbent is preserved as
an incomplete result and cannot open calibration.

## CPU toy checks

Tests use only tiny synthetic bases and stubs; they do not open the registered
dataset or run production/calibration data. The HiGHS toy solve runs in a clean
child process because a preceding `linprog` may initialize SciPy's
process-global scheduler with a different thread count. The production solver
does not reset that scheduler or fall back to different thread settings.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m unittest discover -s tests -p test_source_pareto.py -v
```
