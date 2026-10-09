# Fixed source-segment FIT diagnostic

This is a prospective, one-use diagnostic. It asks whether a fixed grid of
convex mixtures between the frozen seed-17 FIT reference and already frozen
feasible starts or terminal source vectors contains a point that passes the
existing FIT hard gates. It does not optimize source weights and does not
alter any mask, teacher, optical basis, resist parameter, or guard.

## Registered inputs and grid

The plan is derived from the failed, closed schema-6, schema-7, and schema-8
FIT reports and their pinned plan/report hashes. For each seed in
`17, 29, 43, 71, 101`, it evaluates these segments in order:

1. Reference to that seed's schema-8 feasible initialization.
2. Reference to that seed's terminal step-204 schema-6 vector.
3. Reference to that seed's terminal step-204 schema-7 vector.
4. Reference to that seed's terminal step-204 schema-8 vector.

Every segment uses exactly `alpha = k/256` for integer `k=0,...,256`, in
increasing order. The float64 candidate is computed directly as
`(1-alpha)*reference + alpha*endpoint`; it is never projected, clipped, or
renormalized. The complete design contains 20 segments and 5,140 ordered
slots. Endpoints, their float64/float32 hashes, and report provenance are
frozen in the external prospective plan.

## Audits, scoring, and selection

Each slot receives a full float64 simplex, original nominal-polytope, and
augmented guarded-domain check. The existing float32 critical-pixel guard
audit is applied, and the existing `source_coverage.hard_metrics` path scores
all seven fixed FIT layouts over the 0.98, 1.00, and 1.02 dose images. The
unchanged original-four and new-three qualification gates are applied. A
qualified FIT point is ranked only with the existing all-seven beta-800
critical-corner checkpoint rank. Every slot is retained, including exact or
float32 duplicates; score caching may not suppress its own provenance or
float64 audit.

The physical cache is keyed by the exact float32 weight bytes and process
thread count. It reuses only raster metrics and critical-guard outputs.
Each slot recomputes its float64 audit, beta-800 selection objective, source
distances and rank. The screening process uses one CPU thread. Before scoring,
all seven reference layouts must match the original host thread configuration
exactly. Every screen-qualified slot receives an uncached full evaluation at
the original host thread count; only points that pass that evaluation are
ranked. The process restores its original thread setting on every exit.
This avoids claiming a numerical screening artifact as improvement. A negative
screening result is limited to the fixed one-thread grid: candidates missed by
screening are not exhaustively rescored at the host thread count.

The report separately identifies a best passing FIT point and whether every
one of the five seed groups contains a passing point. These are diagnostic
summaries, not an optimizer run or independent generalization result. No
calibration data is opened or used, even if a FIT point passes, and the final
three layouts are never indexed or evaluated.

## Lineage, lifecycle, and limits

`scripts/diagnose_source_segments.py --mode freeze-plan` creates the external
candidate plan after the diagnostic code, tests, and this document are stable.
The plan pins the parent schema-8 plan/report, the schema-6/7/8 failed FIT
reports, fixed layout hashes, physics, endpoint vectors, exact grid/order,
budgets, and source-file hashes. Freezing also installs the plan SHA256 in
`source_segment_sweep.py`. The runner accepts `--mode preflight` (identity
checks only) or `--mode run`; run mode exclusively consumes the new plan's
marker before loading FIT data, generating new FIT targets, preparing bases,
or scoring. It never calls the source-coverage optimizer's `run()` entry point.

The source fingerprint covers the imported optical, training, robustness,
coverage and prerequisite-runner modules as well as the new diagnostic.
Text hashes normalize CRLF to LF for Windows/Linux transport; only the plan
pin literal in the segment module is zeroed to break the pin/hash cycle.
Preflight and each seed/final/failure boundary revalidate all 14 prerequisite
artifacts and the schema-6/7/8 reports and plans. Incomplete runs retain the
best qualified point found so far and explicit records for remaining slots
when the registered grid could be constructed. The CLI returns a nonzero
exit status for error or timeout. Calibration stays closed on every path.

The setup limit is 300 seconds, each seed has a 600-second scoring/audit
limit, and the whole run has a 3,600-second wall limit. Progress is saved
every 128 attempted slots. A timeout or exception preserves the partial slot
history and is incomplete evidence; it cannot be described as proof that the
registered grid has no passing point. The diagnostic does not establish
continuous-segment feasibility, global infeasibility, process-window
performance, or generalization beyond these fixed FIT layouts.

Example invocation after the new source bundle and external plan are staged:

```powershell
python scripts/diagnose_source_segments.py --mode preflight `
  --plan-file /home/daniel/experiments/robust-source-quality-20261005-766872/segment_sweep_candidate_plan.json `
  --expected-plan-sha256 <frozen-sha256>
python scripts/diagnose_source_segments.py --mode run `
  --plan-file /home/daniel/experiments/robust-source-quality-20261005-766872/segment_sweep_candidate_plan.json `
  --expected-plan-sha256 <frozen-sha256> `
  --output-root /home/daniel/experiments/robust-source-quality-20261005-766872/segment-sweep-runs `
  --device cuda
```
