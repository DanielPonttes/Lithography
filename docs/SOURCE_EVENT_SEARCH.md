# Source-only dose-threshold event search

This extension proposes source mixtures along a segment using the fact that
weighted-basis aerial intensity is affine in the 49 source weights. For each
fixed FIT layout, each pixel and each dose, it solves

```text
dose * I(alpha) = threshold,  alpha in [0, 1]
```

It proposes each sorted crossing, both segment endpoints, and one midpoint in
each open interval between crossings. Coincident or numerically near-identical
alphas are merged deterministically; equal float64 source vectors are audited
once with every alias retained. A pixel that stays exactly on the threshold
throughout the segment is recorded, not expanded into infinitely many events.

The float64 affine model generates proposals only. Hard qualification uses the
canonical float32 weighted-basis evaluator at doses 0.98, 1.00 and 1.02, with
threshold 0.225 and sigmoid steepness 50. Qualification and ranking never use
the float64 event estimate. The initial incumbent and best known incumbent
remain in the candidate list ahead of event proposals; the frozen reference
also remains as the comparison point. Unattempted candidates are written down
as such, and an incomplete budget is not evidence that no qualifying point
exists.

Source vectors must be strictly nonnegative and have unit flux within the
declared simplex tolerance. Event alpha tolerance is restricted to `[0, 0.5)`
so deduplication cannot merge the two immutable closed-segment endpoints.
Protected reference/initial/best-known candidates are evaluated before any
ordinary candidate even if their scoring consumes the wall-clock budget; their
time still counts in the reported elapsed time and later candidates are marked
not attempted once a budget is reached.

## Direct simulator confirmation

The diagnostic CLI can validate the frozen schema-8 plan without loading FIT
data, then compare a source JSON through `DifferentiableAbbeLitho.forward`, GPU
weighted-basis evaluation, and canonical CPU `source_coverage.hard_metrics` on
the four frozen original FIT layouts plus the three pinned new FIT layouts.
The report requires exact per-layout hard-count agreement across all three
routes, plus aerial-tensor allclose for the two GPU routes. It installs the
supplied 49 values directly in a fixed source adapter; it does not reconstruct
logits or apply softmax. The report captures
the weight-file byte hash, exact float64/float32 vector hashes, plan input
hashes, direct-versus-basis aerial parity, per-layout hard counts, and candidate
minus reference mean metrics. It loads only the `fit` dataset entry, keeps
calibration closed, never indexes final3, and does not consume an event-search
attempt marker. An existing output filename is refused before FIT preflight,
and the final result is published without replacement. The CLI exits with code
3 when evaluator parity or the frozen aggregate quality gates fail. A gain is
described as confirmed only when the three routes agree and the candidate
passes those gates. If the pinned segment report is supplied, its stored
per-layout counts (including positive-target counts) must also match the
canonical CPU route for the historical comparison to be verified.

```powershell
$python = 'D:\Codex\PythonEnvs\lithography-cu128\Scripts\python.exe'
$data = 'D:\Codex\Lithography\work\server_benchmark\robust-source-quality-20261005-766872'
$plan = Join-Path $data 'direct_pv_objective_candidate_plan.json'
$weights = Join-Path $data 'segment_5fd31de_selected_fit.json'
$sourceReport = Join-Path $data 'segment_5fd31de_report.json'

& $python .\scripts\diagnose_source_events.py `
  --mode preflight `
  --coverage-plan-file $plan `
  --expected-coverage-plan-sha256 de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127 `
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44

& $python .\scripts\diagnose_source_events.py `
  --mode direct-verify `
  --coverage-plan-file $plan `
  --expected-coverage-plan-sha256 de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127 `
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44 `
  --weights-file $weights `
  --source-report-file $sourceReport `
  --expected-source-report-sha256 a66908b8a8b902bdcc1846a35c55ec4828080b9fba2499cbb6ebacdbff08d3b4 `
  --output-file (Join-Path $data 'source_event_direct_verification.json')
```

The first mode is metadata/input preflight only. The second does the one-source
FIT parity confirmation; it does not run the event search or claim a new
quality gain. When a source report is supplied, the CLI verifies that report's
hash, selected vector and stored per-layout hard metrics, then compares the
direct output with those historical metrics. Its output file must be new and
outside this repository.

## Prospective search API and evidence boundary

`source_event_search.py` exposes the event proposal generator, incumbent
preserver, per-candidate float32 hard auditor, hard-only ranker, source/input
identity helpers, and a marker function with the distinct prefix
`source_event_attempt_`. A prospective runner must freeze a plan containing
the code, data, layout and source-vector hashes before it prepares optics or
evaluates candidates. It must validate those pins and create the new marker
exclusively before scoring, then recheck the same hashes when it writes the
report. The marker is intentionally not created by `preflight` or
`direct-verify`.

The comparison helper `fair_quality_runtime_protocol` specifies shared FIT
inputs, the same preserved incumbents, the same hard evaluator and quality
gates, equal candidate-count and wall-clock budgets, method-isolated cache
scopes, setup and proposal-generation timing, paired run order, and the scope
of incomplete results. A published quality-time comparison needs complete paired observations for
the same frozen workload hash. The runner is implemented below, but this
document does not claim that its prospective event-versus-grid run has
completed. The prior 5,140-point grid timing is context only, not a speedup
baseline.

The hard gate in `score_candidate_set` requires no blank print, no regression
in original-FIT aggregate hard counts, and no nominal/worst-dose regression
plus a strict band reduction on the three new FIT layouts. A caller may add a
source-domain auditor for convex guards; every candidate's audit result is
reported and must pass to qualify. Only actually attempted hard-qualified
records enter the deterministic lexicographic ranking. Gates use aggregate
means; a mean can conceal a per-layout tradeoff, so reports retain every
layout's counts and should be read alongside the means.


## Prospective paired runner

`scripts/benchmark_source_events.py` has three explicit modes. `freeze` hashes
the parent coverage plan, prior segment plan/report, selected FIT weights,
source code, datasets, diagnostics, manifest, lineage artifacts, and all
frozen source vectors. Its output plan must be new and beside the prerequisite
manifest so the one-use marker is colocated. `preflight` rechecks pins using
metadata only. `run` rechecks pins, writes the exclusive event-attempt marker,
and only then invokes the FIT loader, builds optics, or scores. Calibration
remains closed; the final3 split is never indexed or evaluated.

The fixed workload is five paired seeds with three paired repeats (30 arms).
Each arm first scores the frozen reference, that seed's initial vector, and the
preserved FIT-qualified slot 4755 incumbent. Event and grid arms share these
incumbents, all seven FIT layouts, the canonical float32 hard evaluator, all
source guards, the frozen cache policy, and the same captured original-host
thread audit policy. Primary scoring uses one CPU thread. The runner then
re-evaluates the protected reference and best-known source, plus each otherwise
hard-qualified candidate, at the original host's `torch.get_num_threads()`
count. A protected-vector count mismatch aborts; a candidate mismatch rejects
that candidate. The original thread count is recorded in the report, and the
one-thread setting is restored after every audit. In the optional cached
objective, the primary cache stores only canonical hard counts; these golden
checks always call the uncached evaluator. Per-arm primary-score seconds cover
only one-thread cache misses, while golden-audit seconds are a subset of total
candidate-scoring/audit and arm-wall seconds. These phase fields overlap; do
not sum them as independent elapsed time.

Each arm has a 600-second wall budget and a 1,031-attempt cap including the
three distinct protected vectors; the grid has at most 1,028 additional
positions (four segments × 257 points). The cap is checked between candidates,
so an in-flight canonical score and golden audit can finish after 600 seconds.
Protected incumbent scoring, proposal generation, audits, and progress writes
count toward arm time. Shared FIT loading and basis preparation happen once
before the arms and are separately timed, so this is a paired quality-time
comparison and not a claim of end-to-end setup speedup.

On the event arm, the float64 affine model proposes crossings, endpoints, and
interval midpoints. If a segment produces more than 257 proposals, a
deterministic stratified subset of 257 is evaluated. This sampling is not an
exhaustive threshold partition and may miss narrow feasible intervals. Both
event and grid proposals are actually checked with canonical float32 metrics;
only hard-qualified, golden-audited candidates are ranked. The report separates
time to improve the preserved checkpoint rank from time to a Pareto hard-PV
improvement in new-FIT band, worst-dose L2, and nominal L2 counts. A rank gain
may come from soft terms or tie-breaks and is not itself evidence of additional
hard-pixel coverage. An incomplete arm or unattempted candidate is not evidence
that no qualifying point exists.

The default freeze policy is `off`, which preserves direct uncached primary
scoring. To measure the prior within-method, cross-seed reuse behavior, freeze
with `--cache-policy per-method-repeat-physical-f32`. This policy creates a
new objective identity (`prospective_source_event_vs_grid_quality_time_cached_physical_f32_v2`) and a new frozen-plan SHA256, so it consumes a distinct event marker. It creates one cache per method and paired repeat, shares that cache across the repeat's five seeds, and never shares entries across event/grid methods or repeats. Keys are SHA256 hashes of the exact 49 little-endian float32 weight bytes in the fixed pinned basis/target/physical context. Cached values contain only canonical hard-count metrics; float64 domain/nominal/critical guards and soft rank inputs are recalculated. Cache lookup, copies, and scoring count against the arm wall budget. Each arm and the complete run report lookups, hits, misses, entries, lookup/copy time, and actual primary hard-metric scoring time. Do not reuse an old plan or marker for this objective.

All per-candidate records are appended once to `candidate_audit.jsonl` with
method, seed, and repeat lineage. `progress.json` contains bounded arm
summaries, not copies of the full candidate journal. The report and journal are
published outside the source repository. The commands below use the canonical
Linux server layout; replace only `REPO` if the frozen checkout has a different
location. The local Windows artifact copy is not a runnable substitute because
the pinned plans contain server-side absolute input paths.

```bash
REPO=/home/daniel/work/source-feasible-starts-20261009
DATA=/home/daniel/experiments/robust-source-quality-20261005-766872
PYTHON=/home/daniel/venvs/litho/bin/python
cd "$REPO"

# Metadata-only freeze for the cache-enabled v2 objective. For the original
# uncached objective, omit --cache-policy and use a new output-plan filename.
"$PYTHON" -B scripts/benchmark_source_events.py --mode freeze \
  --cache-policy per-method-repeat-physical-f32 \
  --coverage-plan-file "$DATA/direct_pv_objective_candidate_plan.json" \
  --expected-coverage-plan-sha256 de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127 \
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44 \
  --segment-plan-file "$DATA/segment_sweep_candidate_plan.json" \
  --expected-segment-plan-sha256 5c38f94ff79b7f4e2738664b211d4b0c1c0692f95ecb1ab56548736e50b1170d \
  --segment-report-file "$DATA/segment_5fd31de_report.json" \
  --expected-segment-report-sha256 a66908b8a8b902bdcc1846a35c55ec4828080b9fba2499cbb6ebacdbff08d3b4 \
  --selected-weights-file "$DATA/segment_5fd31de_selected_fit.json" \
  --output-plan-file "$DATA/source_event_quality_time_cached_plan_v2.json"

# Copy the event-plan SHA256 printed by freeze into EVENT_PLAN_SHA256.
EVENT_PLAN="$DATA/source_event_quality_time_cached_plan_v2.json"
EVENT_PLAN_SHA256='<sha256 printed by freeze>'

# Metadata-only check. It consumes no marker and does not load FIT data.
"$PYTHON" -B scripts/benchmark_source_events.py --mode preflight \
  --event-plan-file "$EVENT_PLAN" \
  --expected-event-plan-sha256 "$EVENT_PLAN_SHA256" \
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44

# The real prospective run can take up to 30 arms × 600 seconds, plus shared
# setup and any in-flight canonical/golden audit overrun. Use a new external
# output directory; this consumes the distinct source-event attempt marker.
"$PYTHON" -B scripts/benchmark_source_events.py --mode run \
  --event-plan-file "$EVENT_PLAN" \
  --expected-event-plan-sha256 "$EVENT_PLAN_SHA256" \
  --expected-previous-sha256 984dd65885ffc66b4eff3a70b4b8a81707841a777e8dd33ede9fe356daa5eb44 \
  --output-root "$DATA/source-event-runs"
```

Exit code 3 means an arm hit a declared budget and the report preserves the
incomplete status and unattempted work; it is not a negative search result.
The implementation and these commands are prospective protocol only until a
new frozen event plan and complete paired report are produced and reviewed.
