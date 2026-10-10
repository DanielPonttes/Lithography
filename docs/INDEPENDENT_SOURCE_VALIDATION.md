# Independent fixed-source GLP validation

This evaluator scores three frozen 49-point source vectors on the supplied
`StdMetal` and `StdContact` GLP tiles (reported as `StdMetal271` and
`StdContact165`): the event-selected candidate, the cached-plan reference, and
its cached best-known source. It does not optimize sources or select layouts
by their measured result.

The dataset contributes 271 metal and 165 contact GLPs. Within each family,
identical raster targets collapse to the lexicographically first tile, with
all aliases kept in the protocol. Tiles with identical rasters join their
cell-group bootstrap units, including matches across families. The filename
cell group is the stem before `__`, with a trailing `_X` plus digits removed.
These are supplied benchmark tiles; upstream tile curation may already have
changed or omitted source geometry.

## Frozen raster and model

GLP input must declare `EQUIV 1 1000 MICRON` and contain integer-coordinate,
Manhattan `PGON` solids. Coordinates are interpreted as nanometers. Each tile
is rasterized on a 1024×1024 canvas at 4 nm per pixel. The bounding box is
translated to the canvas center using a 4 nm multiple. The rasterizer samples
pixel centers with half-open scanline edges and unions polygon interiors. It
does not scale, crop, or clip geometry. Target masks, GLP file hashes, raster
hashes, vector hashes, source-code hashes, and dataset commit are frozen before
scoring.

The fixed scalar Abbe model uses NA 1.35, 193 nm wavelength, 9×9 source points
inside sigma 0.9 with sigma inner 0.3, zero focus, doses 0.98/1/1.02, threshold
0.225, and steepness 50. Sources are supplied directly as float32 weights;
they are not renormalized. For each layout, the evaluator prepares the
source-independent GPU basis once, transfers its float32 intensities to CPU
once, and contracts each fixed source vector on one CPU thread. This weighted
CPU1 result is canonical. Per-source GPU direct and GPU weighted-basis results
are independent parity checks. TF32 is explicitly disabled for CUDA matmul
and cuDNN; the frozen protocol records the compute policy, and the run records
the observed PyTorch/CUDA environment. Aerial images use rtol 1e-5 and atol
1e-6. Hard masks use the same sigmoid `>= 0.5` decision at each dose. Any
numerical or hard-mask parity mismatch remains in the output and makes
validation fail; no layout is dropped or source adjusted.

The primary layout scores are PVband pixels divided by target-positive pixels,
nominal XOR mismatch pixels divided by target-positive pixels, and worst-dose
XOR mismatch pixels divided by target-positive pixels. The spatial mismatch
rates catch equal-area prints that are displaced from the target. These scores
are averaged within each pinned cell-group component and then equally across
groups. Absolute printed-area discrepancies are retained as secondary
diagnostics and do not satisfy the fidelity gate. The report includes paired
candidate-minus-reference and candidate-minus-best-known group deltas, family
and pooled summaries, and a 10,000-draw percentile bootstrap interval using
seed 20261010. Group intervals are descriptive and do not establish
independent sampling.

The 512×512 padding sensitivity check reuses the first lexicographic supplied
tile in each family and the same three sources. It is diagnostic only and
cannot change the 1024×1024 primary result. This adapted scalar model and these
GLP tiles do not establish official SOCS, EPE, hotspot, full-mask, or production
generalization results. The cached FIT hashes are logged as excluded lineage;
the evaluator does not load FIT data and does not assess any earlier human
exposure to these geometries.

## Run protocol

Write all output to a new directory outside the code repository and source
dataset. Freeze parses all 436 GLPs and writes the immutable protocol and
packed target snapshots without importing the optical model:

```powershell
python scripts/evaluate_independent_sources.py freeze `
  --metal-dir <lithobench>/benchmark/StdMetal `
  --contact-dir <lithobench>/benchmark/StdContact `
  --output-dir <external-run-directory> `
  --cached-plan <pinned-event-cached-candidate-plan.json> `
  --event-candidate docs/evidence/source-events-20261010/event_uncached_selected_candidate.json `
  --event-report docs/evidence/source-events-20261010/event_uncached_50e2d8f_benchmark_report.json `
  --upstream-head 9c74e82218e377eaf6d02d113fc1ce6e36c92aa6
```

The command prints the protocol SHA-256. Validate that exact hash before the
consumed marker is created:

```powershell
python scripts/evaluate_independent_sources.py preflight `
  --protocol <external-run-directory>/protocol.json `
  --expected-protocol-sha256 <printed-sha256>
```

Then score once, using a new run directory for the unique consumed marker and
progress files:

```powershell
python scripts/evaluate_independent_sources.py run `
  --protocol <external-run-directory>/protocol.json `
  --expected-protocol-sha256 <printed-sha256> `
  --run-dir <new-external-run-directory> `
  --device cuda
```

The selected event JSON and uncached report are pinned to SHA-256
`60d54f2887dff4c5701ea6a2e1c289cc69ae662cbd7c7aa40d8a779cf52d95c2` and
`04b6e03caaac23cd21ef07f287fd02d493650d098051c0b8f665e528b64b14df`.
Existing markers and output artifacts are never overwritten. Runtime, input,
or memory failures leave a durable `partial` result and cannot pass the claim
gate. A completed numerical parity mismatch retains all measurements while
failing validation. The primary claim gate also requires a PVband group
bootstrap upper bound below zero, nonincreasing group-mean nominal and
worst-dose spatial mismatch fractions versus the reference, and no blank
candidate layout. Nonincreasing printed-area discrepancy alone is not enough
to pass the fidelity gate.
