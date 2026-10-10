# Completed source-event search results

## Scope

This note records two completed, separately pinned comparisons of bounded event proposals against a uniform source-segment grid: uncached v1 and cache-enabled v2. The benchmark used the original four FIT layouts plus three fixed new FIT layouts, seven layouts total. Masks and source-basis inputs were frozen. The 49 source weights were nonnegative, unit-flux, and source-only; focus was fixed at zero. The three tested doses were 0.98, 1.00, and 1.02, with hard printing obtained from the sigmoid resist at threshold 0.225 and steepness 50. Calibration remained closed, and final3 layouts were never indexed or evaluated.

Both benchmark reports declare status `complete`, record successful final source-pin rechecks, and record that each new attempt marker was consumed before FIT loading or optics preparation. Each plan and marker was distinct. Both comparisons are quality-time comparisons on the same five fixed seeds, not independent-group experiments or hardware speedup claims.

## Proposal and comparison protocol

For the source segment

\[
\mathbf{s}(\lambda)=(1-\lambda)\mathbf{s}_0+\lambda\mathbf{s}_1,\quad 0\leq\lambda\leq1,
\]

basis intensity is affine in \(\lambda\). At dose \(d\), ideal binary transitions occur at \(dI_{\mathbf{s}(\lambda)}(x,y)=0.225\). The event arm proposed segment endpoints, deduplicated float64 threshold crossings across the frozen layouts and doses, and midpoints between ordered event values. Its deterministic proposal cap was 257 per segment. All 60 segment instances across the three repeats were below the cap; the maximum recorded event proposal count on a segment was 33. This is a bounded proposal set, not exhaustive float32 transition enumeration: the search can miss narrow intervals or behavior that changes after float32 rounding.

The paired grid arm evaluated the same 20 fixed seed-segment paths at \(\lambda=k/256\), \(k=0,\ldots,256\). Both methods protected the same three sources in every seed arm: the frozen reference, that seed's feasible initialization, and the previously qualified best-known FIT vector (slot 4755, seed group 101). In uncached v1, both arms disabled physical score caching, used one primary CPU scoring thread, applied the same hard qualification and ranking procedure, and gave every hard-qualified candidate a fresh uncached golden audit at the original host setting of 24 CPU threads. Each arm had a 1,031-candidate cap and a 600-second wall-time ceiling. The five fixed seeds (17, 29, 43, 71, 101) were measured in three paired timing repeats, with arm order alternated, for 15 pairs and 30 arms total. These repeats reuse the same FIT layouts and are descriptive, not independent replicates.

Optical and basis setup was shared outside the per-arm wall-time measurement and took 4.284609 seconds in uncached v1 and 4.128725 seconds in cache-enabled v2. Per-arm elapsed time includes periodic report writes. Timings for proposal generation, scoring, golden audits, and protected incumbents describe phases or subsets inside the arm and should not be added together. Runtime reflects this pinned host and workload; it is not a GPU throughput benchmark.

An unrelated GPU workload shared the server during both protocol windows. Cache-enabled runner tests overlapped the final uncached timing pair by about 20 seconds. Those tests were not another comparison arm. This host-load overlap, the different protocol order, and shared hardware mean that the two sets of elapsed times do not isolate a causal cache effect. Candidate counts and the exact cross-protocol hard-metric parity do not depend on interpreting that runtime difference.

## Completed uncached v1 comparison

| Method | Evaluations across 15 arms | Per-arm evaluations | Median full-arm wall time | Per-arm wall-time range | Rank improvement | Hard-PV improvement beyond protected best |
|---|---:|---:|---:|---:|---:|---:|
| Event proposals | 1,200 | 76–88 | 6.643 s | 5.609–9.450 s | 15/15 | 0/15 |
| Uniform \(k/256\) grid | 15,465 | 1,031 each | 60.649 s | 54.248–126.747 s | 0/15 | 0/15 |

The median of the 15 paired grid-to-event ratios of full-arm wall time was 9.5816, with a range of 6.6664–18.8760. The grid evaluated about 12.89 times as many candidates in this finite workload. This is a comparison of two capped search procedures and their quality-time outcomes; it must not be restated as a 9.58x hardware speedup.

The ranking first respects the registered hard-metric ordering and then uses the frozen soft-objective and LP-distance tie-breaks. Event search improved that checkpoint rank in all 15 arms. In no arm did event or grid search improve hard PV-band beyond the protected best-known source. The rank result is therefore a tie-break improvement under the frozen protocol, not an additional hard-pixel gain. Each method retained the best-known source even when no newly scored point improved it.

## Second completed cache-enabled v2 comparison

The cache-enabled v2 run was a new frozen protocol with its own plan identity and attempt marker. It repeated the same five seeds, three paired timing rounds, source paths, protected incumbents, 1,031-candidate cap, and 600-second ceiling. The cache was scoped separately by search method and timing repeat, shared only across that repeat's five seeds, and never crossed between event/grid methods or repeats. Its key was the hash of the exact 49 little-endian float32 source bytes plus the pinned physical context. Cached payloads held canonical per-layout hard counts only. Float64 feasibility/domain audits, soft ranking fields, and every original-thread golden audit remained uncached.

| Search | Candidate evaluations | Cache hits / lookups | Hit rate | Median full-arm wall time | Per-arm wall-time range | Rank improvement | Hard-PV improvement beyond protected best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Event proposals | 1,200 | 888 / 1,200 | 74.000% | 6.221 s | 5.502–8.549 s | 15/15 | 0/15 |
| Uniform \(k/256\) grid | 15,465 | 9,318 / 15,465 | 60.252% | 58.751 s | 54.136–124.284 s | 0/15 | 0/15 |

The median paired grid-to-event full-arm wall-time ratio was 9.5982 (range 7.3029–18.3337). As in v1, this is a paired quality-time description of a finite workload, not a speedup claim. Cache-enabled and uncached medians must not be compared as a controlled estimate of cache benefit because of the shared host load and test overlap noted above.

Across all 30 method/seed/repeat arms, the cached and uncached runs had exactly the same best-qualified candidate identity, complete rank tuple, qualified hard-metric record, checkpoint-rank improvement identity, and hard-PV-improvement identity. The cache changed neither selected FIT candidate nor any recorded hard-count result. In aggregate, the cache recorded 10,206 hits from 16,665 lookups (61.242%).

## Direct confirmation of the selected event vector

The selected event-native vector was `seed-29:schema7_endpoint:000014`. Its 49-weight float64 and float32 hashes were:

- float64 vector: `36c66f0b29bdeced656a4734d906bb5cd667c26daa6808e89dd1131e5159f3c6`
- float32 vector: `901b22f41ef587357dab92e5ef80f557a9d0bdd22fc86560944d5b6dd7f58b41`

The direct simulator and weighted-basis aerial tensors were within the verification tolerance (relative tolerance \(10^{-5}\), absolute tolerance \(10^{-6}\)). Direct-GPU, weighted-basis-GPU, and canonical-CPU routes returned exactly matching per-layout hard counts for both candidate and reference on all seven layouts. The direct verification marked all registered aggregate hard gates as passed. No historical source report was supplied to that verification, so this confirms cross-route evaluator parity for the captured run, not parity with an earlier archived report.

| New FIT layout | PV-band, reference → candidate | Nominal \(L_2\), reference → candidate | Worst-dose \(L_2\), reference → candidate |
|---|---:|---:|---:|
| Finite ribbons | 205 → 203 | 41 → 40 | 118 → 118 |
| Asymmetric line ends | 130 → 130 | 16 → 17 | 73 → 73 |
| Irregular contacts | 149 → 149 | 49 → 49 | 89 → 89 |

Across the three new layouts, aggregate PV-band count changed from 484 to 482 (0.413%); mean nominal and worst-dose counts were unchanged at 35.333 and 93.333 pixels. The original-four means remained 234.5 PV-band, 0 nominal, and 150.25 worst-dose pixels. The aggregate nominal mean hides a per-layout tradeoff: finite ribbons improved by one nominal pixel and asymmetric line ends worsened by one. The reported gate is an aggregate gate; no claim of per-layout non-regression is made. The 482-pixel PV-band total also equals the already preserved best-known source, so it is not a new hard-PV improvement over that incumbent.

## Reproducibility identities

| Evidence | Identity |
|---|---|
| Uncached v1 report SHA-256 | `04b6e03caaac23cd21ef07f287fd02d493650d098051c0b8f665e528b64b14df` |
| Cache-enabled v2 report SHA-256 | `9ba0f2d772ba36e107e4bb01f807ac2f86b34a1e97d478d5d2ab01ad03bc141f` |
| Selected event-vector file SHA-256 | `60d54f2887dff4c5701ea6a2e1c289cc69ae662cbd7c7aa40d8a779cf52d95c2` |
| Direct-confirmation file SHA-256 | `534e2a680912e06274692b430aab9244363b1c71929823a9f943efcbcfdf7784` |
| Uncached v1 event plan SHA-256 | `854b81a18ebd36589a5b50f6316059bfd45fa502c767e6bbb6f148d76f71a7cf` |
| Cache-enabled v2 event plan SHA-256 | `777393f20ca3ae28f9b74a61eb4d0d5fc488280ed793cbeaa1baa5c2bd1227dc` |
| Coverage plan SHA-256 | `de435a135c44fad5346cbc2cc21985cd54042d4655b037d87df3bb4f132a1127` |
| Uncached v1 source snapshot | `50e2d8f1f56517c35d1e2c232abd6c15cbbfb3b6` |
| Cache-enabled v2 source snapshot | `690275d1f590104f4d9be77afed60246dde333aa` |
| Pinned source-identity manifest SHA-256 | `9ba8d523498598f68f7b69adf93f35c3be850a0da800c4a2cf9da5dd034a69f0` |
| Event-search module SHA-256 | `d2f4b9dd822a601b2eebc48a2f38220a231389189a67b3d9c5e74790c56fa3fa` |
| Benchmark runner SHA-256 | `a0e47b469c6e3d219347f6e7540a310d66abf6f401460fb01639daa1e916497c` |
| Cache-enabled v2 source-identity manifest SHA-256 | `6d266285460b66395aa69a52f8d810896da2b605684ed473130c19589598a84c` |

The captured environment was Linux x86-64, Python 3.12.3, PyTorch 2.11.0+cu130, CUDA 13.0, and an NVIDIA GeForce RTX 5090. Primary per-arm scoring used one CPU thread; the golden audit used the original 24-thread setting. `matmul_allow_tf32` was false and `cudnn_allow_tf32` was true. The report pins imported code and inputs, includes the 14 parent-lineage artifacts, and stores its source pin recheck. The direct-confirmation artifact itself has `source_report_provenance.verified=false` because no pinned source report was supplied.

## Interpretation limits

The result is restricted to these fixed FIT layouts, source paths, proposals, evaluator version, and timing protocol. The event set is finite and computed from ideal float64 crossings; it is not a complete search over float32 classifications, all source vectors, or all feasible source regions. Five seeds repeated three times do not provide independent layout-level replication. No process-window result, EPE analysis, hotspot labels or classification metric, official holdout, or generalization result is claimed. The prior known incumbent remained protected in both arms and retained the best measured hard PV-band count.

## Archived evidence

The following captured JSON files preserve the hashes listed above:

- [Uncached v1 benchmark report](evidence/source-events-20261010/event_uncached_50e2d8f_benchmark_report.json)
- [Cache-enabled v2 benchmark report](evidence/source-events-20261010/event_cached_690275d_benchmark_report.json)
- [Selected event source and expected hard metrics](evidence/source-events-20261010/event_uncached_selected_candidate.json)
- [Direct simulator confirmation](evidence/source-events-20261010/event_uncached_selected_direct_confirmation.json)

The pinned dataset and full candidate journals remain in the experiment archive; these four files are the captured result evidence, not a self-contained dataset release.
