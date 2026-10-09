# Completed source-segment diagnostic (2026-10-09)

## What was tested

The source-only diagnostic evaluated all 5,140 registered slots: five seed
groups, four reference-to-endpoint segments per group, and 257 fixed alphas
`k/256`. Masks, targets, the 49-weight radius-0.9 source support, unit flux,
optical settings and hard gates were preserved. No adaptive refinement,
calibration selection, neural training, mask changes, MRC or dynamic hotspot
weights were used.

The code and protocol are in [SOURCE_SEGMENT_SWEEP.md](SOURCE_SEGMENT_SWEEP.md).
The source commit is `5fd31de35d2890005d3687c391a68ef889bc6e79`.

## Completed outcome

All five groups completed. The run took **514.4497057229746 seconds**
(8 min 34 s), including **5.0062955300090834 seconds** of setup.
Screening used one process-local Torch CPU thread. All seven reference
hard metrics matched the original 24-thread configuration exactly; every
screen-qualified slot received an uncached full evaluation at 24 threads.

| Seed group | Qualified slots | Scoring/audit wall seconds |
| --- | ---: | ---: |
| 17 | 15 | 74.418180 |
| 29 | 15 | 40.271877 |
| 43 | 260 | 272.485649 |
| 71 | 15 | 45.334162 |
| 101 | 37 | 71.034240 |

There were **342 qualified ordered slots**, representing **312 distinct
float64 vectors** and **282 distinct float32 vectors**. These counts are not
independent samples: the groups reuse terminal source paths and the same FIT
layouts. There were 75 qualifying schema-7 terminal-segment slots and 267
qualifying feasible-initialization-segment slots. No schema-6 or schema-8
terminal-segment slot qualified.

Among qualifying slots, 331 had new-three mean PV-band 161 pixels and 11 had
160.666667 pixels. The common reference mean was 161.333333 pixels.
The physical cache reused 3,091 screening evaluations; every slot retained
its own float64 feasibility audit, objective, distances, rank and provenance.
The final report contains all 5,140 attempted records without omissions.

## Best FIT point and per-layout tradeoff

The frozen FIT rank selected **slot 4755, seed group 101**, on the segment
from the reference to the schema-7 terminal vector, at **alpha = 129/256 =
0.50390625**. Selection was performed after all groups completed, using FIT
metrics only.

- Float64 source SHA-256:
  `388ff3bd997b4cbe7543a9d0fce15e2c46c2854ee9659a623b30539235069e86`.
- Float32 source SHA-256:
  `aa24fb03e849007cb3530f0de31968d88ad9af8089fe1f1da78841bb77df7f46`.

| FIT group | Metric | Reference mean | Best point mean |
| --- | --- | ---: | ---: |
| Original four | PV-band | 234.500000 | 234.500000 |
| Original four | Nominal L2 | 0.000000 | 0.000000 |
| Original four | Worst-dose L2 | 150.250000 | 150.250000 |
| New three | PV-band | 161.333333 | 160.666667 |
| New three | Nominal L2 | 35.333333 | 35.333333 |
| New three | Worst-dose L2 | 93.333333 | 93.333333 |

The new-three aggregate PV-band count decreased from **484 to 482 pixels**:
a **0.413223%** reduction. Mean nominal and worst-dose fidelity did not
regress. Fidelity was not unchanged for every layout:

| New FIT layout | PV reference / candidate | Nominal L2 reference / candidate | Worst-dose L2 reference / candidate |
| --- | ---: | ---: | ---: |
| Finite ribbons | 205 / 203 | 41 / 40 | 118 / 118 |
| Asymmetric line ends | 130 / 130 | 16 / 17 | 73 / 73 |
| Irregular contacts | 149 / 149 | 49 / 49 | 89 / 89 |

Finite ribbons decreased PV-band by two pixels and nominal error by one;
line ends increased nominal error by one. Thus the mean-fidelity non-regression gate passed
through a per-layout tradeoff. All float64 simplex, full original nominal
polytope, augmented guard-domain, float32 critical-guard and no-blank audits
passed for the selected point.

The **seed-43 feasible initialization itself**, before optimizer updates,
also passed the FIT gates: new-three PV-band 161, nominal L2 35.333333 and
worst-dose L2 93.333333. The earlier optimizer reports did not score this
initialization as a selectable hard-metric checkpoint. This is a concrete
reason to register initial hard metrics and retain the best hard-qualified
incumbent in the next optimizer protocol.

## What this establishes

The completed finite grid contains source vectors that pass the current
development FIT gates. The failure of the earlier smooth-optimizer checkpoints
did not establish that these gates were infeasible.

The best improvement is small and occurs on FIT layouts already used during
development. It does **not** establish improved hotspot detection, independent
generalization, focus-dose process-window robustness, official LithoBench
performance, or publication readiness. The five groups and hundreds of slots
must not be presented as independent statistical replication.

Calibration stayed **closed by design**, even though every group contained
passing points. The final three layouts were **never indexed or evaluated**.
A new frozen protocol is required before further optimization or confirmation;
this consumed diagnostic must not be rerun.

The next source-only method should score initialization, preserve a
hard-qualified incumbent, and register intermediate line-search candidates.
An improvement claim additionally needs a matched baseline and layouts that
have not been used to revise or choose the method. These next steps are not
implemented in this diagnostic.

## Artifacts and verification

The immutable server report is:
`/home/daniel/experiments/robust-source-quality-20261005-766872/segment-sweep-runs/segment-sweep-20261009T224641Z_9ba4bc75/segment_sweep_report.json`.

The full report and selected FIT weights were archived outside Git.
A root audit verified every slot's grid order, vector bytes and hashes,
qualified golden audit, own metric vector, ranking and selected global FIT
minimum. Remote and local artifact hashes matched:

| Artifact | SHA-256 |
| --- | --- |
| Report | `a66908b8a8b902bdcc1846a35c55ec4828080b9fba2499cbb6ebacdbff08d3b4` |
| Consumed plan | `5c38f94ff79b7f4e2738664b211d4b0c1c0692f95ecb1ab56548736e50b1170d` |
| Source bundle | `27a38e09fb3e843a813e751f3770151180e8538ebca177b5ca9fb2ac2c50865c` |
| Run log | `1e3461ce8dbc1871ae4323cdd52e7a4427bad9d21f74e3c765a2e30160bfc2c2` |
| Preflight | `b3ebfb9e2ae36a1b33879f68ee00c1bdc0d1221cd03020329b1f621d5305d733` |
| Server tests | `28c08e848301d324074fc985ecc132081e1dfacc8cac6d57ac941a24f06c5500` |
| Consumed marker | `4c8ac5a08b703eb5cd861e32e2b9b03e6ead8d75da35b699e0b57e6ab3daf4e9` |

The local unique harness passed **201/201 tests**, zero skips, in 88.930 s.
The server passed the same **201/201 tests** in 13.533 s.
Muse Spark 1.3 xhigh completed code and manuscript reviews. Grok 4.7 xhigh
fast reached its 10-minute limit without a final code review; partial stdout
was preserved. The mandatory exact Claude Opus 5.5 High review remains
pending because its account quota is unavailable until November 3, 2026.
The Luna draft was completed by the primary agent under an explicit user
exception after Luna hit its usage limit. No missing review is represented
as completed.

The open manuscript was edited in place. The built-in LaTeX compiler failed
during Windows helper preparation, before providing TeX diagnostics; the
revised PDF remains unvalidated.
