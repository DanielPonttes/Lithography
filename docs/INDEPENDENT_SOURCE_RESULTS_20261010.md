# Independent fixed-source transfer evaluation (2026-10-10)

## Scope and frozen method

This is a post-selection, fixed-vector transfer check on the supplied LithoBench
`StdMetal` and `StdContact` GLP collections. It is not an optimization run: the
same three frozen 49-weight vectors were applied to every layout: the
schema-7 event candidate, the cached-plan reference, and the cached best-known
source. No source was changed, ranked, or selected using these measurements.

## Original Neural-ILT results reported in the supplied PDF

The supplied anonymous five-page review copy, *Physics-Informed Hotspot Detection via Neural-ILT Training: A PV-Band Based Approach on LithoBench* (pp. 3–4), reports its own pretrained Neural-ILT `Init` versus PV-band-aware `Fine-tuned` results, averaged over its stated 10-case MetalSet test split. These values are transcribed from the PDF and were not reproduced here:

| Original-paper metric | Init | PV-band-aware fine-tuned | Change reported in PDF |
| --- | ---: | ---: | ---: |
| L2 pattern error | 36,688.4 | 27,492.4 | −25.1% |
| PV-band area | 42,659.3 | 42,864.5 | +0.5% |
| EPE count | 7.3 | 2.0 | −72.6% |
| Mask shots | 472.2 | 513.2 | +8.6% |

The original setup describes Neural-ILT mask optimization on MetalSet with LithoBench Quasar illumination, resist threshold 0.225 and steepness parameter $\alpha=85$; its process corners jointly vary dose by $\pm2\%$ and defocus by $\pm25$ nm. The current transfer check instead applies frozen source vectors to fixed StdMetal/StdContact masks with the adapted scalar-Abbe evaluator ($\alpha=50$, zero focus), and reports group-normalized PV-band and binary XOR errors. It does not train or evaluate the original Neural-ILT mask model, report EPE or mask shots, or match the original process window. The original PDF gives no timing measurements, so the current paired 9.6-fold grid/event search-time ratio is not an end-to-end Neural-ILT speedup or a direct head-to-head comparison. The PDF's introduction claims four LithoBench subsets, while its setup and quantitative table describe MetalSet and 10 test cases; its heatmap-statistics section separately refers to four MetalSet cases, so generalization scope is internally inconsistent.

SHA-256 of the supplied PDF copy: `b912bfd1a37a497e46f41a4536c245764c6cb1b7c2fec0fbd6a44fdd8501a085`.

## Comparison baselines

The quality plots compare the event candidate with two internal frozen vectors: the cached-plan reference and the cached best-known source. The runtime plot compares event proposals with the paired uniform $k/256$ segment grid on the same frozen workload. These are within-study comparators; they do not compare against the original published NeuralILT baseline or an end-to-end Tensor-of-Light pipeline. The GLP transfer check uses an adapted scalar-Abbe model and is not official SOCS scoring.

The input inventory contained 271 StdMetal and 165 StdContact files. Exact
raster aliases were collapsed to the lexicographically first representative,
leaving 220 unique metal and 165 unique contact masks (385 total). Scores use
pinned cell-group components: 39 metal, 19 contact, and 41 pooled groups.
Pooled components can link family groups; they are not independent samples.
The unit of summary and bootstrap is the cell-group component, equally
weighted, with 10,000 percentile-bootstrap draws and seed 20261010.

Masks were scanline-rasterized without scaling or clipping onto a 1024×1024
canvas at 4 nm/pixel. The evaluation used the fixed scalar Abbe model with 49 allowed source
points on a 9×9 grid inside the sigma≤0.9 pupil disk, NA 1.35, wavelength
193 nm, zero focus, doses 0.98/1.00/1.02, threshold 0.225, and steepness 50.
The sigma-inner value 0.3 defines an initialization prior only. This is an
adapted scalar model on the supplied GLPs, not official SOCS, EPE, hotspot, or
full-mask scoring. For each layout, one GPU-prepared basis supplied the
canonical one-thread CPU weighted result; direct-GPU and weighted-GPU routes
were checked independently. Aerial tolerance was rtol 1e-5 and atol 1e-6;
hard masks were compared exactly. TF32 was disabled.

The primary metrics below are equal means over pinned group means of each
layout's PV-band, nominal XOR mismatch, and worst-dose XOR mismatch fractions,
each divided by target-positive pixels. Table values are `reference →
candidate`; lower is better.

| Scope | Groups / unique layouts | PV-band fraction | Nominal mismatch fraction | Worst-dose mismatch fraction |
| --- | ---: | ---: | ---: | ---: |
| Pooled | 41 / 385 | 0.03241619 → 0.03239917 | 0.36485624 → 0.36484352 | 0.37674230 → 0.37672758 |
| StdMetal | 39 / 220 | 0.02022984 → 0.02022498 | 0.18982900 → 0.18984598 | 0.19444469 → 0.19445163 |
| StdContact | 19 / 165 | 0.06690953 → 0.06685180 | 0.86575970 → 0.86565762 | 0.89881725 → 0.89872973 |

The pooled PV-band point estimate improved by 0.0525% relative to the
reference. Its paired group delta was −1.7017×10⁻⁵, with a 95% interval
[−4.8912×10⁻⁵, +1.4937×10⁻⁵] and 20 wins, 2 ties, and 19 losses. The interval
crosses zero, so the pooled result does not establish superiority. StdContact
had a favorable PV-band delta (−5.7731×10⁻⁵; interval
[−9.7660×10⁻⁵, −1.9401×10⁻⁵]), while StdMetal's interval crossed zero. The
StdMetal nominal mismatch fraction worsened by 1.6980×10⁻⁵ (95% interval
[2.1671×10⁻⁶, 3.3006×10⁻⁵]). Candidate and best-known were nearly tied:
the candidate's pooled PV-band fraction was higher by 1.5495×10⁻⁶, with an
interval crossing zero. These small relative PV-band shifts coexist with severe absolute StdContact spatial error: candidate equal-group nominal and worst-dose mismatch fractions were 0.86566 and 0.89873, or 86.6% and 89.9% of target-positive pixels.

| Scope | Target-positive pixels | Source | PV-band pixels | Nominal XOR pixels | Worst-dose XOR pixels |
| --- | ---: | --- | ---: | ---: | ---: |
| Pooled | 2,925,771 | Reference | 97,712 | 1,122,813 | 1,161,631 |
|  |  | Candidate | 97,689 | 1,122,763 | 1,161,578 |
|  |  | Best-known | 97,691 | 1,122,763 | 1,161,580 |
| StdMetal | 2,096,000 | Reference | 42,190 | 404,226 | 415,361 |
|  |  | Candidate | 42,197 | 404,245 | 415,374 |
|  |  | Best-known | 42,198 | 404,245 | 415,375 |
| StdContact | 829,771 | Reference | 55,522 | 718,587 | 746,270 |
|  |  | Candidate | 55,492 | 718,518 | 746,204 |
|  |  | Best-known | 55,493 | 718,518 | 746,205 |

## Parity, sensitivity, and claim gate

All 385 layouts were scored for all three sources: 1,155 scored records,
zero runtime/error records, and no blank candidate layout. Four source/layout
records failed exact hard-mask parity, each by one pixel at one dose:

- `StdContact165:AOI21_X4__1_0.glp`, event candidate, dose 1.00;
- `StdMetal271:CLKGATE_X4__1_0.glp`, reference, dose 0.98;
- `StdMetal271:OAI21_X2__0_0.glp`, best-known, dose 1.02; and
- `StdMetal271:TLAT_X1__2_0.glp`, event candidate, dose 1.02.

Every aerial comparison in those rows remained within the recorded allclose
tolerance; maximum absolute differences were between 5.96×10⁻⁸ and
3.58×10⁻⁷. The hard-mask differences were retained in the output and caused
the exact parity validation to fail. Numerical closeness of aerials does not
make the thresholded masks identical.

The 512×512 padding check covered one representative tile per family
(`AND2_X4__0_0.glp`) for all three sources, six records total. PV-band fraction
shifted by +9.5397×10⁻⁴ (candidate/best-known) and +7.1548×10⁻⁴ (reference) on
StdMetal, and by −8.7796×10⁻⁴ (candidate/best-known) and −1.1706×10⁻³
(reference) on StdContact. The reference StdContact check had one hard-mask
parity failure at dose 1.02. This two-layout sensitivity sample is diagnostic
only and does not alter the primary 1024×1024 scores.

The registered primary claim gate is **false**. Although mean nominal and
worst-dose spatial mismatch did not regress in the pooled summary and no
candidate layout was blank, the pooled PV-band confidence interval did not
lie below zero and exact numerical/hard-mask parity did not pass. Therefore
these measurements do not support a source-quality superiority claim. They
also do not show that the candidate is broadly better than the cached
best-known vector. There is no external-runtime comparison in this run.

## Visual summaries

The deterministic plotting script reads only the frozen quality result and the two captured timing reports. It performs no scoring, optimization, or new experiment. Each chart is exported as PNG for inspection and PDF for print-quality reuse.

![Candidate minus frozen-reference and best-known deltas with descriptive group-bootstrap intervals](../paper/figures/transfer-quality-deltas.png)

*Figure 1. The forest plot reports candidate-minus-comparator deltas for PV-band, nominal mismatch, and worst-dose mismatch in pooled, StdMetal, and StdContact scopes. Values are percentage points of normalized rates; negative values indicate lower error. Pooled intervals use linked group components and are descriptive.* [PDF](../paper/figures/transfer-quality-deltas.pdf)

![Absolute nominal and worst-dose mismatch by source and layout family](../paper/figures/transfer-absolute-errors.png)

*Figure 2. Absolute error fractions retain the large StdContact errors visible: candidate equal-group means are 86.6% nominal and 89.9% worst-dose.* [PDF](../paper/figures/transfer-absolute-errors.pdf)

![Matched seed-repeat runtime ratios for event and uniform-grid source search](../paper/figures/paired-runtime-ratios.png)

*Figure 3. Each point is a paired uniform-grid/event arm-time ratio for the fixed seed and repeat. The medians are 9.582 for uncached v1 and 9.598 for cache-enabled v2; these describe this CPU search workload and are not hardware speedups.* [PDF](../paper/figures/paired-runtime-ratios.pdf)

## Reproducibility and chronology

The immutable evidence is copied byte-for-byte under
[`docs/evidence/independent-source-20261010/`](evidence/independent-source-20261010/). The top-level run summary is available at [`run-29f83e4/result.json`](evidence/independent-source-20261010/run-29f83e4/result.json).
The protocol ID is `independent-fixed-source-glp-20261010T160743Z`; the run
completed on 2026-10-10 in 108.655 seconds with status `complete`, 1,155 of
1,155 expected records, and no fatal error. Recorded environment: NVIDIA
GeForce RTX 5090, PyTorch 2.11.0+cu130, CUDA 13.0, one PyTorch CPU thread,
CUDA matmul TF32 disabled, and cuDNN TF32 disabled. The retained run metadata
do not record a Python version.

| Pinned artifact | SHA-256 |
| --- | --- |
| Protocol | `c9fccd1bc8a27c10dbf5a5b437d36de50e0ec1532b00c28eb3f3190cbe327ebb` |
| Result | `292dba0647c159e5015db6d81ab7cc431f63880f9eeb165300e09d803522eab1` |
| Layout journal (1,155 lines) | `155719356138ebd1d347b4342162c3395d17aa2205c463c2e04392e7cadc3674` |
| Padding journal (6 lines) | `6e5558de0d48e5a192b13ffc7cac5ce71817b112f132fe48e321d1a75ac2c650` |
| Progress | `80ea52efbecb0637eeec4e420008f5e26fe8c0a1667e5baf064b4404a2534564` |
| Consumed marker | `676fb0cedd537ebef3bcccee5db63085f5f7067659e3bf45fe4691858b9060b3` |

The evaluator source was pinned to local repository revision
`29f83e4d155b8792c8aebf1ebef5cc0c46505746` (script SHA-256
`4bff953b53f09602d59fe3029c48528271fbbbd85365994e7d0f927faff3b5d8`),
`light_source.py` SHA-256
`36d64aa579625694c3db41c921d7b3475896bc4569f5958c10ec4889f800765f`, and
upstream LithoBench revision `9c74e82218e377eaf6d02d113fc1ce6e36c92aa6`.
The float32 weight-vector hashes are event candidate
`901b22f41ef587357dab92e5ef80f557a9d0bdd22fc86560944d5b6dd7f58b41`,
reference `98ea7a216ca3ad6149abe7a8a5482da69990a441b855648ac67bb88268721d2e`,
and best-known
`aa24fb03e849007cb3530f0de31968d88ad9af8089fe1f1da78841bb77df7f46`.

Earlier freezes were rejected before source scoring when the full upstream
revision/header and bounding checks were not yet correct. A consumed attempt
(`run-65c6d23`) then failed on the first layout before any optical
measurement: the standalone CLI could not import `light_source`, and its
pretty-printed JSON writer was incompatible with the one-record-per-line
journal reader. That attempt is archived and contains no scored result. The
successful run used the corrected, separately pinned code and fresh protocol;
there was no source tuning or rerun after examining its outcomes.

As of 2026-10-10, the required Claude Opus 5.5 High review remained pending
until the exact-model quota reset on 2026-11-03. A Muse review is supplementary
and does not replace that requested review.
