# Shift Equivariance of NMS-Free Real-Time Detectors

**What does end-to-end inference cost in stability?**

YOLO26 ships NMS-free one-to-one inference as the default edge deployment path. This repo
measures what that costs when the input moves by a single pixel.

The question is answerable without confounds because the YOLO26 and YOLOv10 dual heads expose
**both** inference paths on **one released checkpoint** — backbone, neck and weights are
byte-identical across the head switch, verified here by state-dict fingerprint — so **the
aggregation head is the only variable**. Existing anti-aliasing results in detection compare
differently *trained* networks; this comparison does not. To our knowledge this is the first
shift-equivariance study of one-to-one NMS-free detectors.

No training is involved: everything is inference over public pretrained checkpoints, about
1.5 GPU-hours end to end.

| | |
|---|---|
| [**`RESULTS.md`**](RESULTS.md) | every number, how it was measured, and the runs that went against us |
| [**`PROTOCOL.md`**](PROTOCOL.md) | the frozen pre-registration — read this before the results |

---

## What we found

One YOLO26-N checkpoint, full COCO val (5000 images), one-pixel shifts:

| | one-to-one (NMS-free) | one-to-many + NMS | |
|---|---|---|---|
| shift envelope ΔAP | **3.76** | 3.25 | **+15.4%** worse |
| objects lost at some shift | **46.5%** | 28.8% | **+61%** worse |
| box jitter (px) | 1.75 | 1.85 | *better* |

1. **The NMS-free head is less shift-equivariant**, and its instability is in *whether* an
   object is reported rather than *where* it is put — its boxes are if anything steadier.
   Replicates on YOLOv10-N, where the gap is larger still (+26.8%).
2. **The instability is grid-locked, not noise.** 41–89% of the spectral power of AP(δ) lands
   on the network's downsampling periods {2, 4, 8, 16, 32} against 16% expected by chance —
   on **69 of 70 curves across five architectures**, RT-DETR included. Even-pixel shifts beat
   odd ones: the stride-2 stem's signature.
3. **The NMS-free head cannot buy its stability back.** Shift-TTA returns +0.86 AP, but only
   through a merge that reinstates NMS. The merge that stays NMS-free loses 26 AP — merging
   across shifts *is* a duplicate-removal problem.

One pre-registered instrument disagrees: the uniform-shift dispersion test, which we had named
in advance as *the* replication test, finds no gap on YOLOv10-N. It is reported rather than
dropped quietly — see [RESULTS.md](RESULTS.md#9b-the-head-gap-replicates--on-two-instruments-out-of-three).

---

## Quickstart

Requires a CUDA GPU (developed on an A10, 24 GB) and ~5 GB of disk for COCO val.

```bash
git clone https://github.com/amircohenwork/Shift-Equivariance-of-NMS-Free-Real-Time-Detectors.git
cd Shift-Equivariance-of-NMS-Free-Real-Time-Detectors

python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt

make setup     # COCO val2017 images + labels + instances_val2017.json (~2.5 GB)
make test      # unit tests -- no GPU, no dataset, no weights needed, ~10 s
```

Point Ultralytics at the dataset once (the setup script prints the exact command):

```bash
yolo settings datasets_dir="$HOME/datasets"
```

### Reproduce everything

```bash
make phase0        # Gate A + Gate B. Run first; nothing downstream is trustworthy if it fails.
make cache-oracle  # inference for the M=1 grid, both heads, full val   (~2 min)
make oracle        # greedy AP_best / AP_worst, both heads              (~45 min)
make sweep         # uniform-shift curve AP(delta) + flicker/jitter/SI  (~30 min)
make sweep-dense   # step-1 control that validates the step-2 shortcut  (~5 min)
make tta           # shift test-time augmentation                       (~7 min)
make report        # tables (results/REPORT.md) + figures
make fft           # 1-D sweep at 1-px resolution, spectral test, head-gap gate (~15 min)
```


The oracle is the expensive stage and its cost depends on your machine. Check the budget
before committing to it:

```bash
make oracle-budget   # times one AP evaluation and extrapolates K x (M+1)^2 x N
```

### Additional axes

```bash
# genuine small-object oracle. --tag is NOT optional: without it the run overwrites the
# committed all-area assignment .npy files that the published dAP rests on.
python scripts/04_oracle.py --run-id yolo26n_e2e_m1s1_full_pad114 --area small --tag small

# padding ablation: does removing the constant-border position cue remove the periodicity?
# 114 gray (protocol) / 0 black / -1 replicate / -2 reflect
make pad-ablation

# Manfredi & Wang Table 2: the max-shift ladder M in {1,3,7,15} on 100 images
make m-ladder

# numerics control -- the floor for the instance-level metrics
python scripts/08_numerics_control.py --subset 1000

# cross-model check: Gate A first, then the sweeps, then the comparison (~5.5 h on an A10)
make phase0-cross
make crosscheck-sweeps
make crosscheck
```

## Script map

| script | what it does |
|---|---|
| `00_setup_coco.sh` | COCO val2017 only (skips the 19 GB train split Ultralytics would fetch) |
| `01_reproduce_published.py` | **Gate A** — reproduce published AP for both heads |
| `01b_maxdet_ladder.py` | is `max_det` a neutral protocol choice? (yes) |
| `01c_topk_probe.py` | is the exported TensorRT top-k the same function? (yes) |
| `01d_protocol_probe.py` | `rect=True` vs `rect=False` — identifies the published protocol |
| `02_harness_gates.py` | **Gate B** — preprocessing / de-shift / determinism / evaluator / no-fusion |
| `02b_identity_check.py` | whole pipeline vs Ultralytics on the full val set |
| `02c_fusion_delta.py` | how much AP does Conv+BN folding move? |
| `02d_detection_diff.py` | box-by-box diff against Ultralytics' `predictions.json` |
| `02e_numerics_probe.py` | TF32 / cudnn.benchmark / fusion sweep |
| `02f_ap_noise_floor.py` | how much does AP move under score perturbation? |
| `03_build_cache.py` | run inference over a shift grid, persist de-shifted detections |
| `04_oracle.py` | greedy ΔAP (Manfredi & Wang Algorithm 1) |
| `05_sweep.py` | uniform-shift curve + flicker/jitter/Stability Index |
| `06_report.py` | tables and figures |
| `07_shift_tta.py` | shift test-time augmentation, three merge strategies |
| `08_numerics_control.py` | instance-metric noise floor |
| `09_cache_benefit.py` | what the preprocessing cache is worth (measured: 1.01×) |
| `10_sweep1d_fft.py` | 1-D sweep at 1-px resolution + spectral test for grid-locking |
| `11_head_gap.py` | the gate: grid-locked? measurable head gap? + FFT figure |
| `12_crosscheck.py` | is grid-locking universal? does the head gap replicate on YOLOv10-N? |
| `13_pad_ablation.py` | aliasing or border-reading? padding value vs the spectrum |
| `14_final_report.py` | the small-object oracle and the max-shift ladder, read against PROTOCOL §12 |
| `15_paper_figures.py` | spare figures for the report, drawn straight from `results/` |

---

## Layout

```
shifteq/
  modelzoo.py   loading with fuse guards; head switching without touching a weight
  data.py       COCO GT parsing; the one-shot letterbox cache
  shift.py      canvas embedding and the exact inverse map
  infer.py      batched inference over the grid; Ultralytics NMS verbatim
  detstore.py   on-disk detection cache + run manifest
  cocolite.py   COCO AP with cacheable per-image matching (validated to 1e-16)
  oracle.py     union table, accumulators, greedy Algorithm 1
  sweep.py      uniform-shift curve, resumable
  stability.py  flicker/jitter, as parallel-friendly per-shift partials
  stabindex.py  2-D Stability Index (confidence / location / extent decomposition)
  tta.py        shift test-time augmentation, three merge strategies
  pipeline.py   worker pool: GPU inference for δ_{k+1} overlaps CPU eval of δ_k
  spectrum.py   power spectrum of AP(δ), permutation nulls, parity statistic
scripts/        00 setup .. 15 paper figures
tests/          run on a laptop with no GPU, dataset, or weights
ops/            batch runners and the watchdog used for the long GPU sessions
figures/        the three summary figures, drawn from results/
results/        committed outputs of every run (the evidence; RESULTS.md is the write-up)
PROTOCOL.md     the frozen pre-registration -- read this first
```

Every run writes a manifest recording the weights SHA-256, a state-dict fingerprint,
`fused=False` plus the evidence for it, and the full postprocessing configuration.

---

## Pre-registration

[`PROTOCOL.md`](PROTOCOL.md) says exactly what we would measure and what result would count
as a yes or a no. It was written **before** anything was run, and frozen so it cannot be quietly edited afterwards.

## Reading the numbers

Three constraints worth knowing before quoting anything — the full list is in
[RESULTS.md](RESULTS.md#caveats):

* **ΔAP, flicker, dropout and class-flip are comparable only at equal `M` and `N`.** Each is
  an existence claim over a grid of shifts, so all four grow mechanically with the number of
  shifts sampled. `results/stability.csv` shows the same model/head at 48.5% flicker on 1024
  shifts and 39.2% on 256.
* **The sweep numbers use 1000 images at step 2** (the oracle uses all 5000). Step-2 grids
  underestimate the AP range, so read them as lower bounds.
* **`summary_at_assignment` is not an oracle** for AP_s/m/l — it reports those at the
  assignment that maximised overall AP50. The genuine small-object oracle is a separate run.

## License

Copyright (C) 2026 Technion — Israel Institute of Technology

This program is free software: you can redistribute it and/or modify it under the terms of
the **GNU Affero General Public License** as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. The full text is in
[`LICENSE`](LICENSE).

It is distributed in the hope that it will be useful, but **without any warranty**; without
even the implied warranty of merchantability or fitness for a particular purpose.

This repo imports
[Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0, in four core
modules and most of `scripts/`. AGPL is copyleft: anything distributed on top of it must
carry the same licence, so a permissive licence here would be granting rights we do not have
to give. Every other dependency — PyTorch, NumPy, OpenCV, matplotlib — is permissive and does
not constrain the choice.
