# Frozen Experimental Protocol — v1.2.0

**Status: FROZEN.** v1.0.0 was frozen on 2026-08-15 and produced every yolo26n number in
`results/` and `RESULTS.md`. v1.1.0 (2026-08-17) added §10 and §11 — the analysis plans for
the cross-model check and the padding ablation. v1.2.0 (2026-08-19) **adds** §12, the two
remaining variations §9 has declared as intended since v1.0.0: the small-object oracle and
the max-shift ladder. **No v1.0.0 or v1.1.0 value was changed**, so every result already
published stands exactly as it is; superseded hashes are retained in
`protocol.PRIOR_HASHES` so older run manifests remain traceable.

This is a pre-registration. It exists so that a reader can tell, without running anything,
exactly what was measured — and so that we cannot quietly change a threshold after seeing a
result. The machine-readable twin is [`shifteq/protocol.py`](shifteq/protocol.py);
`tests/test_protocol_frozen.py` fails if the two drift apart, and every run manifest records
`protocol_hash()` so a results file can be traced to the protocol that produced it.

**Amendment policy.** Do not edit a frozen value in place. Bump `VERSION`, add a dated
changelog entry, and state in the paper which protocol version produced which table. A
value changed after seeing results is a different experiment and must be labelled as one.

Reproduce the current hash with:

```bash
python -m shifteq.protocol       # prints every value and the hash
```

---

## 1. Shift protocol

### 1.1 No cropping, ever — and how that differs from Manfredi & Wang

Manfredi & Wang (arXiv:2008.05787, Sec. 4 / Fig. 2), following Azulay & Weiss, embed each
image into a **larger black canvas** at offset δ rather than shifting-and-cropping. The
reason is specific to detection: cropping even a few pixels can remove part of an object
and change the ground truth, which confounds *"the detector moved its answer"* with *"the
input genuinely lost information"*. Objects at frame edges make this a live risk, not a
theoretical one.

**We preserve that property. We do not use their exact recipe.** The deviations are
deliberate and listed here so they can be challenged:

| | Manfredi & Wang | this study | why |
|---|---|---|---|
| what is embedded | the raw image | the **letterboxed** image | the network needs a fixed, stride-aligned input; letterboxing first is what makes δ=(0,0) byte-identical to the published val pipeline (gate B1) |
| fill value | black (0) | **gray (114)** | Ultralytics' letterbox constant. Changing it changes *two* things at once (fill and protocol-identity), so 114 is the default and 0 is available as an ablation via `--pad-value 0` |
| canvas size | image + M | `align(imgsz + M, stride)` = 672 | stride-aligned so the anchor grid never changes shape between shifts |

**The no-crop property still holds exactly.** The letterboxed tile is pasted into the canvas
as a rigid unit and is fully present at every δ. An object's entire local neighbourhood —
including the letterbox border beside it — translates with it. Only the tile's *absolute
position* within the canvas changes, which is precisely the variable under study.

**Known cost of the `grow` choice, measured not assumed.** The network sees 672 px rather
than the 640 px it was validated at. Gate B6 measures this at δ=(0,0) on a **500-image**
probe: +0.225 AP for the one-to-one head, +0.082 for the one-to-many head.

**On the full validation set the effect is much smaller**, and the 500-image figure for the
one-to-one head was mostly sampling noise:

| N | e2e | nms |
|---|---|---|
| 500 (gate B6 probe) | +0.225 | +0.082 |
| **5000 (full val)** | **+0.029** | **+0.068** |

Quote the full-val row. The canvas costs about **0.03–0.07 AP**, not a quarter of a point,
so the input-size objection to `grow` is weaker than the probe suggested — but it is still
non-zero, which is why `fit` exists and why B6 measures rather than asserts.

### 1.2 The two embedding modes

Both are no-crop. `grow` is frozen as the default; `fit` exists so the input-size objection
can be answered empirically rather than argued.

| mode | letterbox target (`tile`) | network input (`canvas`) | trade |
|---|---|---|---|
| **`grow`** (default) | `imgsz` = 640 | `align(640 + M, 32)` = 672 | full resolution, byte-identical preprocessing, non-validated input size |
| `fit` | `imgsz − M` = 609 | `imgsz` = 640 | validated input size, ~5% lower resolution |

Select with `--embed-mode {grow,fit}`. Non-default modes are tagged in the run id, so a
`fit` cache can never be mistaken for a `grow` one.

### 1.3 Frozen values

| parameter | value |
|---|---|
| `embed_mode` | `grow` |
| `imgsz` | 640 |
| `stride` | 32 |
| `pad_value` | 114 (gray); `0`, `-1` replicate, `-2` reflect available as ablations |
| shift representation | `(dy, dx)`, integers only — **no interpolation at any point** |
| shift range | `[0, max_shift]²`; non-negative because the canvas grows bottom-right |
| δ=(0,0) | tile at the top-left corner of the canvas |
| ground truth | **never transformed** — predictions are mapped back to original-image coordinates and evaluated against unmodified COCO annotations |
| letterbox rect | `False` (square). The published table uses `rect=True`; `scripts/01d` quantifies the gap (+0.21 e2e, +0.05 nms) |

---

## 2. Inference and post-processing

Identical for both heads — this is the whole point of the ablation.

| parameter | value | note |
|---|---|---|
| `conf_thres` | 0.001 | Ultralytics val default |
| `iou_thres` (NMS) | 0.7 | one-to-many path only |
| `max_det` | 300 | verified neutral between heads (`scripts/01b`: both gain ~0.6 AP over 50→500, curves parallel, saturating at 300) |
| `multi_label` | `True` | Ultralytics val default |
| `agnostic_nms` | `False` | |
| dtype | float32 | |
| **TF32** | **off** | on by default on Ampere; moves individual confidences by up to **0.137** |
| `cudnn.deterministic` | `True` | |
| `cudnn.benchmark` | `False` | |
| **fusion** | **off** | fusing an end-to-end head calls `Detect.fuse()`, which sets `cv2 = cv3 = None` and deletes the one-to-many branch |
| NMS kernel | `torchvision.ops.nms` | `torchvision` must be in `sys.modules` or Ultralytics silently substitutes a different, 21× slower kernel producing materially different output |

---

## 3. COCO detection AP

Standard COCO, restated so nothing is implicit.

| parameter | value |
|---|---|
| IoU thresholds | 0.50 : 0.05 : 0.95 (10) |
| recall thresholds | 0.00 : 0.01 : 1.00 (101) |
| `maxDets` | 100, applied per **(image, category)** — not per image |
| area ranges | all `[0, 1e10]`, small `[0, 32²]`, medium `[32², 96²]`, large `[96², 1e10]` |
| crowd GT | ignored; IoU against them is intersection-over-**detection-area** |
| category with GT, no detections | contributes AP 0 |
| category with no GT | excluded entirely |
| tie-break | total order `(category asc, score desc, image asc, shift asc)` so rankings are reproducible; pycocotools leaves ties to input order (difference <1e-9) |
| verification | agrees with `faster-coco-eval` to **1.1e-16** on real COCO data (gate B4) |

---

## 4. Oracle envelope — ΔAP

Manfredi & Wang Algorithm 1.

| parameter | value |
|---|---|
| `max_shift` | **1** — the only value comparable to their Table 1 |
| `step` | 1 (all 4 shifts) |
| iterations `K` | 1 — the paper measures <1% further movement |
| **objective** | **AP50** |
| objective area range | `all` |
| initialisation | δ\*ᵢ = (0,0) for every image |
| image order | dataset order (sorted COCO `image_id`) |
| `N` | 5000 (full COCO val2017) |

**On the objective.** Sec. 3.1: *"In practice, we use AP50 as the metric, that is, using 0.5
as the IoU threshold"*. The greedy search optimises **AP50**; the AP (0.50:0.95) column of
their Table 1 is AP read off **at the assignment AP50 chose** — it is not a separate
AP-driven search. Sec. 4.1 confirms the side-effect: ΔAP50 grows monotonically with max
shift but ΔAP does not, *"a result of our evaluation metric, that uses AP50 to find best and
worst shifts"*. Running with `--objective AP` is legitimate but produces a **different
metric** that cannot be placed beside their table.

**Comparability.** ΔAP widens mechanically with the number of shifts available per image.
Only runs with the same `M` and the same `N` may be compared.

---

## 5. Uniform-shift sweep — AP(δ)

| parameter | value |
|---|---|
| `max_shift` | 31 (one full stride minus one pixel) |
| `step` | 2 → 16×16 = 256 shifts |
| `N` | 1000 |
| subset seed | 0 |
| subset rule | `numpy.random.default_rng(0).choice(n, replace=False)`, then **sorted** |

**Step-2 is a lower bound.** Validated against a step-1 run on 100 images (1024 shifts):
dense AP range 4.041, step-2 3.297, step-4 3.160. The coarse grid **underestimates the
range by ~18%** — it can only miss extrema, never invent them — and it misses them at the
*minimum*. Quote step-2 ranges as lower bounds.

---

## 6. Prediction-to-ground-truth matching

This is the definition every instance-level metric rests on. It is **not** COCOeval's
matching; it is a separate, simpler rule chosen so that "was this object detected at this
shift?" has one unambiguous answer.

### 6.1 Algorithm

For each image and each shift δ, independently:

```
1. Take the de-shifted predictions for that image.
2. Discard any with score < tau.                              tau = 0.25
3. Sort the remainder by score, DESCENDING, stable (mergesort on -score).
4. Compute the IoU matrix against that image's ground truth.
   Crowd GT use intersection-over-detection-area (COCO convention).
5. Walk the predictions in that order. For prediction d:
     candidates = { g : g not yet claimed AND IoU(d, g) >= 0.5 [AND class(g) == class(d)] }
     if candidates is empty: skip d
     else: claim g* = argmax IoU over candidates,   ties -> lowest GT index
6. Each GT may be claimed by at most ONE prediction per shift (one-to-one).
```

Step 5 runs **twice, independently**:

* **class-aware** (the bracketed class condition is applied) → drives `detected(g, δ)`,
  the dropout/flicker statistics, and box jitter;
* **class-agnostic** (class condition dropped) → drives the class-flip statistic, since
  detecting an object *as the wrong class* is a different failure from not detecting it.

The two passes maintain separate claim sets and cannot interfere.

### 6.2 Declared choices

* **Greedy, not Hungarian.** Deliberately mirrors COCOeval's own rule rather than an
  optimal assignment. An optimal matching would report slightly higher detection rates and
  would not correspond to how AP is computed.
* **Crowd GT** participate in matching (so they can absorb a detection, as in COCOeval) but
  are **excluded from every reported aggregate**.
* **Unmatched GT** count as not-detected at that shift and contribute SI = 0.
* **τ is a real knob, not a formality.** At COCO's evaluation threshold of 0.001 almost
  nothing ever drops out and flicker is meaningless. τ = 0.25 is the Ultralytics
  predict-time default — the threshold a deployment would actually use.

### 6.3 Frozen thresholds

| parameter | value |
|---|---|
| `tau` (score) | **0.25** |
| IoU threshold | **0.5** |
| assignment | one-to-one, greedy, score-descending |
| IoU tie-break | lowest GT index |
| score sort | stable (mergesort) |

---

## 7. Instance-level metrics

All defined over the set of shifts in the grid, per ground-truth object.

| metric | definition |
|---|---|
| **flicker rate** | fraction of GT whose `detected(g, ·)` is **not constant** over the grid |
| **dropout rate** | fraction of GT detected at δ=(0,0) that are missed at **≥ 1** shift |
| **class-flip rate** | fraction of GT assigned **more than one distinct class** across shifts by their best class-agnostic match |
| **box jitter** | per-GT standard deviation (ddof=1) of each de-shifted coordinate, averaged over the 4 coordinates, in **original-image pixels**; also reported normalised by `sqrt(GT area)` |
| **mean detect fraction** | mean over GT of (shifts detected / shifts total) |

A perfectly shift-equivariant detector scores 0 on the first four.

**Comparability — the same constraint ΔAP carries.** Flicker, dropout and class-flip are
defined as "not constant over the grid" / "missed at ≥ 1 shift" / "more than one class
across shifts". Every one of them is **monotone non-decreasing in the number of shifts
sampled**: adding a shift can only add ways to differ, never remove one. So these numbers
are comparable **only between runs with the same shift grid**, exactly as ΔAP is comparable
only at the same `M` and `N`. This is visible in the committed data — `results/stability.csv`
has the same model and head at 1024 shifts (48.5% flicker, 63.2% dropout) and at 256 shifts
(39.2%, 46.5%).

Box jitter and the Stability Index do **not** share this property: both are averages over
observed shifts rather than existence claims over the grid, so they are stable as the grid
is refined. Quote flicker/dropout with the grid; jitter and SI stand on their own.

### 7.1 Stability Index (2-D adaptation of arXiv:2407.04305)

```
SI_c  = max(0, 1 - |c1 - c2| / (c^0.99 - c^0.01))     confidence
SI_l  = IoU(B1_loc, B2_loc)     boxes share the GT extent, differ only in centre
SI_e  = IoU(B1_ext, B2_ext)     boxes share the GT centre, differ only in extent
SI2D  = SI_c * (SI_l + SI_e) / 2
```

| parameter | value |
|---|---|
| heading term `SI_h` | **omitted** — no 2-D analogue |
| pivot box | the **ground-truth** box (the 3-D paper's geometric mean over two frames collapses to this when the object itself does not move) |
| `c^0.01`, `c^0.99` | taken from the δ=(0,0) pass and **frozen for the whole grid** |
| object present in only one member of a pair | **SI = 0** (a missed detection is maximal instability, not missing data) |
| pairing | `ref` — every δ against δ=(0,0). `adjacent` also available |

### 7.2 Required control

Instance-level metrics **must** be reported alongside the numerics floor from
`scripts/08_numerics_control.py`: the same input through the same weights with only the
arithmetic changed. Both heads end in a discrete selection (NMS, or the one-to-one head's
top-k) which can flip under a 1e-8 score difference, so a floor is not optional.

Measured floor: flicker 0.01–0.06%, jitter 0.006 px, SI 0.998. Reported shift effects
exceed it by ~3900× and ~290× respectively.

---

## 8. Shift test-time augmentation

| parameter | value |
|---|---|
| merge NMS IoU | 0.7 |
| `max_det` | 300 |
| strategies | `nms`, `topk`, `vote` |
| `vote` rule | NMS, then keep detections backed by ≥ half the shifts at IoU ≥ 0.7 |

**Headline comparison is `nms` vs `topk` on identical detections** — that difference is the
cost of staying NMS-free under shift-TTA, and it is the reason `topk` is included despite
performing terribly.

---

## 9. What is deliberately NOT fixed

Declared so that varying them later is understood as intended, not as fishing:

* **model** — yolo26n is primary; yolov8n / yolov10n / yolo11n / rtdetr-l are cross-checks
* **head** — the variable under study
* `max_shift` for the ladder experiment (M ∈ {1, 3, 7, 15}, on 100 images, per their Table 2)
* `pad_value` for the padding ablation (114 / 0 / replicate / reflect)
* `--area` for the small-object oracle
* `N` — but any change must be reported, since ΔAP is not comparable across `N`

---

## 10. Cross-model check (added v1.1.0)

Declared **before any cross-model sweep was run**, so the outcome cannot have shaped the
test. Two separate questions, two separate arms; they must not be reported as one result.

### 10.1 Replication of the head gap — `yolov10n`

`yolov10n.pt` is the only other released checkpoint that carries a one-to-one and a
one-to-many branch trained together, so it is a genuine second instance of the core
ablation: same design, different weights, different training recipe.

* **Instrument**: identical to `11_head_gap.py` — paired sign-flip dispersion test across
  δ, 10 curves (2 axes × {AP, AP50, AP_small, AP_medium, AP_large}), α = 0.05, 20 000
  permutations, absolute (not relative) deviations.
* **Declared in advance**: `AP_small` along x is the curve on which yolo26n showed its
  largest effect (e2e 1.9× more dispersed, p < 0.0001). It is therefore the pre-specified
  primary curve for replication; the other nine are secondary and reported as a count.
* **What counts as a failed replication**: the primary curve not significant, or
  significant in the opposite direction. A partial result (fewer significant curves, same
  sign) is weaker evidence, not a failure, and must be reported as such.
* Sign agreement across the two checkpoints is reported as a **descriptive concordance**,
  not a p-value — the ten curves are not independent of one another.

### 10.2 Universality of grid-locking — `yolov8n`, `yolo11n`, `rtdetr-l`

These are one-head models and contribute nothing to the head ablation. They test only
whether the stride-periodic response is a property of detectors built on a strided
downsampling cascade, or a YOLO26 quirk. RT-DETR-L matters most here: a transformer
set-prediction head over an HGNetv2 backbone shares almost nothing with YOLO26 except the
2/4/8/16/32 downsampling ladder, so if it is grid-locked too, the cascade is the mechanism.

Cross-model dispersion is reported **both** in AP points and as a coefficient of variation
(mean |deviation| ÷ mean AP). The arms sit between ~37 and ~53 AP, so absolute points alone
would partly measure where a model sits on the AP scale.

### 10.3 Gate A applies to every arm — with one declared deviation

No model's shift behaviour is quoted before it reproduces its published COCO AP. Tolerance
is 0.1 AP, the same as yolo26n, **except `rtdetr-l`, which is 0.3** and is declared here as
a deviation rather than applied silently. Its published 53.0 is the RT-DETR-R50 number from
the original paper's PaddlePaddle evaluation, quoted by Ultralytics for their PyTorch port
rather than re-measured. Measured here at 52.77 with Ultralytics' own validator; the
residual does not move with `conf` or `max_det`
(`results/phase0/rtdetr_protocol_probe.json`), so it is a port gap. For that arm Gate A is a
sanity check — a preprocessing mistake would cost whole AP points, not a quarter of one —
rather than a reproduction, and no absolute-AP claim is made from it.

---

## 11. Padding ablation (added v1.1.0)

Kayhan & van Gemert (CVPR'20): a constant padding border is an implicit **absolute
position** cue. That is a competing explanation for everything measured so far, and the two
explanations make opposite predictions:

| mechanism | prediction when the border stops being constant |
|---|---|
| aliasing from strided downsampling | stride periodicity survives |
| border position cue | stride periodicity weakens or disappears |

The four arms split into two **groups**, and the grouping is the design:

| `pad_value` | border | absolute-position cue available? |
|---|---|---|
| `114` | gray constant — the frozen protocol | yes |
| `0` | black constant — Manfredi & Wang's canvas | yes |
| `-1` | replicate (edge pixels extended) | no |
| `-2` | reflect (mirrored content) | no |

Comparing 114 against 0 alone cannot decide anything, because both are constants; `0`
controls for the specific *value* while `{-1, -2}` remove the *constant-ness*. The flag sets
both borders — the letterbox bars when the preprocessing cache is built and the canvas frame
at embed time — so this changes one thing, not two.

* **N = 1000**, both heads, δx and δy over [0, 63] at 1-px resolution. Declared in advance:
  if the group difference in stride power is within ±0.10 the ablation is treated as
  underpowered rather than null, and re-run at N = 5000 before any claim is made.
* Comparisons against the pad-114 baseline use **relative** dispersion, because changing the
  padding changes the AP level and absolute points would conflate "wobbles less" with
  "scores lower".

---

## 12. The two remaining declared variations (added v1.2.0)

§9 has listed both since v1.0.0 as axes intended to be varied. This section says what each
one predicts and what would refute it — written **before either was run**, for the same
reason §10 was: an instrument whose expected outcome is only written down afterwards can be
read either way.

### 12.1 Small-object oracle — `--area small`

§4 pins the greedy objective's area range to `all`. The `AP_small` figure reported next to
it comes from `summary_at_assignment`: AP_small evaluated **at the assignment that maximised
overall AP50**. That is a description of that assignment, not a search over small objects,
and the README says so. This is the real one.

* **Instrument**: identical to §4 in every respect — `max_shift` 1, `step` 1, `K` 1,
  objective AP50, initialisation δ\*ᵢ = (0,0), image order, `N` = 5000 — with the objective's
  **area range** changed from `all` to `small` and nothing else.
* **Arms**: `yolo26n` e2e/nms (primary) and `yolov10n` e2e/nms (replication), reusing the
  existing detection caches. No new inference.
* **Declared in advance**: the instance metrics put the head difference overwhelmingly in
  small-object dropout — 80.2% vs 57.7% on yolo26n, 80.9% vs 53.2% on yolov10n — so a
  small-object envelope should separate the heads **more** than the all-area envelope did.
  The prediction is Δ(e2e) > Δ(nms) on both checkpoints, with a **relative** gap exceeding
  the all-area gap. The **primary quantity is the ΔAP ratio**, because that is what the
  +15.4% (yolo26n) and +26.8% (yolov10n) figures the prediction is calibrated against are;
  the ΔAP50 ratio is reported alongside it as secondary. Both are stated whatever they say,
  so that neither can be selected after the fact.
* **What would refute it**: e2e's small-object envelope no wider than nms's, or a relative
  gap *smaller* than the all-area gap. Either says small objects are not where the head
  difference lives, and finding 1's stated mechanism would need rewriting rather than
  footnoting.
* **Comparability**: ΔAP is only ever comparable within a fixed (`M`, `N`, area). A
  small-object envelope sits on a different AP scale; only the e2e-vs-nms contrast *within*
  it is readable. It must not be placed beside the all-area ΔAP as if it were a larger
  version of the same number.
* Artifacts are written with `--tag small`, because without it the run overwrites the
  committed all-area assignment `.npy` files that the published ΔAP rests on.

### 12.2 Max-shift ladder — Manfredi & Wang Table 2

* **Grid**: M ∈ {1, 3, 7, 15}, `step` 1, both heads, `yolo26n`. The shifted set grows
  quadratically in M, so **N = 100** is what makes M=15 reachable — the same constraint that
  drove their choice. They drew 100 images at random; ours is the frozen §1.3 subset rule
  (`default_rng(0).choice`, then sorted), i.e. deterministic rather than random.
* **Declared in advance**, from their Sec. 4.1:
  1. ΔAP50 grows **monotonically** with M, and **sub-linearly**;
  2. best and worst are roughly **symmetric about the δ=(0,0) baseline** — the unshifted
     number behaves like an average case, not a best case;
  3. ΔAP (0.50:0.95) **need not** grow monotonically, because the search optimises AP50.
     This is their own caveat and the reason §4 pins the objective to AP50.
* **What would refute it**: super-linear growth in M, or a strongly asymmetric envelope.
* **Comparability**: ΔAP is not comparable across `M` or across `N`. The ladder's M=1 point
  is an N=100 measurement and is **not** the published M=1 / N=5000 number; quoting one for
  the other would be an error. The ladder is read as a **shape**, not as levels.

---

## Changelog

| version | date | change |
|---|---|---|
| 1.0.0 | 2026-08-15 | Initial freeze. Protocol under which all results in `results/` and `RESULTS.md` were produced. |
| 1.1.0 | 2026-08-17 | Added §10 (cross-model check) and §11 (padding ablation). Both were already declared as intended variations in §9 of v1.0.0; v1.1.0 specifies the analysis for them. Written **after** the yolo26n full-validation results were known and **before** any cross-model or padding sweep had been run. Nothing about the yolo26n analysis changed. |
| 1.2.0 | 2026-08-19 | Added §12 — the small-object oracle and the max-shift ladder, the last two variations §9 declared in v1.0.0. States what each predicts and what would refute it, written **before either was run**. No v1.0.0 or v1.1.0 value changed. |
