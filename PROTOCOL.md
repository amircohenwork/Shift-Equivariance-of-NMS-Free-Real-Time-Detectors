# Frozen Experimental Protocol — v1.0.0

**Status: FROZEN as of 2026-08-15.** Everything in `results/` and every number in
`README.md` was produced under this version.

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
than the 640 px it was validated at. Gate B6 measures this at δ=(0,0): **+0.225 AP**
overall, **−0.065 AP** on small objects. It is reported as a result, not hidden.

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

## Changelog

| version | date | change |
|---|---|---|
| 1.0.0 | 2026-08-15 | Initial freeze. Protocol under which all results in `results/` and `README.md` were produced. |
