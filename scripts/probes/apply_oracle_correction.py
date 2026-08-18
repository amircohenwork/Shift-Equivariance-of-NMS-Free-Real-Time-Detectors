#!/usr/bin/env python3
"""Rewrite the committed oracle results with AP recomputed by the fixed accumulator.

The greedy assignments are unchanged -- a 600-image probe found 0 argmax disagreements
between the buggy and fixed objectives, because the bug is a near-constant offset that
cancels in the differences the search compares. Only the AP values attached to those
assignments move, by ~0.0013.

The original values are preserved inside an `accumulator_correction` block rather than
overwritten silently: a results file that quietly changed its numbers would be worse than
one that was wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from shifteq.cocolite import build_matches, class_map_inverse, npig_per_cat
from shifteq.data import load_coco_gt
from shifteq.detstore import DetStore, coco80_to_coco91
from shifteq.oracle import NumpyAccumulator, build_union
from shifteq.utils import LOGGER, read_json, setup_logging, write_json

setup_logging()
from ultralytics.data.utils import check_det_dataset  # noqa: E402

root = Path(check_det_dataset("coco.yaml")["path"])
ann = root / "annotations" / "instances_val2017.json"

NOTE = ("TorchAccumulator on CUDA computed `tpc / npig` with a Python-int divisor, which "
        "lowers to a reciprocal multiply (one ULP below true division). COCO's 101 recall "
        "thresholds are exact decimals, so categories whose npig makes m/npig land on a "
        "threshold had their interpolation index flipped: 7 of 80 categories, ~0.0013 AP "
        "per evaluation. Fixed by dividing by a device tensor. AP recomputed here with "
        "NumpyAccumulator, which agrees with cocolite.evaluate_detections and hence with "
        "faster-coco-eval (gate B4) to 1.1e-16. Greedy assignments unchanged: 0/600 argmax "
        "disagreements in results/oracle/*_greedy_flip_probe.json.")

for head in ("e2e", "nms"):
    run = f"yolo26n_{head}_m1s1_full_pad114"
    p = Path(f"results/oracle/{run}_AP50_all.json")
    r = read_json(p)
    if "accumulator_correction" in r:
        LOGGER.info(f"{p.name}: already corrected, skipping")
        continue

    store = DetStore("cache/dets", run)
    m = store.manifest
    gt = load_coco_gt(ann, list(m.image_ids))
    cmi = class_map_inverse(gt, coco80_to_coco91())
    npig = npig_per_cat(gt, "all")
    grid = sorted(store.available_shifts())
    ps = [build_matches(gt, store.load(dy, dx), cmi, area_key="all") for dy, dx in grid]
    acc = NumpyAccumulator(build_union(ps, npig, gt.num_images, gt.num_cats))

    before = {"baseline": dict(r["baseline"]), "best_AP": r["best"]["AP"],
              "best_AP50": r["best"]["AP50"], "worst_AP": r["worst"]["AP"],
              "worst_AP50": r["worst"]["AP50"], "delta": dict(r["delta"])}

    def ev(assign):
        a, a50 = acc.ap(acc.new_mask(np.asarray(assign, dtype=np.int64)))
        return a, a50

    b_ap, b_ap50 = ev(np.zeros(gt.num_images, dtype=np.int64))
    r["baseline"] = {"AP": b_ap, "AP50": b_ap50}
    for d in ("best", "worst"):
        a, a50 = ev(np.load(f"results/oracle/{run}_{d}_assignment.npy"))
        r[d]["AP"], r[d]["AP50"] = a, a50
    r["delta"] = {
        "AP_best": r["best"]["AP"], "AP_worst": r["worst"]["AP"],
        "dAP": r["best"]["AP"] - r["worst"]["AP"],
        "AP50_best": r["best"]["AP50"], "AP50_worst": r["worst"]["AP50"],
        "dAP50": r["best"]["AP50"] - r["worst"]["AP50"],
    }
    r["backend"] = "numpy (recomputed; search ran on torch/cuda)"
    r["accumulator_correction"] = {"applied": "2026-08-17", "note": NOTE,
                                   "values_before": before}
    write_json(p, r)
    LOGGER.info(f"{head}: dAP {before['delta']['dAP'] * 100:.4f} -> {r['delta']['dAP'] * 100:.4f}  "
                f"dAP50 {before['delta']['dAP50'] * 100:.4f} -> {r['delta']['dAP50'] * 100:.4f}")
    LOGGER.info(f"  wrote {p}")
