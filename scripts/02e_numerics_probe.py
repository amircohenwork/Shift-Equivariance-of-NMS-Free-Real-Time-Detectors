#!/usr/bin/env python3
"""Which numerics knob accounts for the last 0.033 AP -- and is it larger than the effect
we are trying to measure?

The trail so far: our identity-geometry pipeline sits 0.033 AP below Ultralytics'
validator on both heads. Detections are 99.999% identical (4 of 4998 images differ in
count, 3 detections out of 747k), JSON rounding is worth -0.0008, and Conv+BN folding is
worth +0.001. But scores on matched detections differ by up to 1.07e-3 -- three orders of
magnitude more than fp32 convolution noise, and almost exactly the relative precision of
**TF32** (10 mantissa bits, ~1e-3).

`shifteq.utils.set_deterministic` turns TF32 off. Ultralytics leaves it on, which is the
right default for throughput and the wrong one here. This script measures the difference.

Why it matters far beyond reproducing a published number: this study asks how much a
one-pixel translation perturbs a detector. If the arithmetic itself moves confidences by
~1e-3, that noise floor sits on top of every shift measurement. Quantifying it tells us
whether our signal clears it -- and TF32 is silently on by default on every Ampere-class
GPU, which is what most people will run this on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, PreprocCache, load_coco_gt  # noqa: E402
from shifteq.detstore import coco80_to_coco91  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner  # noqa: E402
from shifteq.modelzoo import MODELS, load_detector, set_head  # noqa: E402
from shifteq.shift import ShiftGeometry  # noqa: E402
from shifteq.utils import LOGGER, read_json, set_deterministic, setup_logging, write_json  # noqa: E402


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def run(head, spec, cache, gt, cmi, npig, device, batch, *, tf32: bool, fuse: bool,
        deterministic: bool = True):
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    set_tf32(tf32)
    net, spec = load_detector(spec, device=device)
    set_head(net, head)
    if fuse:
        net.fuse(verbose=False)
    geom = ShiftGeometry(spec.imgsz, 0, spec.stride)
    dets = ShiftRunner(net, spec, cache, geom, PostprocCfg(), device, batch).run_shift(0, 0)
    ap = {k: v * 100 for k, v in evaluate_detections(gt, dets, cmi, npig_cache=npig).items()}
    return ap, dets


def score_spread(a, b) -> dict:
    """How far apart are two runs' confidences, per image, on same-count images?"""
    worst = 0.0
    n_same = 0
    for i in range(a.n_images):
        _, sa, _ = a.per_image(i)
        _, sb, _ = b.per_image(i)
        if len(sa) != len(sb) or not len(sa):
            continue
        n_same += 1
        worst = max(worst, float(np.abs(np.sort(sa)[::-1] - np.sort(sb)[::-1]).max()))
    return {"images_compared": n_same, "max_abs_score_delta": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--head", default="nms", choices=["e2e", "nms"])
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--gateA", default="results/phase0/gateA_reproduce.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(args.data)
    root = Path(data["path"])
    ann = Path(args.ann) if args.ann else root / "annotations" / "instances_val2017.json"
    img_dir = Path(args.img_dir) if args.img_dir else root / "images" / "val2017"

    spec = MODELS[args.model]
    gt = load_coco_gt(ann)
    ids = [int(i) for i in gt.image_ids]
    cmi = class_map_inverse(gt, coco80_to_coco91())
    npig = {a: npig_per_cat(gt, a) for a in AREA_KEYS}
    cache = PreprocCache(Path(args.cache_dir) / "preproc", spec, ids, img_dir).build()

    gA = read_json(args.gateA) if Path(args.gateA).exists() else {}
    target = gA.get("harness_baseline", {}).get(args.head, {}).get("mAP50-95")

    configs = [
        ("ours (tf32 off, unfused, deterministic)", dict(tf32=False, fuse=False)),
        ("tf32 ON, unfused",                        dict(tf32=True,  fuse=False)),
        ("tf32 ON, fused  (= Ultralytics defaults)", dict(tf32=True,  fuse=True)),
        ("tf32 off, unfused, cudnn.benchmark",       dict(tf32=False, fuse=False, deterministic=False)),
    ]
    out: dict = {"model": spec.name, "head": args.head,
                 "ultralytics_square_baseline": target, "runs": {}}
    base_dets = None
    for label, kw in configs:
        r, d = run(args.head, spec, cache, gt, cmi, npig, args.device, args.batch, **kw)
        row = {"AP": r["AP"], "AP50": r["AP50"], "n_dets": d.n_dets, **kw}
        if target is not None:
            row["delta_vs_ultralytics"] = r["AP"] - target
        if base_dets is None:
            base_dets = d
        else:
            row.update(score_spread(base_dets, d))
        out["runs"][label] = row
        LOGGER.info(
            "{:44s} AP={:.4f}  dets={:,}  vs-ultralytics {:+.4f}  max|dscore|={:.2e}".format(
                label, r["AP"], d.n_dets, row.get("delta_vs_ultralytics", float("nan")),
                row.get("max_abs_score_delta", 0.0)
            )
        )

    set_tf32(False)
    write_json(args.out or f"results/phase0/numerics_probe_{args.head}.json", out)
    LOGGER.info(f"wrote results/phase0/numerics_probe_{args.head}.json")


if __name__ == "__main__":
    main()
