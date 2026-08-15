#!/usr/bin/env python3
"""The noise floor: how much does COCO AP move under perturbations too small to see?

This closes out the Gate B investigation and, more importantly, sets the bar every later
result has to clear.

The trail: our identity-geometry pipeline lands 0.033 AP below Ultralytics' validator.
Ruled out, each with its own script -- preprocessing (byte-identical on all 5000 images,
02_ B1), the de-shift map (exact, B2), the evaluator (agrees with faster-coco-eval to
1e-16, B4), max_det (01b), top-k selection (01c), the letterbox protocol (01d),
Conv+BN folding (+0.001, 02c), TF32 and cudnn.benchmark (02e). What remains, from the
content diff in 02d: about half the images carry confidence differences at the **1e-5**
level, which is the last digit Ultralytics writes to predictions.json.

So the hypothesis is that COCO AP is simply that sensitive. AP ranks ~750k detections
globally and interpolates precision at 101 recall points; near-ties are everywhere, and
a perturbation far below any meaningful threshold reshuffles enough of them to move the
third decimal.

This script tests it directly: take our own detections, jitter the confidences by a
controlled epsilon, and measure the AP spread. If epsilon = 1e-5 reproduces ~0.03 AP,
the residual is explained and -- much more usefully -- we have a number that says how
large a shift-induced AP difference has to be before it means anything at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, load_coco_gt  # noqa: E402
from shifteq.detstore import DetStore, ShiftDets, coco80_to_coco91  # noqa: E402
from shifteq.utils import LOGGER, setup_logging, write_json  # noqa: E402


def jitter(dets: ShiftDets, eps: float, rng) -> ShiftDets:
    """Multiplicative jitter, so the perturbation scales with the confidence the way a
    floating-point rounding difference does."""
    s = dets.score.astype(np.float64)
    s = s * (1.0 + rng.uniform(-eps, eps, size=s.shape))
    return ShiftDets(counts=dets.counts, cls=dets.cls,
                     score=np.clip(s, 1e-9, 1.0).astype(np.float32),
                     xyxy=dets.xyxy, shift=dets.shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default="cache/dets")
    ap.add_argument("--run-id", default="yolo26n_nms_m0s1_full_pad114")
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--eps", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/phase0/ap_noise_floor.json")
    args = ap.parse_args()

    setup_logging()
    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    store = DetStore(args.store_dir, args.run_id)
    man = store.manifest
    gt = load_coco_gt(ann, man.image_ids)
    cmi = class_map_inverse(gt, coco80_to_coco91())
    npig = {a: npig_per_cat(gt, a) for a in AREA_KEYS}
    dets = store.load(0, 0)

    base = evaluate_detections(gt, dets, cmi, npig_cache=npig)
    LOGGER.info(f"baseline AP = {base['AP'] * 100:.4f}  ({dets.n_dets:,} detections)")

    out: dict = {"run_id": args.run_id, "n_dets": dets.n_dets,
                 "baseline_AP": base["AP"] * 100, "repeats": args.repeats, "levels": {}}
    for e in [float(x) for x in args.eps.split(",")]:
        rng = np.random.default_rng(0)
        aps, ap50s = [], []
        for _ in range(args.repeats):
            r = evaluate_detections(gt, jitter(dets, e, rng), cmi, npig_cache=npig)
            aps.append(r["AP"] * 100)
            ap50s.append(r["AP50"] * 100)
        aps = np.array(aps)
        row = {
            "eps": e, "AP_mean": float(aps.mean()), "AP_std": float(aps.std(ddof=1)),
            "AP_min": float(aps.min()), "AP_max": float(aps.max()),
            "AP_spread": float(aps.max() - aps.min()),
            "max_abs_dev_from_baseline": float(np.abs(aps - base["AP"] * 100).max()),
            "AP50_spread": float(np.max(ap50s) - np.min(ap50s)),
        }
        out["levels"][f"{e:g}"] = row
        LOGGER.info(
            "eps={:<8g} AP {:.4f} +/- {:.4f}  spread {:.4f}  max|dev| {:.4f}  AP50 spread {:.4f}".format(
                e, row["AP_mean"], row["AP_std"], row["AP_spread"],
                row["max_abs_dev_from_baseline"], row["AP50_spread"]
            )
        )

    LOGGER.info("")
    LOGGER.info("Read this as: an AP difference between two shifts is only meaningful if it "
                "exceeds the spread at the epsilon your arithmetic actually achieves.")
    write_json(args.out, out)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
