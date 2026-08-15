#!/usr/bin/env python3
"""Shift test-time augmentation (Manfredi & Wang Sec. 5.4), pointed at the NMS-free head.

Reads a cached M=1 grid and merges the 4 de-shifted prediction sets three ways:

  nms    class-wise NMS over the union -- the paper's aggregation
  topk   concatenate, keep top-k by score, no suppression -- the only merge that keeps
         the "NMS-free" property, and the one we expect to fail
  vote   NMS, then keep only detections backed by a majority of shifts

The headline comparison is not TTA-vs-baseline. It is `nms` vs `topk` on the *same*
detections: if the end-to-end model needs `nms` to benefit from its own shift variance,
then the NMS-free deployment story does not survive contact with test-time augmentation,
and that is a result worth stating.

Also prints AP_best from the oracle if available, since the paper's framing is "how much
of the achievable envelope does TTA actually recover".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, load_coco_gt  # noqa: E402
from shifteq.detstore import DetStore, coco80_to_coco91  # noqa: E402
from shifteq.tta import merge_shifts  # noqa: E402
from shifteq.utils import LOGGER, read_json, set_deterministic, setup_logging, shift_grid, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default="cache/dets")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--strategies", default="nms,topk,vote")
    ap.add_argument("--oracle-json", default=None, help="04_oracle output, to report envelope recovery")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)

    store = DetStore(args.store_dir, args.run_id)
    man = store.manifest
    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    gt = load_coco_gt(ann, man.image_ids)
    cmi = class_map_inverse(gt, coco80_to_coco91())
    npig = {a: npig_per_cat(gt, a) for a in AREA_KEYS}

    grid = shift_grid(man.max_shift, man.step)
    per_shift = [store.load(*s) for s in grid]
    LOGGER.info(f"{args.run_id}: merging {len(grid)} shifts over {gt.num_images} images")

    report = {"run_id": args.run_id, "model": man.model, "head": man.head,
              "max_shift": man.max_shift, "n_shifts": len(grid), "n_images": gt.num_images,
              "iou": args.iou, "max_det": args.max_det, "results": {}}

    base = evaluate_detections(gt, per_shift[0], cmi, npig_cache=npig)
    report["baseline_delta0"] = base
    LOGGER.info(f"  baseline delta=(0,0): AP={base['AP'] * 100:.2f} AP50={base['AP50'] * 100:.2f}")

    for strat in args.strategies.split(","):
        merged = merge_shifts(per_shift, strategy=strat, iou_thr=args.iou, max_det=args.max_det)
        r = evaluate_detections(gt, merged, cmi, npig_cache=npig)
        r["n_dets"] = merged.n_dets
        r["dAP_vs_baseline"] = r["AP"] - base["AP"]
        r["dAP50_vs_baseline"] = r["AP50"] - base["AP50"]
        report["results"][strat] = r
        LOGGER.info(
            f"  {strat:5s}: AP={r['AP'] * 100:.2f} ({r['dAP_vs_baseline'] * 100:+.2f})  "
            f"AP50={r['AP50'] * 100:.2f} ({r['dAP50_vs_baseline'] * 100:+.2f})  "
            f"dets={merged.n_dets:,}"
        )

    if args.oracle_json and Path(args.oracle_json).exists():
        o = read_json(args.oracle_json)
        if "delta" in o:
            best = o["delta"]["AP_best"]
            report["oracle"] = {"AP_best": best, "AP_worst": o["delta"]["AP_worst"]}
            for strat, r in report["results"].items():
                gap = best - base["AP"]
                r["envelope_recovered"] = (r["AP"] - base["AP"]) / gap if gap > 1e-9 else None
                LOGGER.info(f"  {strat:5s}: recovers {100 * (r['envelope_recovered'] or 0):.1f}% "
                            f"of the baseline->AP_best gap")

    if {"nms", "topk"} <= set(report["results"]):
        d = report["results"]["nms"]["AP"] - report["results"]["topk"]["AP"]
        report["nms_minus_topk_AP"] = d
        LOGGER.info(
            f"\n  suppression is worth {d * 100:+.2f} AP on the same detections. "
            "For an end-to-end head, that is the cost of staying NMS-free under shift-TTA."
        )

    out = args.out or f"results/tta/{args.run_id}.json"
    write_json(out, report)
    LOGGER.info(f"wrote {out}")


if __name__ == "__main__":
    main()
