#!/usr/bin/env python3
"""Diff our detections against Ultralytics' predictions.json, box by box.

02b found our identity-geometry pipeline sits 0.033 AP below Ultralytics' validator, the
same offset on both heads. 02c ruled out Conv+BN folding (+0.001). This script stops
guessing and compares the actual outputs: per-image detection counts first, then, for
images whose counts match, the largest coordinate and score disagreement.

Three outcomes, three different problems:
  * counts differ            -> a thresholding / NMS / max_det difference
  * counts match, boxes move -> a preprocessing or de-shift geometry difference
  * everything matches       -> the gap is in the *evaluation*, not the detections
                                (Ultralytics rounds boxes to 3dp and scores to 5dp when
                                 it writes predictions.json; we evaluate full precision)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.cocolite import class_map_inverse, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, load_coco_gt  # noqa: E402
from shifteq.detstore import DetStore, coco80_to_coco91, to_coco_json  # noqa: E402
from shifteq.modelzoo import MODELS, resolve_weights  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402

_A = import_module("01_reproduce_published")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--head", default="nms", choices=["e2e", "nms"])
    ap.add_argument("--store-dir", default="cache/dets")
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pred-json", default=None,
                    help="skip the val run and diff against this predictions.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    spec = MODELS[args.model]
    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    # --- reference: Ultralytics' own square-letterbox run -------------------
    if args.pred_json:
        pj = Path(args.pred_json)
    else:
        LOGGER.info("running Ultralytics val (rect=False) to produce predictions.json ...")
        r = _A.val_one(str(resolve_weights(spec)), args.head, args.data, spec.imgsz, args.batch,
                       args.device, args.workers, str(ann), rect=False)
        pj = Path(r["save_dir"]) / "predictions.json"
        LOGGER.info(f"  reference AP = {r['coco']['mAP50-95']:.4f}   ({pj})")
    theirs_raw = json.loads(pj.read_text())

    # --- ours ---------------------------------------------------------------
    run_id = f"{args.model}_{args.head}_m0s1_full_pad114"
    store = DetStore(args.store_dir, run_id)
    man = store.manifest
    gt = load_coco_gt(ann, man.image_ids)
    cmi = class_map_inverse(gt, coco80_to_coco91())
    npig = {a: npig_per_cat(gt, a) for a in AREA_KEYS}
    dets = store.load(0, 0)
    ours_raw = to_coco_json(dets, man.image_ids, coco80_to_coco91(), round_coords=False)

    LOGGER.info(f"detections: ours {len(ours_raw):,} | theirs {len(theirs_raw):,} "
                f"({len(ours_raw) - len(theirs_raw):+,})")

    def group(rows):
        g = defaultdict(list)
        for r in rows:
            g[int(r["image_id"])].append(r)
        return g

    A, B = group(ours_raw), group(theirs_raw)
    ids = sorted(set(A) | set(B))
    n_count_diff = 0
    worst_box = worst_score = 0.0
    examples = []
    for iid in ids:
        a, b = A.get(iid, []), B.get(iid, [])
        if len(a) != len(b):
            n_count_diff += 1
            if len(examples) < 5:
                examples.append({"image_id": iid, "ours": len(a), "theirs": len(b)})
            continue
        if not a:
            continue
        a = sorted(a, key=lambda r: (-r["score"], r["category_id"]))
        b = sorted(b, key=lambda r: (-r["score"], r["category_id"]))
        ab = np.array([r["bbox"] for r in a], dtype=np.float64)
        bb = np.array([r["bbox"] for r in b], dtype=np.float64)
        asc = np.array([r["score"] for r in a])
        bsc = np.array([r["score"] for r in b])
        worst_box = max(worst_box, float(np.abs(ab - bb).max()))
        worst_score = max(worst_score, float(np.abs(asc - bsc).max()))

    LOGGER.info(f"images with different detection counts: {n_count_diff} / {len(ids)}")
    LOGGER.info(f"max |bbox delta| on count-matched images: {worst_box:.6f} px")
    LOGGER.info(f"max |score delta| on count-matched images: {worst_score:.8f}")
    if examples:
        LOGGER.info(f"  examples: {examples}")

    # --- is the residual purely the JSON rounding? --------------------------
    full = evaluate_detections(gt, dets, cmi, npig_cache=npig)["AP"] * 100
    rounded_json = to_coco_json(dets, man.image_ids, coco80_to_coco91(), round_coords=True)
    tmp = Path("/tmp/ours_rounded.json")
    tmp.write_text(json.dumps(rounded_json))
    rounded = _A.coco_eval_json(tmp, ann)["mAP50-95"]
    theirs = _A.coco_eval_json(pj, ann)["mAP50-95"]
    LOGGER.info("")
    LOGGER.info(f"AP, our detections, full precision : {full:.4f}")
    LOGGER.info(f"AP, our detections, rounded as they do: {rounded:.4f}  ({rounded - full:+.4f})")
    LOGGER.info(f"AP, their predictions.json         : {theirs:.4f}  ({theirs - full:+.4f})")

    report = {
        "model": spec.name, "head": args.head, "pred_json": str(pj),
        "n_ours": len(ours_raw), "n_theirs": len(theirs_raw),
        "images_with_count_diff": n_count_diff, "n_images": len(ids),
        "max_bbox_delta_px": worst_box, "max_score_delta": worst_score,
        "examples": examples,
        "AP_ours_full_precision": full, "AP_ours_rounded": rounded, "AP_theirs": theirs,
        "rounding_effect": rounded - full, "residual_after_rounding": theirs - rounded,
    }
    write_json(args.out or f"results/phase0/detection_diff_{args.head}.json", report)
    LOGGER.info(f"wrote results/phase0/detection_diff_{args.head}.json")


if __name__ == "__main__":
    main()
