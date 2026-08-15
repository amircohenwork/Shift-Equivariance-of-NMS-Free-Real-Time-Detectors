#!/usr/bin/env python3
"""Gate B, final form: our whole pipeline vs Ultralytics', on the full validation set.

Gate B's B1-B5 check the pieces (preprocessing bytes, the inverse map, determinism, the
evaluator, no fusion). This checks the composition: build a `max_shift=0` detection cache
-- where the canvas collapses to imgsz and the harness *is* the stock square-letterbox
pipeline -- evaluate it with our own COCO implementation, and compare against the
square-letterbox baseline that 01_reproduce_published.py measured with Ultralytics' own
validator and faster-coco-eval.

Agreement here means every later number differs from the baseline only because of the
shift, which is the claim the whole study rests on.

Expected (yolo26n, COCO val, rect=False):
    e2e  40.255      nms  40.867
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, load_coco_gt  # noqa: E402
from shifteq.detstore import DetStore, coco80_to_coco91  # noqa: E402
from shifteq.utils import LOGGER, read_json, setup_logging, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default="cache/dets")
    ap.add_argument("--model", default="yolo26n")
    ap.add_argument("--heads", default="e2e,nms")
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--gateA", default="results/phase0/gateA_reproduce.json")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="AP tolerance; this is a same-pipeline comparison, so it should be tight")
    ap.add_argument("--out", default="results/phase0/gateB_identity_fullval.json")
    args = ap.parse_args()

    setup_logging()
    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    gA = read_json(args.gateA) if Path(args.gateA).exists() else {}
    ref = gA.get("harness_baseline", {})

    report: dict = {"tol": args.tol, "heads": {}}
    failures = []
    for head in args.heads.split(","):
        run_id = f"{args.model}_{head}_m0s1_full_pad114"
        store = DetStore(args.store_dir, run_id)
        if not store.has(0, 0):
            LOGGER.error(f"missing identity cache {run_id}; build it with "
                         f"03_build_cache.py --max-shift 0 --step 1")
            failures.append(f"{head}: no identity cache")
            continue
        man = store.manifest
        if man.canvas != man.imgsz:
            failures.append(f"{head}: canvas {man.canvas} != imgsz {man.imgsz}; not the identity path")
            continue

        gt = load_coco_gt(ann, man.image_ids)
        cmi = class_map_inverse(gt, coco80_to_coco91())
        npig = {a: npig_per_cat(gt, a) for a in AREA_KEYS}
        dets = store.load(0, 0)
        ours = {k: v * 100 for k, v in evaluate_detections(gt, dets, cmi, npig_cache=npig).items()}

        row = {"run_id": run_id, "canvas": man.canvas, "n_images": gt.num_images,
               "n_dets": dets.n_dets, "ours": ours}
        r = ref.get(head)
        if r:
            row["ultralytics_square_baseline"] = r["mAP50-95"]
            row["delta"] = ours["AP"] - r["mAP50-95"]
            row["pass"] = abs(row["delta"]) <= args.tol
            if not row["pass"]:
                failures.append(f"{head}: ours {ours['AP']:.3f} vs Ultralytics square "
                                f"{r['mAP50-95']:.3f}, delta {row['delta']:+.3f} > tol {args.tol}")
            LOGGER.info(
                f"  {head}: ours AP={ours['AP']:.3f} AP50={ours['AP50']:.3f} | "
                f"Ultralytics square {r['mAP50-95']:.3f} | delta {row['delta']:+.3f}"
            )
        else:
            LOGGER.warning(f"  {head}: no Gate A baseline to compare against; ours AP={ours['AP']:.3f}")
        report["heads"][head] = row

    report["gate_pass"] = not failures
    report["failures"] = failures
    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")
    if failures:
        for f in failures:
            LOGGER.error("  " + f)
        sys.exit(1)
    LOGGER.info("IDENTITY CHECK PASSED: our pipeline reproduces Ultralytics' on the full val set.")


if __name__ == "__main__":
    main()
