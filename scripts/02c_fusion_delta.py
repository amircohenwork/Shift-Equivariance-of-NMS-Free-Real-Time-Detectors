#!/usr/bin/env python3
"""How much AP does Conv+BN folding move? Measure it, don't assume it.

`modelzoo.py` refuses to fuse for two reasons: fusing an end-to-end head deletes the
one-to-many branch, and folding changes the low-order bits of every activation. The first
is categorical. The second is a quantitative claim, and 02b turned it into a number worth
pinning down: our unfused pipeline scores 0.033 AP *below* Ultralytics' validator on both
heads -- the same offset for e2e and nms, which is the signature of a shared-backbone
numerics effect rather than anything head-specific.

This script runs our own pipeline twice on identical weights, fused and unfused, so the
only variable is the folding. If the fused run lands on Ultralytics' number, the residual
in 02b is fully explained and the harness is exact.

Fusing order matters: `BaseModel.fuse()` calls `Detect.fuse()` -- which sets
`cv2 = cv3 = None` -- only when the head is currently in end2end mode. So for the NMS arm
we must switch the head *before* fusing, or the branch we want to measure is destroyed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, PreprocCache, load_coco_gt  # noqa: E402
from shifteq.detstore import coco80_to_coco91  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner  # noqa: E402
from shifteq.modelzoo import MODELS, available_heads, load_detector, set_head  # noqa: E402
from shifteq.shift import ShiftGeometry  # noqa: E402
from shifteq.utils import LOGGER, read_json, set_deterministic, setup_logging, write_json  # noqa: E402


def run(head: str, fuse: bool, spec, cache, gt, cmi, npig, device, batch) -> dict:
    net, spec = load_detector(spec, device=device)
    # switch first, fuse second -- see the module docstring
    set_head(net, head)
    if fuse:
        net.fuse(verbose=False)
        LOGGER.info(f"  fused; heads now available: {available_heads(net)}")
    geom = ShiftGeometry(spec.imgsz, 0, spec.stride)      # identity geometry
    runner = ShiftRunner(net, spec, cache, geom, PostprocCfg(), device, batch)
    dets = runner.run_shift(0, 0)
    r = {k: v * 100 for k, v in evaluate_detections(gt, dets, cmi, npig_cache=npig).items()}
    r["n_dets"] = dets.n_dets
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--heads", default="e2e,nms")
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--gateA", default="results/phase0/gateA_reproduce.json")
    ap.add_argument("--out", default="results/phase0/fusion_delta.json")
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
    ref = gA.get("harness_baseline", {})

    out: dict = {"model": spec.name, "n_images": len(ids), "heads": {}}
    for head in args.heads.split(","):
        LOGGER.info(f"=== {head}: unfused ===")
        u = run(head, False, spec, cache, gt, cmi, npig, args.device, args.batch)
        LOGGER.info(f"=== {head}: fused ===")
        f = run(head, True, spec, cache, gt, cmi, npig, args.device, args.batch)
        row = {
            "unfused": u, "fused": f,
            "fused_minus_unfused_AP": f["AP"] - u["AP"],
            "n_dets_delta": f["n_dets"] - u["n_dets"],
        }
        r = ref.get(head)
        if r:
            row["ultralytics_square_baseline"] = r["mAP50-95"]
            row["fused_minus_ultralytics"] = f["AP"] - r["mAP50-95"]
            row["unfused_minus_ultralytics"] = u["AP"] - r["mAP50-95"]
        out["heads"][head] = row
        LOGGER.info(
            f"  {head}: unfused {u['AP']:.3f} | fused {f['AP']:.3f} "
            f"({row['fused_minus_unfused_AP']:+.3f}) | Ultralytics "
            f"{r['mAP50-95'] if r else float('nan'):.3f} "
            f"(fused delta {row.get('fused_minus_ultralytics', float('nan')):+.3f})"
        )
        LOGGER.info(f"  {head}: detection count changes by {row['n_dets_delta']:+d} "
                    f"out of {u['n_dets']:,} -- folding is not bit-neutral")

    write_json(args.out, out)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
