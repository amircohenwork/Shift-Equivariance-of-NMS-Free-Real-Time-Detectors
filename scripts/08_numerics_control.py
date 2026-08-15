#!/usr/bin/env python3
"""Control arm: how unstable is the detector when the input does NOT change at all?

Phase 0 turned up something that changes how the instance-level metrics have to be read.
Running the *same* image through the *same* weights twice, changing only the arithmetic
(Conv+BN folding, or TF32), leaves confidences almost identical -- but moves **72% of the
surviving boxes**. The reason is that both aggregation heads end in a discrete selection
(NMS for one-to-many, top-k for one-to-one), and a discrete selection is chaotically
sensitive: a 1e-8 score difference flips which of two overlapping candidates survives, and
the survivor can be a materially different box.

So "the box moved when I shifted the image by one pixel" is only evidence of shift
variance if it exceeds "the box moved when I changed nothing but the floating-point
arithmetic". This script measures the latter: it runs delta=(0,0) twice under two numerics
configurations and pushes the pair through exactly the same flicker / jitter / Stability
Index machinery the sweep uses.

Read the output as a floor. Any instance-level instability the sweep reports at or below
these values is arithmetic, not geometry.

(`02f_ap_noise_floor.py` is the corresponding floor for dataset AP. It comes out very
low -- 1% score jitter moves AP by 0.012 -- because AP is a rank statistic over 750k
detections and averages the churn away. The instance-level metrics do not average it away,
which is exactly why they need their own control.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse  # noqa: E402
from shifteq.data import PreprocCache, load_coco_gt, select_subset  # noqa: E402
from shifteq.detstore import ShiftDets, coco80_to_coco91  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner  # noqa: E402
from shifteq.modelzoo import MODELS, load_detector, set_head  # noqa: E402
from shifteq.shift import ShiftGeometry  # noqa: E402
from shifteq.stabindex import SIAccumulator  # noqa: E402
from shifteq.stability import StabilityAccumulator  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402


def run_once(spec, cache, head, device, batch, *, tf32: bool, fuse: bool, canvas_shift: int):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    net, sp = load_detector(spec, device=device)
    set_head(net, head)
    if fuse:
        net.fuse(verbose=False)
    geom = ShiftGeometry(sp.imgsz, canvas_shift, sp.stride)
    return ShiftRunner(net, sp, cache, geom, PostprocCfg(), device, batch).run_shift(0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--heads", default="e2e,nms")
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results/phase0/numerics_control.json")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(args.data)
    root = Path(data["path"])
    ann = Path(args.ann) if args.ann else root / "annotations" / "instances_val2017.json"
    img_dir = Path(args.img_dir) if args.img_dir else root / "images" / "val2017"

    spec = MODELS[args.model]
    gt_all = load_coco_gt(ann)
    ids = select_subset([int(i) for i in gt_all.image_ids], args.subset, args.subset_seed)
    gt = load_coco_gt(ann, ids)
    cmi = class_map_inverse(gt, coco80_to_coco91())
    cache = PreprocCache(Path(args.cache_dir) / "preproc", spec, ids, img_dir).build()

    out: dict = {"model": spec.name, "n_images": gt.num_images, "tau": args.tau, "heads": {}}
    for head in args.heads.split(","):
        LOGGER.info(f"=== {head}: numerics-only control (delta=(0,0) twice) ===")
        a = run_once(spec, cache, head, args.device, args.batch, tf32=False, fuse=False, canvas_shift=0)
        b = run_once(spec, cache, head, args.device, args.batch, tf32=True, fuse=True, canvas_shift=0)
        # relabel the second run as a different "shift" so the pairing logic engages
        b = ShiftDets(counts=b.counts, cls=b.cls, score=b.score, xyxy=b.xyxy, shift=(0, 1))

        stab = StabilityAccumulator(gt, cmi, tau=args.tau)
        si = SIAccumulator(gt, stab.idx, mode="ref")
        for d in (a, b):
            from shifteq.stability import shift_partial

            p = shift_partial(gt, stab.idx, d, cmi, args.tau)
            stab.merge(p)
            si.merge(p)

        row = {"n_dets_a": a.n_dets, "n_dets_b": b.n_dets,
               "stability": stab.summary(), "stability_index": si.summary()}
        out["heads"][head] = row
        s, x = row["stability"], row["stability_index"]
        LOGGER.info(
            f"  {head}: flicker={s['flicker_rate'] * 100:.2f}%  dropout={s['dropout_rate'] * 100:.2f}%  "
            f"class_flip={s['class_flip_rate'] * 100:.2f}%  jitter={s['jitter_px']:.4f} px"
        )
        LOGGER.info(
            f"  {head}: SI={x['SI']:.4f} (conf {x['SI_conf']:.4f}, loc {x['SI_loc']:.4f}, "
            f"ext {x['SI_ext']:.4f})"
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_json(args.out, out)
    LOGGER.info(f"wrote {args.out}")
    LOGGER.info("These are floors: shift-induced instability at or below them is arithmetic.")


if __name__ == "__main__":
    main()
