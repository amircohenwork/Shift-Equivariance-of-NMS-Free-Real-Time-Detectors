#!/usr/bin/env python3
"""Phase-0 follow-up: how each head's COCO AP responds to `max_det`.

Why this exists, and what it found
----------------------------------
Gate A reproduces the published NMS number to 0.03 AP but overshoots the published e2e
number by 0.16. The probe showed e2e at `max_det=100` lands on 40.05 -- suspiciously close
to the published 40.1 -- which suggested the two heads respond to the detection cap
differently: the one-to-one head emits a fixed-size set with no suppression, so every
extra slot is another candidate, whereas NMS has already removed the one-to-many head's
duplicates. If true, the heads could only be compared at a fixed, stated `max_det`.

**The ladder refutes that.** On yolo26n / COCO val:

    max_det      50      100     200     300     500
    e2e       39.643  40.050  40.218  40.255  40.281   (+0.637 over the range)
    nms       40.295  40.685  40.829  40.867  40.888   (+0.593 over the range)

The curves are parallel and both saturate by 300 (300->500 is +0.026 and +0.021). The cap
does not treat the heads differently, and the 40.05 coincidence was just that. `max_det`
is therefore a neutral protocol choice; we keep the Ultralytics val default of 300 for
both heads, and this file stands as the evidence that the choice does not bias the
comparison.

Which leaves the +0.16 on e2e still unexplained -- see 01c_topk_probe.py.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.modelzoo import MODELS, resolve_weights  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402

_A = import_module("01_reproduce_published")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-dets", default="50,100,200,300,500")
    ap.add_argument("--out", default="results/phase0/maxdet_ladder.json")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    spec = MODELS[args.model]
    weights = str(resolve_weights(spec))

    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    mds = [int(x) for x in args.max_dets.split(",")]
    out: dict = {"model": spec.name, "published": spec.published_map, "max_dets": mds, "runs": {}}

    for head in spec.heads:
        for md in mds:
            r = _A.val_one(weights, head, args.data, args.imgsz, args.batch, args.device,
                           args.workers, str(ann), max_det=md)
            c = r["coco"]
            out["runs"][f"{head}/{md}"] = c
            LOGGER.info(
                "{:4s} max_det={:3d}  AP={:.3f}  AP50={:.3f}  AP_s={:.3f}  AP_l={:.3f}".format(
                    head, md, c["mAP50-95"], c["mAP50"], c["mAP_small"], c["mAP_large"]
                )
            )

    # monotone-vs-saturating summary
    for head in spec.heads:
        vals = [out["runs"][f"{head}/{m}"]["mAP50-95"] for m in mds]
        out[f"{head}_gain_50_to_max"] = vals[-1] - vals[0]
        out[f"{head}_gain_100_to_300"] = (
            out["runs"][f"{head}/300"]["mAP50-95"] - out["runs"][f"{head}/100"]["mAP50-95"]
            if 300 in mds and 100 in mds else None
        )
        LOGGER.info(f"{head}: AP({mds[0]})->AP({mds[-1]}) = {vals[0]:.3f} -> {vals[-1]:.3f} "
                    f"({out[f'{head}_gain_50_to_max']:+.3f})")

    write_json(args.out, out)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
