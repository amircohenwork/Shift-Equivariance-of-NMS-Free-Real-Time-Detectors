#!/usr/bin/env python3
"""Last stop on Gate A's residual: does the *letterbox protocol* explain it?

max_det (01b), confidence, precision, and the exact-vs-grouped top-k selection (01c) have
all been ruled out. The remaining free variable in the published protocol is `rect`:
with `rect=False` every image is letterboxed into a square 640; with `rect=True` the
dataloader batches images of similar aspect ratio and pads only to the batch's shape, so
each image keeps more resolution and less gray border.

`ultralytics/cfg/default.yaml` documents `rect: False` but comments "rectangular batching
for val when mode='val'", so it is genuinely ambiguous which one the table used.

This matters beyond the gate: our shift harness *must* use square letterboxing (a
per-batch shape would make the canvas size depend on which images share a batch), so if
`rect=True` turns out to be the published protocol we need to say that our baseline is
the square one and quote the difference.
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
    ap.add_argument("--out", default="results/phase0/protocol_probe.json")
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

    out = {"model": spec.name, "published": spec.published_map, "runs": {}}
    for head in spec.heads:
        for rect in (False, True):
            r = _A.val_one(weights, head, args.data, args.imgsz, args.batch, args.device,
                           args.workers, str(ann), rect=rect)
            key = f"{head}/rect={rect}"
            out["runs"][key] = r["coco"]
            tgt = spec.published_map.get(head)
            delta = r["coco"]["mAP50-95"] - tgt if tgt else float("nan")
            out["runs"][key]["delta_vs_published"] = delta
            LOGGER.info("{:20s} AP={:.3f}  AP50={:.3f}  published={}  delta={:+.3f}".format(
                key, r["coco"]["mAP50-95"], r["coco"]["mAP50"], tgt, delta))

    write_json(args.out, out)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
