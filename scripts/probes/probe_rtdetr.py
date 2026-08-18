#!/usr/bin/env python3
"""Why is RT-DETR-L 0.23 AP below its published 53.0?

A set-prediction detector emits a fixed 300 queries per image with no NMS, so the two knobs
that can move its AP are the confidence floor -- which discards low-scoring queries that
COCO AP still rewards at the tail of the PR curve -- and max_det. Both are protocol rather
than modelling, so a ladder settles whether the residual is a loading bug (which would
invalidate the whole RT-DETR arm) or a threshold choice (which would not).
"""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

g = import_module("01_reproduce_published")

from shifteq.modelzoo import MODELS, resolve_weights  # noqa: E402
from shifteq.utils import LOGGER, setup_logging, write_json  # noqa: E402

setup_logging()
spec = MODELS["rtdetr-l"]
weights = str(resolve_weights(spec))

from ultralytics.data.utils import check_det_dataset  # noqa: E402

ann = str(Path(check_det_dataset("coco.yaml")["path"]) / "annotations" / "instances_val2017.json")

LADDER = (
    ("conf0.001_md300", dict(conf=0.001, max_det=300)),   # what Gate A ran
    ("conf0_md300", dict(conf=0.0, max_det=300)),         # no floor: all 300 queries kept
    ("conf0_md1000", dict(conf=0.0, max_det=1000)),       # max_det above the query count
    ("conf0.001_md100", dict(conf=0.001, max_det=100)),   # COCO's own maxDets
)

out = {}
for label, kw in LADDER:
    r = g.val_one(weights, "e2e", "coco.yaml", 640, 32, "0", 16, ann, spec=spec, **kw)
    ap = r["mAP50-95"]
    out[label] = {**r["coco"], **kw, "delta_vs_published": ap - 53.0}
    LOGGER.info(f"  {label:<18s} AP={ap:.3f}  (published 53.0, delta {ap - 53.0:+.3f})")

best = min(out, key=lambda k: abs(out[k]["delta_vs_published"]))
LOGGER.info(f"closest to the published number: {best} "
            f"({out[best]['delta_vs_published']:+.3f})")
out["closest"] = best
write_json("results/phase0/rtdetr_protocol_probe.json", out)
