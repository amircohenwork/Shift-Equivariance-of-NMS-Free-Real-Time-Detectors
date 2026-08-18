#!/usr/bin/env python3
"""Two committed results disagree by 0.0013 AP on a quantity that should be identical:

    results/oracle/..._AP50_all.json   "baseline"        40.247293   (union table + accumulator)
    results/tta/....json               "baseline_delta0" 40.248626   (cocolite.evaluate_detections)

Both are "AP over all 5000 images with every image at delta=(0,0)" for the same run_id.
`tests/test_oracle.py::test_union_select_equals_from_scratch` pins those two code paths to
1e-12 on synthetic data, so either something differs on real data, or the two runs consumed
different detections -- this project had a known detection-cache contamination incident.

This recomputes both from the detection cache as it stands now. If they agree with each
other, the code paths are fine and the committed pair came from different caches.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import numpy as np

from shifteq.cocolite import (build_matches, class_map_inverse, evaluate_detections,
                              npig_per_cat)
from shifteq.data import load_coco_gt
from shifteq.detstore import DetStore, coco80_to_coco91
from shifteq.oracle import NumpyAccumulator, build_union
from shifteq.utils import LOGGER, setup_logging

setup_logging()

RUN = sys.argv[1] if len(sys.argv) > 1 else "yolo26n_e2e_m1s1_full_pad114"
store = DetStore("cache/dets", RUN)
m = store.manifest
LOGGER.info(f"run_id={m.run_id} head={m.head} canvas={m.canvas} max_shift={m.max_shift}")
LOGGER.info(f"weights_sha256={m.weights_sha256[:16]} fingerprint={m.weight_fingerprint[:16]}")
LOGGER.info(f"shifts available: {sorted(store.available_shifts())}")

from ultralytics.data.utils import check_det_dataset  # noqa: E402

root = Path(check_det_dataset("coco.yaml")["path"])
gt = load_coco_gt(root / "annotations" / "instances_val2017.json", list(m.image_ids))
cmi = class_map_inverse(gt, coco80_to_coco91())
npig = {"all": npig_per_cat(gt, "all")}

d0 = store.load(0, 0)
LOGGER.info(f"delta=(0,0): {d0.n_dets:,} detections")

# path A: the evaluator that Gate B4 validated against faster-coco-eval to 1.1e-16
a = evaluate_detections(gt, d0, cmi, areas=("all",), npig_cache=npig)

# path B: the union table the oracle actually uses, with every image assigned shift 0
grid = sorted(store.available_shifts())
per_shift = [build_matches(gt, store.load(dy, dx), cmi, area_key="all") for dy, dx in grid]
u = build_union(per_shift, npig["all"], gt.num_images, gt.num_cats)
acc = NumpyAccumulator(u)
ap, ap50 = acc.ap(acc.new_mask(np.zeros(gt.num_images, dtype=np.int64)))

LOGGER.info("")
LOGGER.info(f"  cocolite.evaluate_detections : AP={a['AP'] * 100:.6f}  AP50={a['AP50'] * 100:.6f}")
LOGGER.info(f"  union table + accumulator    : AP={ap * 100:.6f}  AP50={ap50 * 100:.6f}")
LOGGER.info(f"  difference                   : {abs(a['AP'] - ap) * 100:.6f} AP")
LOGGER.info("")
LOGGER.info("  committed oracle baseline    : 40.247293")
LOGGER.info("  committed tta  baseline      : 40.248626")
