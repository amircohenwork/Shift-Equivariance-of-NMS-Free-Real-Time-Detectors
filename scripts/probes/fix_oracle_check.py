#!/usr/bin/env python3
"""How much does the CUDA accumulator defect move the published Delta-AP?

`TorchAccumulator(device="cuda")` disagrees with `NumpyAccumulator` (and with
`cocolite.evaluate_detections`, which Gate B4 validated against faster-coco-eval to 1.1e-16)
by ~0.0013 AP. torch/cpu agrees exactly, so the defect is in a CUDA kernel path, and the
committed oracle results were produced on CUDA.

Delta-AP is a *difference* of two AP values from the same accumulator, so a constant bias
would cancel. This measures whether it actually does.

The greedy assignments are committed as .npy, so the corrected numbers do not require
re-running the 20,000-call search: evaluate the same assignment with the validated
accumulator. (The search itself was steered by CUDA values, so a numpy-driven greedy could
land on a slightly different assignment -- but AP at a *given* assignment is exactly what
Delta-AP quotes, and any better assignment only widens the envelope.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from shifteq.cocolite import build_matches, class_map_inverse, npig_per_cat
from shifteq.data import load_coco_gt
from shifteq.detstore import DetStore, coco80_to_coco91
from shifteq.oracle import NumpyAccumulator, TorchAccumulator, build_union
from shifteq.utils import LOGGER, setup_logging, write_json

setup_logging()
from ultralytics.data.utils import check_det_dataset  # noqa: E402

root = Path(check_det_dataset("coco.yaml")["path"])
ann = root / "annotations" / "instances_val2017.json"
out = {}

for head in ("e2e", "nms"):
    run = f"yolo26n_{head}_m1s1_full_pad114"
    store = DetStore("cache/dets", run)
    m = store.manifest
    gt = load_coco_gt(ann, list(m.image_ids))
    cmi = class_map_inverse(gt, coco80_to_coco91())
    npig = npig_per_cat(gt, "all")
    grid = sorted(store.available_shifts())
    ps = [build_matches(gt, store.load(dy, dx), cmi, area_key="all") for dy, dx in grid]
    u = build_union(ps, npig, gt.num_images, gt.num_cats)
    N = NumpyAccumulator(u)
    C = TorchAccumulator(u, device="cuda")

    row = {"run": run}
    for name, assign in (
        ("baseline", np.zeros(gt.num_images, dtype=np.int64)),
        ("best", np.load(f"results/oracle/{run}_best_assignment.npy")),
        ("worst", np.load(f"results/oracle/{run}_worst_assignment.npy")),
    ):
        assign = np.asarray(assign, dtype=np.int64)
        n_ap, n_ap50 = N.ap(N.new_mask(assign))
        c_ap, c_ap50 = C.ap(C.new_mask(assign))
        row[name] = {"numpy_AP": n_ap * 100, "cuda_AP": c_ap * 100,
                     "numpy_AP50": n_ap50 * 100, "cuda_AP50": c_ap50 * 100,
                     "AP_error": (c_ap - n_ap) * 100, "AP50_error": (c_ap50 - n_ap50) * 100}
        LOGGER.info(f"  {head}/{name:8s} numpy AP={n_ap * 100:.6f} cuda AP={c_ap * 100:.6f} "
                    f"(cuda error {(c_ap - n_ap) * 100:+.6f})")

    for k, lo, hi in (("dAP", "worst", "best"), ("dAP50", "worst", "best")):
        f = "numpy_AP" if k == "dAP" else "numpy_AP50"
        g = "cuda_AP" if k == "dAP" else "cuda_AP50"
        row[k] = {"corrected": row[hi][f] - row[lo][f], "as_published": row[hi][g] - row[lo][g]}
        row[k]["shift"] = row[k]["corrected"] - row[k]["as_published"]
        LOGGER.info(f"  {head}/{k}: published {row[k]['as_published']:.4f} -> "
                    f"corrected {row[k]['corrected']:.4f} ({row[k]['shift']:+.4f})")
    out[head] = row
    LOGGER.info("")

d_pub = out["e2e"]["dAP"]["as_published"] - out["nms"]["dAP"]["as_published"]
d_cor = out["e2e"]["dAP"]["corrected"] - out["nms"]["dAP"]["corrected"]
r_pub = out["e2e"]["dAP"]["as_published"] / out["nms"]["dAP"]["as_published"]
r_cor = out["e2e"]["dAP"]["corrected"] / out["nms"]["dAP"]["corrected"]
LOGGER.info(f"HEADLINE  e2e-minus-nms dAP: published {d_pub:.4f} -> corrected {d_cor:.4f}")
LOGGER.info(f"HEADLINE  ratio e2e/nms:     published {r_pub:.4f} -> corrected {r_cor:.4f} "
            f"({(r_pub - 1) * 100:.1f}% -> {(r_cor - 1) * 100:.1f}%)")
out["headline"] = {"gap_published": d_pub, "gap_corrected": d_cor,
                   "ratio_published": r_pub, "ratio_corrected": r_cor}
write_json("results/oracle/cuda_accumulator_correction.json", out)
LOGGER.info("wrote results/oracle/cuda_accumulator_correction.json")
