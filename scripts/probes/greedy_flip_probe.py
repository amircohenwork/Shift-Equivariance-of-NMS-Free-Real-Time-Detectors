#!/usr/bin/env python3
"""Did the CUDA division bug change which shift the greedy picked?

Correcting the accumulator moves each AP evaluation by ~0.0013, and that barely moves
Delta-AP because it nearly cancels in a difference. But the greedy *search* compared
candidate shifts using the buggy values, so the assignment it found could in principle
differ from the one a correct search would find. That is a separate question from "are the
reported numbers right", and it is not answered by re-evaluating the committed assignment.

This measures it directly: for each image, evaluate all 4 candidate shifts under the fixed
accumulator and under a deliberately re-broken one (Python-int divisor, the old behaviour),
holding every other image at the committed assignment -- exactly the comparison the greedy
makes. Then count how often the argmax disagrees.

A disagreement rate of 0 means the committed assignment is what a correct search would have
produced, and the corrected Delta-AP is final. Anything above 0 means the oracle should be
re-run with the fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from shifteq.cocolite import REC_THRS, T, build_matches, class_map_inverse, npig_per_cat
from shifteq.data import load_coco_gt
from shifteq.detstore import DetStore, coco80_to_coco91
from shifteq.oracle import R, TorchAccumulator, _EPS, build_union
from shifteq.utils import LOGGER, setup_logging, write_json

setup_logging()
N_PROBE = int(sys.argv[2]) if len(sys.argv) > 2 else 600
HEAD = sys.argv[1] if len(sys.argv) > 1 else "e2e"


class BuggyAccumulator(TorchAccumulator):
    """The pre-fix code path: `tpc / npig` with a Python int divisor, which CUDA lowers to
    a reciprocal multiply."""

    def ap(self, mask):
        torch_ = self.torch
        self.n_calls += 1
        cat = self.cat[mask]
        tp = (self.tp_bits[mask].unsqueeze(0) >> self.bitw) & 1
        ig = (self.ig_bits[mask].unsqueeze(0) >> self.bitw) & 1
        keep = 1 - ig
        tp_cum = torch_.cumsum(tp * keep, dim=1)
        fp_cum = torch_.cumsum((1 - tp) * keep, dim=1)
        starts = torch_.searchsorted(cat, self.k_ar, right=False).tolist()
        ends = torch_.searchsorted(cat, self.k_ar, right=True).tolist()
        tot = tot50 = 0.0
        n = n50 = 0
        for k in range(self.u.n_cats):
            npig = int(self.npig_cpu[k])
            if npig == 0:
                continue
            n += T * R
            n50 += R
            s, e = starts[k], ends[k]
            if e <= s:
                continue
            b_tp = tp_cum[:, s - 1 : s] if s else 0
            b_fp = fp_cum[:, s - 1 : s] if s else 0
            tpc = (tp_cum[:, s:e] - b_tp).double()
            fpc = (fp_cum[:, s:e] - b_fp).double()
            rc = tpc / npig                      # <-- the bug
            pr = tpc / (tpc + fpc + _EPS)
            pr = torch_.cummax(pr.flip(1), dim=1).values.flip(1)
            nd = e - s
            idx = torch_.searchsorted(rc.contiguous(), self.rec.expand(T, R).contiguous(),
                                      right=False)
            ok = (idx < nd).double()
            g = pr.gather(1, idx.clamp(max=nd - 1)) * ok
            tot += float(g.sum())
            tot50 += float(g[0].sum())
        return (tot / n if n else float("nan"), tot50 / n50 if n50 else float("nan"))


run = f"yolo26n_{HEAD}_m1s1_full_pad114"
store = DetStore("cache/dets", run)
m = store.manifest
from ultralytics.data.utils import check_det_dataset  # noqa: E402

root = Path(check_det_dataset("coco.yaml")["path"])
gt = load_coco_gt(root / "annotations" / "instances_val2017.json", list(m.image_ids))
cmi = class_map_inverse(gt, coco80_to_coco91())
npig = npig_per_cat(gt, "all")
grid = sorted(store.available_shifts())
ps = [build_matches(gt, store.load(dy, dx), cmi, area_key="all") for dy, dx in grid]
u = build_union(ps, npig, gt.num_images, gt.num_cats)

fixed = TorchAccumulator(u, device="cuda")
buggy = BuggyAccumulator(u, device="cuda")
assign = np.asarray(np.load(f"results/oracle/{run}_best_assignment.npy"), dtype=np.int64)

rng = np.random.default_rng(0)
probe = rng.choice(gt.num_images, size=min(N_PROBE, gt.num_images), replace=False)
LOGGER.info(f"{run}: probing {len(probe)} images x {len(grid)} shifts, objective AP50 "
            "(the greedy's objective)")

mask_f = fixed.new_mask(assign)
mask_b = buggy.new_mask(assign)
flips = 0
margins = []
for c, i in enumerate(probe, 1):
    i = int(i)
    vf, vb = np.empty(len(grid)), np.empty(len(grid))
    for s in range(len(grid)):
        fixed.set_image(mask_f, i, s)
        buggy.set_image(mask_b, i, s)
        vf[s] = fixed.ap(mask_f)[1]      # AP50 -- what greedy() optimises
        vb[s] = buggy.ap(mask_b)[1]
    fixed.set_image(mask_f, i, int(assign[i]))
    buggy.set_image(mask_b, i, int(assign[i]))
    if int(vf.argmax()) != int(vb.argmax()):
        flips += 1
        LOGGER.info(f"    FLIP image idx {i}: fixed picks {grid[int(vf.argmax())]}, "
                    f"buggy picks {grid[int(vb.argmax())]}")
    top2 = np.sort(vf)[::-1]
    margins.append(top2[0] - top2[1])
    if c % 200 == 0:
        LOGGER.info(f"    {c}/{len(probe)} probed, {flips} flips so far")

margins = np.array(margins)
res = {"run": run, "n_probed": len(probe), "n_flips": int(flips),
       "flip_rate": flips / len(probe),
       "top2_margin_AP50": {"median": float(np.median(margins)),
                            "p05": float(np.quantile(margins, 0.05)),
                            "min": float(margins.min())},
       "bug_magnitude_AP50": 9.6e-6}
LOGGER.info("")
LOGGER.info(f"  argmax disagreements: {flips}/{len(probe)} ({100 * flips / len(probe):.2f}%)")
LOGGER.info(f"  greedy top-2 margin (AP50): median {np.median(margins):.3e}, "
            f"5th pct {np.quantile(margins, 0.05):.3e}, min {margins.min():.3e}")
LOGGER.info(f"  bug moves AP50 by ~9.6e-6 -- compare against the margins above")
write_json(f"results/oracle/{run}_greedy_flip_probe.json", res)
