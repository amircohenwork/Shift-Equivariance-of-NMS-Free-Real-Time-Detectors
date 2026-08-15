"""Manfredi & Wang's oracle envelope: AP_best, AP_worst, dAP.

What this metric actually is
----------------------------
dAP is **not** the spread of AP over uniform shifts. AP_best is the largest dataset AP
obtainable when the shift is chosen *per image, adversarially in our favour*; AP_worst
is the smallest. They are the two ends of an envelope over (M+1)^(2N) assignments, not
two points on a curve. Because AP couples images through a single global score ranking,
the optimum cannot be found per image, so the paper approximates it with Algorithm 1:
sweep images in order and, for each, take the shift that maximises (or minimises)
dataset AP with every other image's shift held fixed.

Cost is K x (M+1)^2 x N *full-dataset* AP evaluations -- 20,000 at M=1 on COCO val, with
K=1 sufficient (the paper measures <1% further movement from later iterations). Every one
of those is CPU/GPU accumulation work on cached matches, not a forward pass. This, not
the inference, is the budget item that decides whether the experiment finishes.

The selection objective is AP50, not AP
---------------------------------------
Easy to get wrong, and it changes the numbers. Sec. 3.1: "In practice, we use AP50 as the
metric, that is, using 0.5 as the IoU threshold". The greedy search maximises/minimises
**AP50**; the AP (0.50:0.95) column of their Table 1 is AP evaluated *at the assignment
AP50 chose*, not the result of a separate AP-driven search. The paper confirms this
side-effect explicitly in Sec. 4.1: dAP50 grows monotonically with max shift but dAP does
not, "a result of our evaluation metric, that uses AP50 to find best and worst shifts".
`greedy()` therefore reports both at the final assignment, and `--objective` defaults to
AP50. Switching it to AP produces a legitimate but *different* metric that cannot be put
next to their table.

Use `sweep.py` for the uniform-shift curve AP(delta). The two families answer different
questions and must never be collapsed into one number.

Reference values, COCO val, M=1 (their Table 1):
    RetinaNet    AP 36.5  worst/best 35.3/37.5  dAP 2.2 | AP50 56.7  53.9/59.0  dAP50 5.1
    FasterRCNN   AP 37.6  worst/best 36.5/39.4  dAP 2.9 | AP50 59.0  55.7/62.1  dAP50 6.4
    CenterNet    AP 34.6  worst/best 32.9/36.3  dAP 3.4 | AP50 53.0  49.2/57.3  dAP50 8.1
CenterNet is the only one of the three without NMS -- it max-pools heatmap peaks -- and it
is the least stable. That is the closest prior data point to our question, and it is a
weak one: a max-pool peak picker is not one-to-one set prediction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cocolite import REC_THRS, T, MatchArrays
from .data import AREA_KEYS, CocoGT
from .utils import LOGGER, write_json

R = len(REC_THRS)
_EPS = np.spacing(1)


# --------------------------------------------------------------------------------------
# the union table
# --------------------------------------------------------------------------------------
@dataclass
class UnionTable:
    """All (image, shift) detections of one area range, in one globally sorted table.

    Sorting once over the union -- rather than once per candidate -- removes the
    O(D log D) term from the inner loop. Selecting an assignment is then a boolean mask
    over a table that is already in the right order, and the mask only has to be edited
    where the assignment actually changed.
    """

    cat: np.ndarray      # int16  [U] sorted asc; within a category, score desc
    match: np.ndarray    # uint16 [U] bitmask over the 10 IoU thresholds
    ign: np.ndarray      # uint16 [U]
    img: np.ndarray      # int32  [U]
    shift: np.ndarray    # int16  [U] index into the shift list
    npig: np.ndarray     # int64  [K]
    n_images: int
    n_shifts: int
    rows_of_image: list[np.ndarray] = field(default_factory=list)

    @property
    def n_cats(self) -> int:
        return len(self.npig)


def build_union(per_shift: list[MatchArrays], npig: np.ndarray, n_images: int, n_cats: int) -> UnionTable:
    S = len(per_shift)
    cat = np.concatenate([m.cat for m in per_shift])
    match = np.concatenate([m.match for m in per_shift])
    ign = np.concatenate([m.ign for m in per_shift])
    img = np.concatenate([m.img for m in per_shift])
    score = np.concatenate([m.score.astype(np.float64) for m in per_shift])
    shift = np.concatenate([np.full(len(m), s, dtype=np.int16) for s, m in enumerate(per_shift)])

    # Total order: category, then score desc, then image, then shift. The last two keys
    # only matter for exact score ties, but pinning them keeps every candidate ranking
    # reproducible regardless of which shifts happen to be selected.
    o = np.lexsort((shift, img, -score, cat))
    cat, match, ign, img, shift = cat[o], match[o], ign[o], img[o], shift[o]

    order_by_img = np.argsort(img, kind="stable")
    bounds = np.searchsorted(img[order_by_img], np.arange(n_images + 1, dtype=img.dtype))
    rows_of_image = [order_by_img[bounds[i] : bounds[i + 1]] for i in range(n_images)]

    LOGGER.info(f"union table: {len(cat):,} rows | {S} shifts | {n_images} images | {n_cats} cats")
    return UnionTable(cat, match, ign, img, shift, npig, n_images, S, rows_of_image)


# --------------------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------------------
class NumpyAccumulator:
    """Reference implementation. Correct, ~0.1-0.5 s per call on full COCO val."""

    name = "numpy"

    def __init__(self, u: UnionTable):
        self.u = u
        # int32 throughout: the cumulative TP/FP counts cannot exceed the number of
        # detections, and halving the width roughly halves both the peak footprint
        # (~10 x D elements per intermediate) and the memory-bandwidth-bound runtime.
        self.cat = u.cat.astype(np.int32)
        self.tp_bits = u.match.astype(np.int32)
        self.ig_bits = u.ign.astype(np.int32)
        self.shift = u.shift.astype(np.int32)
        self.img = u.img.astype(np.int64)
        self.bitw = np.arange(T, dtype=np.int32)[:, None]
        self.k_ar = np.arange(u.n_cats, dtype=np.int32)
        self.n_calls = 0

    def new_mask(self, assign: np.ndarray) -> np.ndarray:
        return assign.astype(np.int32)[self.img] == self.shift

    def set_image(self, mask: np.ndarray, i: int, s: int) -> np.ndarray:
        rows = self.u.rows_of_image[i]
        mask[rows] = self.shift[rows] == s
        return mask

    def ap(self, mask) -> tuple[float, float]:
        self.n_calls += 1
        cat = self.cat[mask]
        tp = (self.tp_bits[mask][None, :] >> self.bitw) & 1
        ig = (self.ig_bits[mask][None, :] >> self.bitw) & 1
        keep = 1 - ig
        tp_cum = np.cumsum(tp * keep, axis=1, dtype=np.int32)
        fp_cum = np.cumsum((1 - tp) * keep, axis=1, dtype=np.int32)
        starts = np.searchsorted(cat, self.k_ar, side="left")
        ends = np.searchsorted(cat, self.k_ar, side="right")

        tot = tot50 = 0.0
        n = n50 = 0
        for k in range(self.u.n_cats):
            npig = int(self.u.npig[k])
            if npig == 0:
                continue
            n += T * R
            n50 += R
            s, e = int(starts[k]), int(ends[k])
            if e <= s:
                continue  # GT present, nothing predicted -> contributes zeros
            b_tp = tp_cum[:, s - 1 : s] if s else 0
            b_fp = fp_cum[:, s - 1 : s] if s else 0
            tpc = (tp_cum[:, s:e] - b_tp).astype(np.float64)
            fpc = (fp_cum[:, s:e] - b_fp).astype(np.float64)
            rc = tpc / npig
            pr = tpc / (tpc + fpc + _EPS)
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            nd = e - s
            for t in range(T):
                idx = np.searchsorted(rc[t], REC_THRS, side="left")
                ok = idx < nd
                q = float(pr[t, np.minimum(idx, nd - 1)][ok].sum())
                tot += q
                if t == 0:
                    tot50 += q
        return (tot / n if n else float("nan"), tot50 / n50 if n50 else float("nan"))


class TorchAccumulator:
    """Same arithmetic on the GPU. The 20,000-candidate greedy pass goes from hours to
    minutes here; the per-category Python loop survives only because 80 small kernel
    launches are still cheaper than padding 80 ragged segments into one dense tensor."""

    name = "torch"

    def __init__(self, u: UnionTable, device: str = "cuda"):
        import torch

        from .modelzoo import resolve_device

        self.torch = torch
        self.dev = resolve_device(device)
        self.u = u
        t = lambda a, d: torch.as_tensor(np.ascontiguousarray(a)).to(self.dev, d)  # noqa: E731
        # int32 for the same reason as the numpy path; on GPU it also doubles the number
        # of elements that fit in a cache line for the cumsum, which dominates.
        self.cat = t(u.cat, torch.int32)
        self.tp_bits = t(u.match, torch.int32)
        self.ig_bits = t(u.ign, torch.int32)
        self.shift = t(u.shift, torch.int32)
        self.img = t(u.img, torch.int64)
        self.npig = t(u.npig, torch.int64)
        self.bitw = torch.arange(T, device=self.dev, dtype=torch.int32)[:, None]
        self.k_ar = torch.arange(u.n_cats, device=self.dev, dtype=torch.int32)
        self.rec = torch.as_tensor(REC_THRS).to(self.dev, torch.float64)
        self.rows_of_image = [t(r, torch.int64) for r in u.rows_of_image]
        self.npig_cpu = u.npig
        self.n_calls = 0

    def new_mask(self, assign: np.ndarray):
        a = self.torch.as_tensor(assign.astype(np.int64)).to(self.dev)
        return a[self.img] == self.shift

    def set_image(self, mask, i: int, s: int):
        rows = self.rows_of_image[i]
        mask[rows] = self.shift[rows] == s
        return mask

    def ap(self, mask) -> tuple[float, float]:
        torch = self.torch
        self.n_calls += 1
        cat = self.cat[mask]
        tp = (self.tp_bits[mask].unsqueeze(0) >> self.bitw) & 1
        ig = (self.ig_bits[mask].unsqueeze(0) >> self.bitw) & 1
        keep = 1 - ig
        tp_cum = torch.cumsum(tp * keep, dim=1)
        fp_cum = torch.cumsum((1 - tp) * keep, dim=1)
        starts = torch.searchsorted(cat, self.k_ar, right=False).tolist()
        ends = torch.searchsorted(cat, self.k_ar, right=True).tolist()

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
            rc = tpc / npig
            pr = tpc / (tpc + fpc + _EPS)
            pr = torch.cummax(pr.flip(1), dim=1).values.flip(1)
            nd = e - s
            idx = torch.searchsorted(rc.contiguous(), self.rec.expand(T, R).contiguous(), right=False)
            ok = (idx < nd).double()
            g = pr.gather(1, idx.clamp(max=nd - 1)) * ok
            tot += float(g.sum())
            tot50 += float(g[0].sum())
        return (tot / n if n else float("nan"), tot50 / n50 if n50 else float("nan"))


def make_accumulator(u: UnionTable, backend: str = "auto", device: str = "cuda"):
    if backend == "auto":
        try:
            import torch

            backend = "torch" if torch.cuda.is_available() else "numpy"
        except Exception:
            backend = "numpy"
    if backend == "torch":
        return TorchAccumulator(u, device)
    return NumpyAccumulator(u)


# --------------------------------------------------------------------------------------
# Algorithm 1
# --------------------------------------------------------------------------------------
@dataclass
class OracleResult:
    objective: str
    direction: str                 # "best" | "worst"
    ap: float
    ap50: float
    assignment: list[int]
    shifts: list[list[int]]
    iterations: list[dict]
    n_ap_calls: int
    seconds: float
    backend: str


def greedy(
    acc,
    shifts: list[tuple[int, int]],
    direction: str = "best",
    objective: str = "AP",
    iters: int = 1,
    image_order: np.ndarray | None = None,
    checkpoint: Path | None = None,
    log_every: int = 200,
) -> OracleResult:
    """Algorithm 1 of Manfredi & Wang, for one direction."""
    if direction not in ("best", "worst"):
        raise ValueError(direction)
    if objective not in ("AP", "AP50"):
        raise ValueError(objective)
    obj_idx = 0 if objective == "AP" else 1
    n_img, n_shift = acc.u.n_images, acc.u.n_shifts
    if n_shift != len(shifts):
        raise ValueError(f"union has {n_shift} shifts, {len(shifts)} given")

    assign = np.zeros(n_img, dtype=np.int64)   # delta*_i = (0,0), the paper's init
    mask = acc.new_mask(assign)
    t0 = time.perf_counter()
    iterations = []
    order = np.arange(n_img) if image_order is None else np.asarray(image_order)

    for k in range(iters):
        changed = 0
        for step, i in enumerate(order):
            i = int(i)
            vals = np.empty(n_shift)
            for s in range(n_shift):
                acc.set_image(mask, i, s)
                vals[s] = acc.ap(mask)[obj_idx]
            pick = int(vals.argmax() if direction == "best" else vals.argmin())
            changed += pick != int(assign[i])
            assign[i] = pick
            acc.set_image(mask, i, pick)

            if log_every and (step + 1) % log_every == 0:
                el = time.perf_counter() - t0
                done = k * n_img + step + 1
                tot = iters * n_img
                LOGGER.info(
                    f"  greedy[{direction}/{objective}] it{k} {step + 1}/{n_img} "
                    f"{objective}={vals[pick]:.5f} changed={changed} "
                    f"{el / 60:.1f}m elapsed, eta {el / done * (tot - done) / 60:.1f}m"
                )
                if checkpoint:
                    write_json(checkpoint, {"iter": k, "step": step, "assign": assign.tolist()})

        ap, ap50 = acc.ap(mask)
        iterations.append({"iter": k, "changed": int(changed), "AP": ap, "AP50": ap50})
        LOGGER.info(f"  greedy[{direction}/{objective}] it{k}: AP={ap:.5f} AP50={ap50:.5f} changed={changed}")
        if changed == 0:
            break

    ap, ap50 = acc.ap(mask)
    return OracleResult(
        objective=objective, direction=direction, ap=ap, ap50=ap50,
        assignment=assign.tolist(), shifts=[list(s) for s in shifts],
        iterations=iterations, n_ap_calls=acc.n_calls,
        seconds=time.perf_counter() - t0, backend=acc.name,
    )


def delta_ap(best: OracleResult, worst: OracleResult) -> dict:
    return {
        "AP_best": best.ap, "AP_worst": worst.ap, "dAP": best.ap - worst.ap,
        "AP50_best": best.ap50, "AP50_worst": worst.ap50, "dAP50": best.ap50 - worst.ap50,
    }


def assignment_histogram(res: OracleResult) -> dict[str, int]:
    """Which shifts the oracle actually picks.

    A flat histogram means the response to a 1-px shift is essentially image-specific
    noise. A peak means a systematic preferred sub-stride phase -- a far more interesting
    and actionable finding, and one worth checking before writing the discussion.
    """
    h = np.bincount(np.asarray(res.assignment), minlength=len(res.shifts))
    return {str(tuple(s)): int(c) for s, c in zip(res.shifts, h)}


def summary_at_assignment(
    gt: CocoGT,
    per_shift_by_area: dict[str, list[MatchArrays]],
    assignment: np.ndarray,
    npig_by_area: dict[str, np.ndarray],
) -> dict[str, float]:
    """Full COCO summary (AP, AP50, AP_s/m/l) at a fixed per-image shift assignment.

    This is a *secondary* read-out of the assignment that optimised overall AP. It is
    deliberately NOT an oracle for the size-conditional metrics: an oracle envelope for
    small objects means running `greedy` again against the small-area union table, at the
    cost of another full K x S x N pass, and must be reported as its own number.
    """
    from .cocolite import accumulate

    out: dict[str, float] = {}
    for a in AREA_KEYS:
        per_shift = per_shift_by_area[a]
        parts = {k: [] for k in ("cat", "match", "ign", "score", "img")}
        for i in range(len(assignment)):
            m = per_shift[int(assignment[i])]
            sel = m.img == i
            parts["cat"].append(m.cat[sel]); parts["match"].append(m.match[sel])
            parts["ign"].append(m.ign[sel]); parts["score"].append(m.score[sel])
            parts["img"].append(m.img[sel])
        c = {k: np.concatenate(v) for k, v in parts.items()}
        r = accumulate(c["cat"], c["match"], c["ign"], npig_by_area[a], gt.num_cats,
                       presorted=False, score=c["score"], img=c["img"])
        if a == "all":
            out["AP"], out["AP50"] = r["AP"], r["AP50"]
        else:
            out[f"AP_{a}"] = r["AP"]
    return out


def estimate_cost(n_images: int, n_shifts: int, iters: int, seconds_per_ap: float) -> dict:
    calls = iters * n_shifts * n_images
    return {"ap_calls": calls, "seconds_per_ap": seconds_per_ap, "hours": calls * seconds_per_ap / 3600.0}
