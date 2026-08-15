"""Instance-level instability: flicker rate and box jitter.

AP is a dataset-level summary and hides the failure mode the proposal is actually about:
a specific object that a 1-px shift makes disappear, or renames, or whose box wobbles.
These metrics are per-ground-truth-object.

Structure: `shift_partial()` is a pure function of one shift's detections, so it can run
in a worker process; `StabilityAccumulator.merge()` folds a partial into running totals
with vectorised ops. Nothing is stored per (GT x shift), so a 32x32 sweep costs O(#GT)
memory, and the expensive part parallelises across the shift grid.

Definitions used here (state them in the paper; they are choices, not standards):

  detected(g, delta)  a de-shifted prediction exists with the same class, IoU >= 0.5,
                      and score >= tau. tau is a real knob: at COCO's evaluation
                      threshold of 0.001 almost nothing ever "drops out", so flicker
                      measured there is meaningless. Default tau = 0.25, the Ultralytics
                      predict-time default.

  flicker             fraction of GT whose detected(g, .) is not constant over the grid.
  dropout             fraction of GT detected at delta=(0,0) that are missed at >= 1 shift.
                      The "your baseline number was partly luck" statistic.
  class flip          fraction of GT assigned more than one distinct class across shifts
                      by their best class-agnostic match.
  jitter              per-GT standard deviation of each de-shifted box coordinate,
                      averaged over the 4 coordinates, in original-image pixels. Also
                      reported normalised by sqrt(GT area), since a 2-px wobble means
                      something very different on a 20-px and a 400-px object.

A perfectly shift-equivariant detector scores 0 on all four.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cocolite import iou_matrix
from .data import AREA_RNG, CocoGT
from .detstore import ShiftDets

SIZE_BUCKETS = ("small", "medium", "large")


def _size_bucket(area: float) -> int:
    if area < AREA_RNG["small"][1]:
        return 0
    if area < AREA_RNG["medium"][1]:
        return 1
    return 2


@dataclass
class GtIndex:
    """Ground truth flattened over images, once, so partials can address GT by integer."""

    image_slices: list[tuple[int, int]]
    cat: np.ndarray       # int16 [G] contiguous category index
    area: np.ndarray      # float32 [G]
    bucket: np.ndarray    # int8 [G]
    valid: np.ndarray     # bool [G] non-crowd

    @property
    def n(self) -> int:
        return len(self.cat)


def build_gt_index(gt: CocoGT) -> GtIndex:
    slices, cats, areas, valid = [], [], [], []
    off = 0
    for iid in gt.image_ids:
        iid = int(iid)
        c = gt.cats[iid]
        slices.append((off, off + len(c)))
        off += len(c)
        cats.append(c.astype(np.int16))
        areas.append(gt.areas[iid])
        valid.append(~gt.iscrowd[iid])
    z = lambda dt: np.zeros(0, dtype=dt)  # noqa: E731
    cat = np.concatenate(cats) if off else z(np.int16)
    area = np.concatenate(areas) if off else z(np.float32)
    return GtIndex(
        image_slices=slices, cat=cat, area=area,
        bucket=np.array([_size_bucket(float(a)) for a in area], dtype=np.int8),
        valid=np.concatenate(valid) if off else z(bool),
    )


@dataclass
class ShiftPartial:
    """One shift's contribution. Small: at most one entry per GT per matching."""

    is_zero: bool
    det_g: np.ndarray      # int64, GT detected (class-aware)
    det_box: np.ndarray    # float64 [n,4] de-shifted boxes of those matches
    det_score: np.ndarray  # float64 [n] confidence of those matches
    agn_g: np.ndarray      # int64, GT matched class-agnostically
    agn_cls: np.ndarray    # int16, predicted class of that match
    shift: tuple[int, int] = (0, 0)


def shift_partial(gt: CocoGT, idx: GtIndex, dets: ShiftDets, class_map_inv: np.ndarray,
                  tau: float = 0.25, iou_thr: float = 0.5) -> ShiftPartial:
    det_g, det_box, det_score, agn_g, agn_cls = [], [], [], [], []
    for i, iid in enumerate(gt.image_ids):
        iid = int(iid)
        g0, g1 = idx.image_slices[i]
        if g1 == g0:
            continue
        gb, gc, gcrowd = gt.boxes[iid], gt.cats[iid], gt.iscrowd[iid]

        dc, ds, db = dets.per_image(i)
        keep = ds >= tau
        if not keep.any():
            continue
        dc, ds, db = dc[keep], ds[keep], db[keep]
        order = np.argsort(-ds.astype(np.float64), kind="mergesort")
        dc, ds, db = dc[order], ds[order], db[order]
        dcat = class_map_inv[dc.astype(np.int64)]

        ious = iou_matrix(db, gb, gcrowd)
        if ious.size == 0:
            continue

        taken = np.zeros(len(gb), bool)
        taken2 = np.zeros(len(gb), bool)
        for d in range(len(db)):
            row = ious[d]
            ok = row >= iou_thr
            if not ok.any():
                continue
            cand = (~taken) & ok & (gc == dcat[d])          # class-aware
            if cand.any():
                g = int(np.where(cand, row, -1.0).argmax())
                taken[g] = True
                det_g.append(g0 + g)
                det_box.append(db[d])
                det_score.append(float(ds[d]))
            cand2 = (~taken2) & ok                          # class-agnostic
            if cand2.any():
                g = int(np.where(cand2, row, -1.0).argmax())
                taken2[g] = True
                agn_g.append(g0 + g)
                agn_cls.append(int(dcat[d]))

    return ShiftPartial(
        is_zero=dets.shift == (0, 0),
        det_g=np.asarray(det_g, dtype=np.int64),
        det_box=np.asarray(det_box, dtype=np.float64).reshape(-1, 4),
        det_score=np.asarray(det_score, dtype=np.float64),
        agn_g=np.asarray(agn_g, dtype=np.int64),
        agn_cls=np.asarray(agn_cls, dtype=np.int16),
        shift=tuple(dets.shift),
    )


class StabilityAccumulator:
    """Running totals over an arbitrary set of shifts.

    Variance is accumulated as sum / sum-of-squares rather than Welford: the coordinates
    are O(10^2) px and there are at most ~10^3 shifts, so float64 has precision to spare,
    and plain sums are associative -- which is what lets partials be computed out of order
    and in parallel.
    """

    def __init__(self, gt: CocoGT, class_map_inv: np.ndarray, tau: float = 0.25, iou_thr: float = 0.5):
        self.gt = gt
        self.idx = build_gt_index(gt)
        self.class_map_inv = class_map_inv
        self.tau = tau
        self.iou_thr = iou_thr
        G = self.idx.n
        self.n_shifts = 0
        self.det_count = np.zeros(G, np.int32)
        self.det_at_zero = np.zeros(G, bool)
        self.cls_first = np.full(G, -1, np.int16)
        self.cls_changed = np.zeros(G, bool)
        self.match_count = np.zeros(G, np.int32)
        self.b_n = np.zeros(G, np.int64)
        self.b_s = np.zeros((G, 4), np.float64)
        self.b_s2 = np.zeros((G, 4), np.float64)

    # ------------------------------------------------------------------
    def update(self, dets: ShiftDets) -> None:
        self.merge(shift_partial(self.gt, self.idx, dets, self.class_map_inv, self.tau, self.iou_thr))

    def merge(self, p: ShiftPartial) -> None:
        self.n_shifts += 1
        g = p.det_g
        if len(g):
            self.det_count[g] += 1
            if p.is_zero:
                self.det_at_zero[g] = True
            self.b_n[g] += 1
            self.b_s[g] += p.det_box
            self.b_s2[g] += p.det_box * p.det_box
        a = p.agn_g
        if len(a):
            first = self.cls_first[a]
            new = first < 0
            self.cls_first[a[new]] = p.agn_cls[new]
            self.cls_changed[a[(~new) & (first != p.agn_cls)]] = True
            self.match_count[a] += 1

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        v = self.idx.valid
        S = self.n_shifts
        out: dict = {"n_shifts": S, "tau": self.tau, "iou_thr": self.iou_thr, "n_gt": int(v.sum())}
        if S == 0 or not v.any():
            return out

        n = np.maximum(self.b_n, 1)[:, None]
        var = np.maximum(self.b_s2 / n - (self.b_s / n) ** 2, 0.0)
        # sample (n-1) correction, matching np.std(ddof=1)
        corr = np.where(self.b_n > 1, self.b_n / np.maximum(self.b_n - 1, 1), 0.0)[:, None]
        jitter_px = np.sqrt(var * corr).mean(axis=1)
        scale = np.sqrt(np.maximum(self.idx.area, 1.0))
        has_var = self.b_n > 1

        def block(sel, prefix=""):
            if not sel.any():
                return {}
            dc = self.det_count[sel]
            flick = (dc > 0) & (dc < S)
            zero = self.det_at_zero[sel]
            drop = zero & (dc < S)
            seen = self.match_count[sel] > 0
            hv = has_var[sel]
            return {
                f"{prefix}n": int(sel.sum()),
                f"{prefix}flicker_rate": float(flick.mean()),
                f"{prefix}dropout_rate": float(drop.sum() / max(int(zero.sum()), 1)),
                f"{prefix}class_flip_rate": float(self.cls_changed[sel][seen].mean()) if seen.any() else 0.0,
                f"{prefix}mean_detect_frac": float((dc / S).mean()),
                f"{prefix}jitter_px": float(jitter_px[sel][hv].mean()) if hv.any() else 0.0,
                f"{prefix}jitter_rel": float((jitter_px[sel] / scale[sel])[hv].mean()) if hv.any() else 0.0,
            }

        out.update(block(v))
        for b, name in enumerate(SIZE_BUCKETS):
            out.update(block(v & (self.idx.bucket == b), prefix=f"{name}_"))
        return out
