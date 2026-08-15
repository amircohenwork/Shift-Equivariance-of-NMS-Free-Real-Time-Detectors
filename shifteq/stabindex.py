"""Stability Index, adapted from 3-D tracking to 2-D shift pairs.

Why bother, given we already have flicker and jitter
----------------------------------------------------
Flicker is binary and jitter is a bare standard deviation; neither says *what kind* of
instability a detector has. The Stability Index of "Towards Stable 3D Object Detection"
(arXiv:2407.04305, reference [5] of the proposal, the source of the claim that stability
decorrelates from accuracy) decomposes it, and that decomposition is exactly the axis this
project needs: if the NMS-free head is less stable, is it because the *score* moves (an
aggregation problem, which is the hypothesis), or because the *box* moves (a backbone
sampling problem, shared by both heads and therefore not the head's fault)?

Their definition, for a pair of predictions of the same object:

    SI_c = max(0, 1 - |c1 - c2| / (c^0.99 - c^0.01))        confidence
    SI_l = IoU(B1_loc, B2_loc)                               location only
    SI_e = IoU(B1_ext, B2_ext)                               extent only
    SI_h = ... heading ...                                   (no 2-D analogue)
    SI   = SI_c * (SI_l + SI_e + SI_h) / 3

`B_loc` keeps each prediction's centre but replaces its extent with a shared *pivot*
(the geometric mean of the two ground-truth extents); `B_ext` does the reverse. That is
what decouples "the box slid" from "the box resized". Dropping the heading term, the 2-D
form is

    SI2D = SI_c * (SI_l + SI_e) / 2

and, following the paper, SI = 0 whenever the object is predicted in one member of the
pair but not the other. A missed detection is maximal instability, not missing data.

Pairing: in the original the pair is two consecutive frames. Here it is two shifts. We
support both `ref` (every delta against delta=(0,0)) and `adjacent` (neighbouring grid
points), because the choice changes what the number means -- `ref` measures drift away
from the canonical input, `adjacent` measures local roughness of the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import CocoGT
from .stability import GtIndex, ShiftPartial, SIZE_BUCKETS


def _pivot_extents(gt_boxes: np.ndarray) -> np.ndarray:
    """2-D pivot: the GT extent itself. In the 3-D paper the pivot is the geometric mean
    of the two frames' GT boxes because the object moves between frames; under a pure
    input shift the ground truth is the same object, so the geometric mean collapses to
    the single GT extent."""
    w = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 1e-6)
    h = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 1e-6)
    return np.stack([w, h], axis=1)


def _centres(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def _extents(b: np.ndarray) -> np.ndarray:
    return np.stack([np.maximum(b[:, 2] - b[:, 0], 1e-6),
                     np.maximum(b[:, 3] - b[:, 1], 1e-6)], axis=1)


def _iou_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of two [n,4] xyxy arrays."""
    ix1 = np.maximum(a[:, 0], b[:, 0])
    iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2])
    iy2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    den = aa + bb - inter
    return np.where(den > 0, inter / den, 0.0)


def _box_from(centre: np.ndarray, extent: np.ndarray) -> np.ndarray:
    half = extent / 2
    return np.concatenate([centre - half, centre + half], axis=1)


@dataclass
class _Snapshot:
    """One shift's per-GT prediction state, densified over all GT."""

    matched: np.ndarray   # bool [G]
    score: np.ndarray     # float64 [G]
    box: np.ndarray       # float64 [G,4]


def _densify(p: ShiftPartial, G: int) -> _Snapshot:
    m = np.zeros(G, bool)
    s = np.zeros(G, np.float64)
    b = np.zeros((G, 4), np.float64)
    if len(p.det_g):
        m[p.det_g] = True
        s[p.det_g] = p.det_score
        b[p.det_g] = p.det_box
    return _Snapshot(m, s, b)


@dataclass
class SIAccumulator:
    """Running SI2D over shift pairs. O(#GT) memory regardless of grid size."""

    gt: CocoGT
    idx: GtIndex
    mode: str = "ref"                 # "ref" | "adjacent"
    conf_lo: float | None = None      # c^0.01, frozen from the reference pass
    conf_hi: float | None = None      # c^0.99

    n_pairs: int = 0
    _ref: _Snapshot | None = field(default=None, init=False)
    _prev: _Snapshot | None = field(default=None, init=False)
    _pivot: np.ndarray = field(init=False)
    _gtbox: np.ndarray = field(init=False)
    sum_si: np.ndarray = field(init=False)
    sum_c: np.ndarray = field(init=False)
    sum_l: np.ndarray = field(init=False)
    sum_e: np.ndarray = field(init=False)
    n_obs: np.ndarray = field(init=False)

    def __post_init__(self):
        boxes = []
        for iid in self.gt.image_ids:
            boxes.append(self.gt.boxes[int(iid)])
        self._gtbox = np.concatenate(boxes) if len(boxes) else np.zeros((0, 4), np.float32)
        self._pivot = _pivot_extents(self._gtbox.astype(np.float64))
        G = self.idx.n
        self.sum_si = np.zeros(G, np.float64)
        self.sum_c = np.zeros(G, np.float64)
        self.sum_l = np.zeros(G, np.float64)
        self.sum_e = np.zeros(G, np.float64)
        self.n_obs = np.zeros(G, np.int64)

    # -----------------------------------------------------------------
    def merge(self, p: ShiftPartial) -> None:
        snap = _densify(p, self.idx.n)
        if self._ref is None and p.is_zero:
            self._ref = snap
            if self.conf_lo is None and snap.matched.any():
                s = snap.score[snap.matched]
                self.conf_lo, self.conf_hi = (float(np.quantile(s, 0.01)),
                                              float(np.quantile(s, 0.99)))
        if self.mode == "ref":
            if self._ref is None or p.is_zero:
                self._prev = snap
                return
            self._pair(self._ref, snap)
        else:
            if self._prev is not None:
                self._pair(self._prev, snap)
            self._prev = snap

    def _pair(self, a: _Snapshot, b: _Snapshot) -> None:
        self.n_pairs += 1
        both = a.matched & b.matched
        either = a.matched | b.matched
        if not either.any():
            return

        si = np.zeros(self.idx.n, np.float64)
        sc = np.zeros(self.idx.n, np.float64)
        sl = np.zeros(self.idx.n, np.float64)
        se = np.zeros(self.idx.n, np.float64)

        if both.any():
            piv = self._pivot[both]
            ca, cb = _centres(a.box[both]), _centres(b.box[both])
            ea, eb = _extents(a.box[both]), _extents(b.box[both])
            pc = _centres(self._gtbox[both].astype(np.float64))

            loc = _iou_pairwise(_box_from(ca, piv), _box_from(cb, piv))
            ext = _iou_pairwise(_box_from(pc, ea), _box_from(pc, eb))
            rng = max((self.conf_hi or 1.0) - (self.conf_lo or 0.0), 1e-6)
            conf = np.maximum(0.0, 1.0 - np.abs(a.score[both] - b.score[both]) / rng)

            sc[both] = conf
            sl[both] = loc
            se[both] = ext
            si[both] = conf * (loc + ext) / 2.0
        # objects matched in exactly one member of the pair keep SI = 0, per the paper

        self.sum_si[either] += si[either]
        self.sum_c[either] += sc[either]
        self.sum_l[either] += sl[either]
        self.sum_e[either] += se[either]
        self.n_obs[either] += 1

    # -----------------------------------------------------------------
    def summary(self) -> dict:
        out: dict = {"mode": self.mode, "n_pairs": self.n_pairs,
                     "conf_lo": self.conf_lo, "conf_hi": self.conf_hi}
        seen = (self.n_obs > 0) & self.idx.valid
        if not seen.any():
            return out
        n = np.maximum(self.n_obs, 1)

        def block(sel, prefix=""):
            if not sel.any():
                return {}
            return {
                f"{prefix}n": int(sel.sum()),
                f"{prefix}SI": float((self.sum_si[sel] / n[sel]).mean()),
                f"{prefix}SI_conf": float((self.sum_c[sel] / n[sel]).mean()),
                f"{prefix}SI_loc": float((self.sum_l[sel] / n[sel]).mean()),
                f"{prefix}SI_ext": float((self.sum_e[sel] / n[sel]).mean()),
            }

        out.update(block(seen))
        for b, name in enumerate(SIZE_BUCKETS):
            out.update(block(seen & (self.idx.bucket == b), prefix=f"{name}_"))
        return out
