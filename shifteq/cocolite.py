"""A COCO bbox-AP evaluator with a cacheable per-image state.

Why not just call pycocotools
-----------------------------
The oracle metric of Manfredi & Wang needs K x (M+1)^2 x N *full-dataset* AP values.
At M=1 on COCO val that is 20,000 evaluations. Each one differs from the last by the
detections of a single image, yet pycocotools would redo the IoU matching for all 5,000
images every time. The matching is the expensive part and it is per-image independent --
so we compute it once per (image, shift) and cache it, leaving only the *accumulation*
(sort by score, cumulative TP/FP, 101-point interpolation) inside the loop.

That is the whole trick: matching is O(N) once, accumulation is O(D) per candidate, and
the 20,000 candidates cost minutes instead of days.

Fidelity
--------
`match_one` is a line-by-line port of `pycocotools.cocoeval.COCOeval.evaluateImg`,
including the two rules that are easy to miss:
  * GT are sorted with non-ignored first, and a detection prefers *any* qualifying
    non-ignored GT over a better-IoU ignored one (the `break` in the original loop);
  * an unmatched detection whose own box area falls outside the area range is *ignored*,
    not counted as a false positive.
`accumulate` reproduces COCOeval.accumulate, including "category with GT but no
detections contributes AP 0" and "category with no GT is excluded entirely".
`tests/test_cocolite_vs_pycocotools.py` asserts agreement to 1e-9 on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import AREA_KEYS, AREA_RNG, CocoGT
from .detstore import ShiftDets

IOU_THRS = np.linspace(0.5, 0.95, 10, dtype=np.float64)
REC_THRS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
T = len(IOU_THRS)
MAXDET = 100  # COCO's maxDets[-1]; also the per-(image,category) truncation


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def iou_matrix(dt: np.ndarray, gt: np.ndarray, iscrowd: np.ndarray) -> np.ndarray:
    """[D,4] x [G,4] xyxy -> [D,G]. Crowd GT use intersection-over-detection-area,
    matching pycocotools' `maskUtils.iou(..., iscrowd)`."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)), dtype=np.float64)
    dt = dt.astype(np.float64)
    gt = gt.astype(np.float64)
    ix1 = np.maximum(dt[:, None, 0], gt[None, :, 0])
    iy1 = np.maximum(dt[:, None, 1], gt[None, :, 1])
    ix2 = np.minimum(dt[:, None, 2], gt[None, :, 2])
    iy2 = np.minimum(dt[:, None, 3], gt[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    da = np.clip(dt[:, 2] - dt[:, 0], 0, None) * np.clip(dt[:, 3] - dt[:, 1], 0, None)
    ga = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    denom = np.where(
        iscrowd[None, :], np.broadcast_to(da[:, None], inter.shape), da[:, None] + ga[None, :] - inter
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, inter / denom, 0.0)
    return out


# --------------------------------------------------------------------------------------
# per-image matching
# --------------------------------------------------------------------------------------
def match_one(
    ious: np.ndarray,        # [D,G], dt already score-sorted desc, gt in original order
    gt_ignore: np.ndarray,   # [G] bool  (iscrowd OR area outside range)
    gt_iscrowd: np.ndarray,  # [G] bool
    dt_out_of_range: np.ndarray,  # [D] bool (detection area outside range)
) -> tuple[np.ndarray, np.ndarray]:
    """Return (matched[T,D] bool, ignored[T,D] bool)."""
    D, G = ious.shape
    matched = np.zeros((T, D), dtype=bool)
    ignored = np.zeros((T, D), dtype=bool)
    if D == 0:
        return matched, ignored
    if G == 0:
        ignored |= dt_out_of_range[None, :]
        return matched, ignored

    # Non-ignored GT first, stable -- this ordering *is* the preference rule.
    gorder = np.argsort(gt_ignore.astype(np.int8), kind="mergesort")
    ig = gt_ignore[gorder]
    crowd = gt_iscrowd[gorder]
    I = ious[:, gorder]

    # Only detections that clear the loosest threshold anywhere can ever match. Skipping
    # the rest is exact and removes ~95% of the sequential work.
    best = I.max(axis=1)
    cand = np.nonzero(best >= IOU_THRS[0] - 1e-12)[0]

    gt_taken = np.zeros((T, G), dtype=bool)
    n_notig = int((~ig).sum())

    for d in cand:
        row = I[d]                                   # [G]
        avail = (~gt_taken) | crowd[None, :]         # [T,G]
        for t in range(T):
            thr = min(IOU_THRS[t], 1 - 1e-10)
            ok = avail[t] & (row >= thr)
            m = -1
            if n_notig and ok[:n_notig].any():
                seg = np.where(ok[:n_notig], row[:n_notig], -1.0)
                m = int(seg.argmax())                # argmax -> first index on ties
            elif ok[n_notig:].any():
                seg = np.where(ok[n_notig:], row[n_notig:], -1.0)
                m = n_notig + int(seg.argmax())
            if m < 0:
                continue
            matched[t, d] = True
            ignored[t, d] = ig[m]
            gt_taken[t, m] = True

    ignored |= ~matched & dt_out_of_range[None, :]
    return matched, ignored


def _bits(x: np.ndarray) -> np.ndarray:
    """[T,D] bool -> [D] uint16 bitmask (bit t = threshold t)."""
    w = (1 << np.arange(T, dtype=np.uint16))[:, None]
    return (x.astype(np.uint16) * w).sum(axis=0, dtype=np.uint16)


def _unbits(b: np.ndarray) -> np.ndarray:
    """[D] uint16 -> [T,D] bool."""
    return ((b[None, :] >> np.arange(T, dtype=np.uint16)[:, None]) & 1).astype(bool)


# --------------------------------------------------------------------------------------
# per-shift match cache
# --------------------------------------------------------------------------------------
@dataclass
class MatchArrays:
    """Flat, one entry per surviving detection, for one shift and one area range."""

    img: np.ndarray     # int32 [D] index into the evaluation image list
    cat: np.ndarray     # int16 [D] contiguous category index
    score: np.ndarray   # float32 [D]
    match: np.ndarray   # uint16 [D] bitmask over IoU thresholds
    ign: np.ndarray     # uint16 [D] bitmask over IoU thresholds

    def __len__(self):
        return len(self.score)


def npig_per_cat(gt: CocoGT, area_key: str) -> np.ndarray:
    """Non-ignored GT count per category. Independent of the shift -- the ground truth
    never moves, only the predictions do -- so the AP denominator is a constant."""
    lo, hi = AREA_RNG[area_key]
    out = np.zeros(gt.num_cats, dtype=np.int64)
    for iid in gt.image_ids:
        iid = int(iid)
        c = gt.cats[iid]
        if len(c) == 0:
            continue
        a = gt.areas[iid]
        ig = gt.iscrowd[iid] | (a < lo) | (a > hi)
        np.add.at(out, c[~ig], 1)
    return out


def build_matches(
    gt: CocoGT,
    dets: ShiftDets,
    class_map_inv: np.ndarray,
    area_key: str = "all",
    maxdet: int = MAXDET,
) -> MatchArrays:
    """Match every detection of one shift against the ground truth.

    `class_map_inv` maps the model's contiguous class index -> the GT's contiguous
    category index (they differ: COCO ids are not dense).
    """
    lo, hi = AREA_RNG[area_key]
    img_l, cat_l, sc_l, m_l, i_l = [], [], [], [], []

    for i, iid in enumerate(gt.image_ids):
        iid = int(iid)
        dc, ds, db = dets.per_image(i)
        gb, gc, ga, gx = gt.boxes[iid], gt.cats[iid], gt.areas[iid], gt.iscrowd[iid]
        if len(dc) == 0:
            continue
        dcat = class_map_inv[dc.astype(np.int64)]
        for k in np.unique(np.concatenate([dcat, gc]) if len(gc) else dcat):
            dsel = np.nonzero(dcat == k)[0]
            if len(dsel) == 0:
                continue
            order = np.argsort(-ds[dsel].astype(np.float64), kind="mergesort")[:maxdet]
            dsel = dsel[order]
            d_box, d_sc = db[dsel], ds[dsel]

            gsel = np.nonzero(gc == k)[0] if len(gc) else np.zeros(0, dtype=np.int64)
            g_box, g_area, g_crowd = gb[gsel], ga[gsel], gx[gsel]
            g_ig = g_crowd | (g_area < lo) | (g_area > hi)

            d_area = (d_box[:, 2] - d_box[:, 0]) * (d_box[:, 3] - d_box[:, 1])
            d_oor = (d_area < lo) | (d_area > hi)

            ious = iou_matrix(d_box, g_box, g_crowd)
            mt, ig = match_one(ious, g_ig, g_crowd, d_oor)

            img_l.append(np.full(len(dsel), i, dtype=np.int32))
            cat_l.append(np.full(len(dsel), k, dtype=np.int16))
            sc_l.append(d_sc.astype(np.float32))
            m_l.append(_bits(mt))
            i_l.append(_bits(ig))

    if not sc_l:
        z = lambda dt: np.zeros(0, dtype=dt)  # noqa: E731
        return MatchArrays(z(np.int32), z(np.int16), z(np.float32), z(np.uint16), z(np.uint16))
    return MatchArrays(
        img=np.concatenate(img_l),
        cat=np.concatenate(cat_l),
        score=np.concatenate(sc_l),
        match=np.concatenate(m_l),
        ign=np.concatenate(i_l),
    )


def canonical_order(m: MatchArrays) -> np.ndarray:
    """Total order used everywhere: category asc, score desc, image asc.

    The image tiebreak makes the ranking independent of concatenation order, so the
    from-scratch evaluator and the oracle's incremental selection agree bit-for-bit.
    pycocotools leaves exact-score ties to mergesort's input order; ties between
    float32 network scores are rare and, when they occur, move AP by <1e-9.
    """
    return np.lexsort((m.img, -m.score.astype(np.float64), m.cat))


# --------------------------------------------------------------------------------------
# accumulation (the inner loop of the oracle)
# --------------------------------------------------------------------------------------
def accumulate(
    cat: np.ndarray,
    match: np.ndarray,
    ign: np.ndarray,
    npig: np.ndarray,
    n_cats: int,
    presorted: bool = True,
    score: np.ndarray | None = None,
    img: np.ndarray | None = None,
) -> dict[str, float]:
    """Precision-recall accumulation over arrays in `canonical_order`.

    Returns AP (0.50:0.95) and AP50. Call once per area range.
    """
    if not presorted:
        assert score is not None and img is not None
        o = np.lexsort((img, -score.astype(np.float64), cat))
        cat, match, ign = cat[o], match[o], ign[o]

    D = len(cat)
    tp = _unbits(match)                       # [T,D]
    valid = ~_unbits(ign)
    tps = tp & valid
    fps = (~tp) & valid

    tp_cum = np.cumsum(tps, axis=1, dtype=np.int64)
    fp_cum = np.cumsum(fps, axis=1, dtype=np.int64)

    starts = np.searchsorted(cat, np.arange(n_cats, dtype=cat.dtype), side="left")
    ends = np.searchsorted(cat, np.arange(n_cats, dtype=cat.dtype), side="right")

    prec = np.full((T, len(REC_THRS), n_cats), -1.0, dtype=np.float64)
    for k in range(n_cats):
        if npig[k] == 0:
            continue
        s, e = int(starts[k]), int(ends[k])
        if e <= s:
            prec[:, :, k] = 0.0            # GT present, nothing predicted -> AP 0
            continue
        base_tp = tp_cum[:, s - 1 : s] if s > 0 else 0
        base_fp = fp_cum[:, s - 1 : s] if s > 0 else 0
        tpc = (tp_cum[:, s:e] - base_tp).astype(np.float64)
        fpc = (fp_cum[:, s:e] - base_fp).astype(np.float64)
        rc = tpc / npig[k]
        pr = tpc / (tpc + fpc + np.spacing(1))
        # monotone-decreasing precision envelope, right to left
        pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
        for t in range(T):
            idx = np.searchsorted(rc[t], REC_THRS, side="left")
            ok = idx < (e - s)
            q = np.zeros(len(REC_THRS))
            q[ok] = pr[t, idx[ok]]
            prec[t, :, k] = q

    sel = prec > -1
    ap = float(prec[sel].mean()) if sel.any() else float("nan")
    sel50 = prec[0] > -1
    ap50 = float(prec[0][sel50].mean()) if sel50.any() else float("nan")
    return {"AP": ap, "AP50": ap50}


# --------------------------------------------------------------------------------------
# convenience: full evaluation of one detection set
# --------------------------------------------------------------------------------------
def evaluate_detections(
    gt: CocoGT,
    dets: ShiftDets,
    class_map_inv: np.ndarray,
    areas: tuple[str, ...] = AREA_KEYS,
    npig_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Standard COCO summary for one detection set: AP, AP50, AP_s/m/l."""
    out: dict[str, float] = {}
    for a in areas:
        m = build_matches(gt, dets, class_map_inv, area_key=a)
        o = canonical_order(m)
        npig = (npig_cache or {}).get(a)
        if npig is None:
            npig = npig_per_cat(gt, a)
        r = accumulate(m.cat[o], m.match[o], m.ign[o], npig, gt.num_cats)
        if a == "all":
            out["AP"] = r["AP"]
            out["AP50"] = r["AP50"]
        else:
            out[f"AP_{a}"] = r["AP"]
    return out


def class_map_inverse(gt: CocoGT, coco91: np.ndarray) -> np.ndarray:
    """model class index (0..79) -> GT contiguous category index."""
    return np.asarray([gt.cat_id_to_index[int(c)] for c in coco91], dtype=np.int64)
