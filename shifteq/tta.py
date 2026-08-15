"""Shift test-time augmentation, and why it is pointed at NMS-free detectors.

Manfredi & Wang Sec. 5.4 turn shift variance into a feature: run the (M+1)^2 shifts, map
every prediction back to original-image coordinates, and merge them with NMS. On COCO they
recover a good part of the gap to AP_best (CenterNet +1.0 AP / +2.2 AP50) for 4x the
inference cost.

That result lands awkwardly on a one-to-one detector, and the awkwardness is a finding:

  * The whole selling point of the end-to-end path is that no NMS runs at inference. But
    merging predictions across shifts is exactly a duplicate-removal problem -- the same
    object is predicted up to (M+1)^2 times. So the NMS-free model can only exploit its own
    shift variance by putting NMS back, at which point the deployment argument for the
    one-to-one head is gone.
  * The alternative merge -- concatenate and keep the top-k by score, with no suppression
    -- is what "NMS-free" would have to mean here, and it should do *badly*, because
    nothing removes the duplicates.

So we implement three merge strategies and report all of them. The comparison
`nms` vs `topk` on the same detections isolates how much of TTA's benefit is duplicate
removal rather than better boxes.
"""

from __future__ import annotations

import numpy as np

from .detstore import ShiftDets


def _nms_per_class(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray,
                   iou_thr: float, max_det: int) -> np.ndarray:
    """Class-wise greedy NMS; returns kept indices, score-sorted."""
    keep: list[int] = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="mergesort")]
        b = boxes[idx]
        area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            keep.append(int(idx[i]))
            if i + 1 == len(idx):
                break
            rest = np.nonzero(alive[i + 1 :])[0] + i + 1
            if not len(rest):
                continue
            ix1 = np.maximum(b[i, 0], b[rest, 0])
            iy1 = np.maximum(b[i, 1], b[rest, 1])
            ix2 = np.minimum(b[i, 2], b[rest, 2])
            iy2 = np.minimum(b[i, 3], b[rest, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            den = area[i] + area[rest] - inter
            iou = np.where(den > 0, inter / den, 0.0)
            alive[rest[iou > iou_thr]] = False
    keep_arr = np.asarray(keep, dtype=np.int64)
    return keep_arr[np.argsort(-scores[keep_arr], kind="mergesort")][:max_det]


def merge_shifts(
    per_shift: list[ShiftDets],
    strategy: str = "nms",
    iou_thr: float = 0.7,
    max_det: int = 300,
) -> ShiftDets:
    """Aggregate several shifts' de-shifted detections into one prediction set per image.

    strategy:
      "nms"   class-wise NMS over the union -- the paper's aggregation.
      "topk"  concatenate and keep the top max_det by score, no suppression. The honest
              "still NMS-free" option; expect it to be worse, and that is the point.
      "vote"  keep only detections whose box is confirmed by a majority of shifts
              (IoU >= iou_thr with a detection of the same class from >= half the shifts),
              then NMS. A stability-aware merge: it drops predictions that only exist at
              some phases of the sampling grid.
    """
    if strategy not in ("nms", "topk", "vote"):
        raise ValueError(strategy)
    n_img = per_shift[0].n_images
    S = len(per_shift)
    counts = np.zeros(n_img, np.int32)
    out_cls, out_score, out_box = [], [], []

    for i in range(n_img):
        cls_l, sc_l, bx_l, sh_l = [], [], [], []
        for s, d in enumerate(per_shift):
            c, sc, b = d.per_image(i)
            if len(c) == 0:
                continue
            cls_l.append(c.astype(np.int64))
            sc_l.append(sc.astype(np.float64))
            bx_l.append(b.astype(np.float64))
            sh_l.append(np.full(len(c), s, np.int64))
        if not cls_l:
            continue
        cls = np.concatenate(cls_l)
        sc = np.concatenate(sc_l)
        bx = np.concatenate(bx_l)
        sh = np.concatenate(sh_l)

        if strategy == "topk":
            k = np.argsort(-sc, kind="mergesort")[:max_det]
        elif strategy == "nms":
            k = _nms_per_class(bx, sc, cls, iou_thr, max_det)
        else:
            k = _nms_per_class(bx, sc, cls, iou_thr, max_det)
            k = k[_support(bx, cls, sh, k, iou_thr, S) * 2 >= S]

        counts[i] = len(k)
        out_cls.append(cls[k].astype(np.int16))
        out_score.append(sc[k].astype(np.float32))
        out_box.append(bx[k].astype(np.float32))

    cat = lambda xs, dt, cols=None: (  # noqa: E731
        np.concatenate(xs) if xs else np.zeros((0, cols) if cols else (0,), dtype=dt)
    )
    return ShiftDets(counts=counts, cls=cat(out_cls, np.int16), score=cat(out_score, np.float32),
                     xyxy=cat(out_box, np.float32, 4), shift=(0, 0))


def _support(bx: np.ndarray, cls: np.ndarray, sh: np.ndarray, keep: np.ndarray,
             iou_thr: float, n_shifts: int) -> np.ndarray:
    """How many distinct shifts back each kept detection."""
    out = np.zeros(len(keep), np.int64)
    a = bx[keep]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(bx[:, 2] - bx[:, 0], 0, None) * np.clip(bx[:, 3] - bx[:, 1], 0, None)
    for j in range(len(keep)):
        same = cls == cls[keep[j]]
        if not same.any():
            continue
        b = bx[same]
        ix1 = np.maximum(a[j, 0], b[:, 0]); iy1 = np.maximum(a[j, 1], b[:, 1])
        ix2 = np.minimum(a[j, 2], b[:, 2]); iy2 = np.minimum(a[j, 3], b[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        den = aa[j] + ab[same] - inter
        iou = np.where(den > 0, inter / den, 0.0)
        out[j] = len(np.unique(sh[same][iou >= iou_thr]))
    return out
