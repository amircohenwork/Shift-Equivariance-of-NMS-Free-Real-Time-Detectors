"""Flicker/jitter, the Stability Index adaptation, and shift-TTA merging.

These metrics have no reference implementation to check against, so the tests pin the
properties that must hold by construction: a perfectly equivariant detector scores 0
instability and SI = 1, a detector that drops an object scores SI = 0 for it, and NMS
merging must actually remove the duplicates that `topk` keeps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.data import load_coco_gt  # noqa: E402
from shifteq.detstore import ShiftDets  # noqa: E402
from shifteq.stabindex import SIAccumulator  # noqa: E402
from shifteq.stability import StabilityAccumulator, shift_partial  # noqa: E402
from shifteq.tta import merge_shifts  # noqa: E402
from synth import make_gt_json  # noqa: E402

N_CAT = 8


def _perfect_dets(gt, shift=(0, 0), score=0.9, jitter=0.0, seed=0, drop=None):
    """Detections that reproduce the ground truth exactly (optionally perturbed)."""
    rng = np.random.default_rng(seed)
    counts, cls, sc, bx = [], [], [], []
    for i, iid in enumerate(gt.image_ids):
        iid = int(iid)
        b = gt.boxes[iid].astype(np.float64).copy()
        c = gt.cats[iid]
        keep = np.ones(len(c), bool)
        if drop is not None and len(c):
            keep[drop % max(len(c), 1) :: max(len(c), 1)] = False
            keep = np.ones(len(c), bool)
            keep[drop % len(c)] = False
        b, c = b[keep], c[keep]
        if jitter:
            b = b + rng.normal(0, jitter, b.shape)
        counts.append(len(c))
        cls.append(c.astype(np.int16))
        sc.append(np.full(len(c), score, np.float32))
        bx.append(b.astype(np.float32))
    cat = lambda xs, dt, n=None: (np.concatenate(xs) if any(len(x) for x in xs)  # noqa: E731
                                  else np.zeros((0, n) if n else (0,), dt))
    return ShiftDets(counts=np.array(counts, np.int32), cls=cat(cls, np.int16),
                     score=cat(sc, np.float32), xyxy=cat(bx, np.float32, 4), shift=shift)


@pytest.fixture(scope="module")
def gt(tmp_path_factory):
    ann, ids, _ = make_gt_json(tmp_path_factory.mktemp("stab") / "gt.json",
                              n_img=15, n_cat=N_CAT, seed=5, crowd_frac=0.0)
    return load_coco_gt(ann, ids)


def test_perfect_detector_is_perfectly_stable(gt):
    cmi = np.arange(N_CAT, dtype=np.int64)
    acc = StabilityAccumulator(gt, cmi, tau=0.5)
    si = SIAccumulator(gt, acc.idx, mode="ref")
    for s in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        d = _perfect_dets(gt, shift=s)
        acc.update(d)
        si.merge(shift_partial(gt, acc.idx, d, cmi, tau=0.5))

    r = acc.summary()
    assert r["flicker_rate"] == 0.0
    assert r["dropout_rate"] == 0.0
    assert r["class_flip_rate"] == 0.0
    assert r["jitter_px"] == pytest.approx(0.0, abs=1e-9)
    assert r["mean_detect_frac"] == pytest.approx(1.0)

    s = si.summary()
    assert s["SI"] == pytest.approx(1.0, abs=1e-9)
    assert s["SI_loc"] == pytest.approx(1.0, abs=1e-9)
    assert s["SI_ext"] == pytest.approx(1.0, abs=1e-9)


def test_jitter_is_measured_in_pixels(gt):
    cmi = np.arange(N_CAT, dtype=np.int64)
    acc = StabilityAccumulator(gt, cmi, tau=0.5)
    for k, s in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        acc.update(_perfect_dets(gt, shift=s, jitter=3.0, seed=k))
    r = acc.summary()
    assert r["jitter_px"] > 0.3, r["jitter_px"]
    assert r["jitter_rel"] > 0.0


def test_dropout_and_flicker_detected(gt):
    cmi = np.arange(N_CAT, dtype=np.int64)
    acc = StabilityAccumulator(gt, cmi, tau=0.5)
    si = SIAccumulator(gt, acc.idx, mode="ref")
    for k, s in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        d = _perfect_dets(gt, shift=s, drop=None if s == (0, 0) else 0)
        acc.update(d)
        si.merge(shift_partial(gt, acc.idx, d, cmi, tau=0.5))
    r = acc.summary()
    assert r["flicker_rate"] > 0.0
    assert r["dropout_rate"] > 0.0
    # objects present at delta=0 but missing later must drag SI strictly below 1
    assert si.summary()["SI"] < 1.0


def test_si_confidence_term_reacts_to_score_swings(gt):
    cmi = np.arange(N_CAT, dtype=np.int64)
    acc = StabilityAccumulator(gt, cmi, tau=0.1)
    si = SIAccumulator(gt, acc.idx, mode="ref", conf_lo=0.0, conf_hi=1.0)
    si.merge(shift_partial(gt, acc.idx, _perfect_dets(gt, (0, 0), score=0.9), cmi, tau=0.1))
    for s in [(0, 1), (1, 0)]:
        si.merge(shift_partial(gt, acc.idx, _perfect_dets(gt, s, score=0.4), cmi, tau=0.1))
    r = si.summary()
    assert r["SI_loc"] == pytest.approx(1.0, abs=1e-9)   # boxes never moved
    assert r["SI_conf"] == pytest.approx(0.5, abs=1e-6)  # |0.9-0.4| / (1-0) -> 1-0.5
    assert r["SI"] == pytest.approx(0.5, abs=1e-6)


def test_tta_nms_removes_duplicates_that_topk_keeps(gt):
    """The comparison the NMS-free story hinges on."""
    per_shift = [_perfect_dets(gt, shift=s, jitter=0.5, seed=k)
                 for k, s in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]
    union = sum(d.n_dets for d in per_shift)
    nms = merge_shifts(per_shift, "nms", iou_thr=0.7, max_det=300)
    topk = merge_shifts(per_shift, "topk", iou_thr=0.7, max_det=300)
    vote = merge_shifts(per_shift, "vote", iou_thr=0.7, max_det=300)

    assert topk.n_dets == union, "topk must keep everything below max_det"
    assert nms.n_dets < topk.n_dets, "NMS must collapse the 4x duplicates"
    assert nms.n_dets == pytest.approx(per_shift[0].n_dets, rel=0.25)
    assert vote.n_dets <= nms.n_dets
    for m in (nms, topk, vote):
        assert m.counts.sum() == len(m.score) == len(m.cls) == len(m.xyxy)


def test_tta_topk_respects_max_det(gt):
    per_shift = [_perfect_dets(gt, shift=s) for s in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    m = merge_shifts(per_shift, "topk", max_det=2)
    assert m.counts.max() <= 2
