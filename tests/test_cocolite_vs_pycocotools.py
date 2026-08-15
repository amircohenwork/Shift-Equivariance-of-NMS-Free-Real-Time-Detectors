"""The evaluator must agree with the reference implementation to machine precision.

Everything in this project -- the sweep curve, the oracle envelope, the head comparison --
is a difference of AP values, often in the third decimal. An evaluator that is merely
"close" to COCOeval would put a systematic bias exactly where the signal is.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.cocolite import evaluate_detections  # noqa: E402
from shifteq.data import load_coco_gt  # noqa: E402
from shifteq.detstore import to_coco_json  # noqa: E402
from synth import make_dets, make_gt_json  # noqa: E402

N_CAT = 12


def _reference(ann_json, preds, image_ids):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        pj = f.name
    errs = []
    for name in ("faster_coco_eval", "pycocotools"):
        try:
            if name == "faster_coco_eval":
                from faster_coco_eval import COCO, COCOeval_faster as E
            else:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval as E
            anno = COCO(str(ann_json))
            pred = anno.loadRes(pj)
            ev = E(anno, pred, iouType="bbox")
            ev.params.imgIds = list(image_ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            s = ev.stats
            return name, {"AP": float(s[0]), "AP50": float(s[1]), "AP_small": float(s[3]),
                          "AP_medium": float(s[4]), "AP_large": float(s[5])}
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {e}")
    pytest.skip("no reference COCO evaluator available (" + "; ".join(errs) + ")")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(tmp_path, seed):
    ann, ids, cat_ids = make_gt_json(tmp_path / "gt.json", n_img=40, n_cat=N_CAT, seed=seed)
    dets = make_dets(ann, ids, N_CAT, seed=seed + 100)

    gt = load_coco_gt(ann, ids)
    cmi = np.arange(N_CAT, dtype=np.int64)          # model class i  -> GT cat index i
    ours = evaluate_detections(gt, dets, cmi)

    preds = to_coco_json(dets, ids, cat_ids, round_coords=False)
    name, ref = _reference(ann, preds, ids)

    for k in ref:
        assert abs(ours[k] - ref[k]) < 1e-9, (
            f"{k}: ours={ours[k]:.12f} {name}={ref[k]:.12f} delta={ours[k] - ref[k]:.3e}"
        )


def test_empty_predictions(tmp_path):
    """A category with GT and no detections must contribute AP 0, not be skipped."""
    from shifteq.detstore import ShiftDets

    ann, ids, _ = make_gt_json(tmp_path / "gt.json", n_img=8, n_cat=4, seed=7)
    gt = load_coco_gt(ann, ids)
    empty = ShiftDets(
        counts=np.zeros(len(ids), np.int32), cls=np.zeros(0, np.int16),
        score=np.zeros(0, np.float32), xyxy=np.zeros((0, 4), np.float32), shift=(0, 0),
    )
    r = evaluate_detections(gt, empty, np.arange(4, dtype=np.int64))
    assert r["AP"] == 0.0 and r["AP50"] == 0.0
