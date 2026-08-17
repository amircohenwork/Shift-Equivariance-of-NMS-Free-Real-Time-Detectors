"""Edge cases where a hand-written COCO evaluator and pycocotools can silently disagree.

`test_cocolite_vs_pycocotools.py` compares the two on randomly generated data. Random data
is exactly what will not contain the cases below: they sit on measure-zero boundaries that
a uniform sampler never lands on, and each one is a place where the reference implementation
has a quirk rather than an obvious behaviour.

The cases planted here, and why each is reachable in this study specifically:

1. **Zero-area detections.** `deshift_boxes` clips predictions to the original image bounds.
   A box predicted entirely inside the gray letterbox padding clips to zero width or height
   -- and under the shift protocol the padding is where boxes move to. pycocotools ignores a
   detection whose area is *not strictly inside* the area range (`area > lo and area < hi`),
   so a zero-area detection is dropped rather than counted as a false positive. Our `d_oor`
   uses `area < lo or area > hi`, which does not drop it. At `lo = 0` those disagree for
   exactly area == 0.

2. **Area exactly on a small/medium/large boundary.** The same asymmetry at 32^2 and 96^2.
   pycocotools applies *different* conventions to ground truth (inclusive) and detections
   (exclusive), which is a quirk worth pinning rather than inferring.

3. **Exact IoU ties.** pycocotools scans ground truth in order and replaces its current best
   on `>=`, so a tie resolves to the *last* candidate; an `argmax` resolves it to the first.
   Two identical ground-truth boxes make this reachable.

Every test asserts twice: that we agree with the reference, **and** that the planted rows
actually move AP. The second assertion is not decoration. The first version of this file
placed the zero-area detections below the true positives in score order, so they landed past
full recall and could not change AP by construction -- the test passed against a reference
that was never exercised. A test that cannot fail is worse than no test, because it is
recorded as evidence.

If any of these fails, the fix is in `shifteq/cocolite.py`, not here.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import evaluate_detections  # noqa: E402
from shifteq.data import load_coco_gt  # noqa: E402
from shifteq.detstore import ShiftDets, to_coco_json  # noqa: E402

W = H = 640
CATS = [1, 2]


def _write_gt(path: Path, anns: list[dict], n_img: int = 2) -> Path:
    images = [{"id": 1000 + i, "file_name": f"{1000 + i:012d}.jpg", "width": W, "height": H}
              for i in range(n_img)]
    cats = [{"id": c, "name": f"c{c}", "supercategory": "s"} for c in CATS]
    path.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return path


def _dets(rows: list[tuple[int, int, float, tuple]], image_ids: list[int]) -> ShiftDets:
    """rows = (image index, class index, score, xyxy). Detections are grouped by image,
    which is the layout `ShiftDets.per_image` assumes."""
    counts = np.zeros(len(image_ids), np.int32)
    for i, _, _, _ in rows:
        counts[i] += 1
    order = sorted(range(len(rows)), key=lambda k: rows[k][0])
    return ShiftDets(
        counts=counts,
        cls=np.array([rows[k][1] for k in order], np.int16),
        score=np.array([rows[k][2] for k in order], np.float32),
        xyxy=np.array([rows[k][3] for k in order], np.float32).reshape(-1, 4),
        shift=(0, 0),
    )


def _reference(ann_json: Path, preds: list[dict], image_ids: list[int]) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        pj = f.name
    try:
        from faster_coco_eval import COCO, COCOeval_faster as E
    except Exception:  # noqa: BLE001
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval as E
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no reference COCO evaluator: {e}")
    anno = COCO(str(ann_json))
    pred = anno.loadRes(pj)
    ev = E(anno, pred, iouType="bbox")
    ev.params.imgIds = list(image_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP_small": float(s[3]),
            "AP_medium": float(s[4]), "AP_large": float(s[5])}


def _evaluate(tmp_path: Path, anns: list[dict], rows: list, tag: str) -> tuple[dict, dict]:
    ann = _write_gt(tmp_path / f"gt_{tag}.json", anns)
    gt = load_coco_gt(ann)
    ids = [int(i) for i in gt.image_ids]
    dets = _dets(rows, ids)
    ours = evaluate_detections(gt, dets, np.array(CATS, dtype=np.int64) - 1)
    ref = _reference(ann, to_coco_json(dets, ids, np.array(CATS), round_coords=False), ids)
    return ours, ref


def _compare(tmp_path: Path, anns: list[dict], rows: list, label: str,
             planted: list[int], tol: float = 1e-9, keys=("AP", "AP50")):
    """Agree with the reference, and prove the planted rows were doing something.

    `planted` indexes the rows whose edge-case property is under test; dropping them must
    change AP, otherwise the comparison never reached the code path it claims to cover.
    """
    ours, ref = _evaluate(tmp_path, anns, rows, label)
    bad = {k: (ours[k], ref[k]) for k in ref
           if not (np.isnan(ours[k]) and np.isnan(ref[k])) and abs(ours[k] - ref[k]) > tol}
    assert not bad, f"{label}: disagreements (ours, reference) = {bad}"

    without = [r for i, r in enumerate(rows) if i not in set(planted)]
    base, _ = _evaluate(tmp_path, anns, without, label + "_base")
    moved = max(abs(ours[k] - base[k]) for k in keys)
    assert moved > 1e-12, (
        f"{label}: dropping the planted rows changed {keys} by {moved:.2e} -- the test is "
        "vacuous and proves nothing about the edge case it names"
    )


def _ann(aid, iid, cat, box, crowd=0):
    x, y, w, h = box
    return {"id": aid, "image_id": iid, "category_id": cat, "bbox": [x, y, w, h],
            "area": w * h, "iscrowd": crowd}


def test_zero_area_detections(tmp_path):
    """A box clipped flat by de-shifting: false positive, or ignored?

    The degenerate boxes are given the *highest* scores on purpose. Below the true positives
    they would sit past full recall and could not move AP whatever the evaluator decided.
    """
    anns = [_ann(1, 1000, 1, (100.0, 100.0, 50.0, 50.0)),
            _ann(2, 1000, 1, (400.0, 400.0, 70.0, 70.0)),
            _ann(3, 1001, 1, (200.0, 200.0, 80.0, 60.0))]
    rows = [
        (0, 0, 0.99, (0.0, 0.0, 0.0, 0.0)),           # fully clipped: zero area
        (0, 0, 0.98, (5.0, 5.0, 5.0, 60.0)),          # zero width
        (0, 0, 0.97, (10.0, 20.0, 410.0, 20.0)),      # zero height
        (0, 0, 0.60, (100.0, 100.0, 150.0, 150.0)),   # a real hit, below the degenerates
        (0, 0, 0.55, (400.0, 400.0, 470.0, 470.0)),   # another real hit
        (1, 0, 0.50, (200.0, 200.0, 280.0, 260.0)),
    ]
    _compare(tmp_path, anns, rows, "zero_area", planted=[0, 1, 2])


@pytest.mark.parametrize("side", [32.0, 96.0])
def test_area_exactly_on_a_size_boundary(tmp_path, side):
    """COCO's small/medium/large edges are 32^2 and 96^2. pycocotools uses an inclusive test
    for ground truth and an exclusive one for detections; we must match both."""
    anns = [_ann(1, 1000, 1, (100.0, 100.0, side, side)),          # GT exactly on the edge
            _ann(2, 1000, 1, (300.0, 300.0, 120.0, 120.0)),
            _ann(3, 1001, 1, (50.0, 50.0, 45.0, 45.0))]
    rows = [
        (0, 0, 0.99, (250.0, 250.0, 250.0 + side, 250.0 + side)),  # FP, exactly on the edge
        (0, 0, 0.90, (100.0, 100.0, 100.0 + side, 100.0 + side)),  # TP, exactly on the edge
        (0, 0, 0.70, (300.0, 300.0, 420.0, 420.0)),
        (1, 0, 0.60, (50.0, 50.0, 95.0, 95.0)),
    ]
    _compare(tmp_path, anns, rows, f"boundary_{int(side)}", planted=[0, 1],
             keys=("AP", "AP50", "AP_small", "AP_medium", "AP_large"))


def test_exact_iou_ties_between_identical_ground_truth(tmp_path):
    """Two identical ground-truth boxes and one detection: the IoU to each is bit-identical,
    so which one is matched is decided purely by the tie-break rule."""
    anns = [_ann(1, 1000, 1, (100.0, 100.0, 60.0, 60.0)),
            _ann(2, 1000, 1, (100.0, 100.0, 60.0, 60.0)),   # exact duplicate
            _ann(3, 1001, 1, (300.0, 300.0, 40.0, 40.0))]
    rows = [
        (0, 0, 0.90, (100.0, 100.0, 160.0, 160.0)),   # IoU 1.0 with both
        (0, 0, 0.80, (103.0, 103.0, 163.0, 163.0)),   # near-tie with both
        (1, 0, 0.70, (300.0, 300.0, 340.0, 340.0)),
    ]
    _compare(tmp_path, anns, rows, "iou_tie", planted=[0, 1])


def test_tie_with_one_ignored_and_one_scored_ground_truth(tmp_path):
    """A crowd region and a normal box in the same place. The scored one must win, whatever
    the IoU ordering says -- that is the `break` rule in pycocotools' matching loop."""
    anns = [_ann(1, 1000, 1, (100.0, 100.0, 60.0, 60.0), crowd=1),
            _ann(2, 1000, 1, (100.0, 100.0, 60.0, 60.0), crowd=0),
            _ann(3, 1001, 1, (300.0, 300.0, 40.0, 40.0))]
    rows = [
        (0, 0, 0.90, (100.0, 100.0, 160.0, 160.0)),
        (0, 0, 0.60, (100.0, 100.0, 160.0, 160.0)),   # second det: scored GT already taken
        (1, 0, 0.70, (300.0, 300.0, 340.0, 340.0)),
    ]
    _compare(tmp_path, anns, rows, "crowd_tie", planted=[0, 1])
