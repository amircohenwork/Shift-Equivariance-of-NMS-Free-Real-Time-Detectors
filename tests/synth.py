"""Synthetic COCO-shaped data, so the evaluator and oracle can be validated on a laptop
without the dataset, the weights, or a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shifteq.detstore import ShiftDets


def make_gt_json(path: Path, n_img: int = 40, n_cat: int = 12, seed: int = 0,
                 crowd_frac: float = 0.08) -> tuple[Path, list[int], np.ndarray]:
    """Write a minimal but COCO-legal instances json covering the cases that break naive
    evaluators: crowd regions, tiny/huge boxes, images with no GT, categories with no GT."""
    rng = np.random.default_rng(seed)
    images, anns, cats = [], [], []
    cat_ids = list(range(1, n_cat + 1))
    for c in cat_ids:
        cats.append({"id": c, "name": f"c{c}", "supercategory": "s"})

    aid = 1
    for i in range(n_img):
        W, H = int(rng.integers(320, 700)), int(rng.integers(280, 640))
        iid = 1000 + i
        images.append({"id": iid, "file_name": f"{iid:012d}.jpg", "width": W, "height": H})
        if i % 11 == 0:
            continue  # image with no annotations
        for _ in range(int(rng.integers(1, 9))):
            w = float(rng.uniform(4, W * 0.6))
            h = float(rng.uniform(4, H * 0.6))
            x = float(rng.uniform(0, max(W - w, 1)))
            y = float(rng.uniform(0, max(H - h, 1)))
            anns.append({
                "id": aid, "image_id": iid,
                "category_id": int(rng.choice(cat_ids[:-1])),  # last category has no GT
                "bbox": [x, y, w, h], "area": w * h,
                "iscrowd": int(rng.random() < crowd_frac),
            })
            aid += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    return path, [im["id"] for im in images], np.array(cat_ids, dtype=np.int64)


def make_dets(gt_json: Path, image_ids: list[int], n_cat: int, seed: int = 1,
              jitter: float = 6.0, fp_per_img: int = 12) -> ShiftDets:
    """Detections = jittered GT (recoverable signal) + random false positives (so the
    precision-recall curve actually has structure to interpolate)."""
    rng = np.random.default_rng(seed)
    with open(gt_json) as f:
        d = json.load(f)
    by_img: dict[int, list] = {i: [] for i in image_ids}
    wh = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    counts, cls, score, xyxy = [], [], [], []
    for iid in image_ids:
        rows = []
        for a in by_img[iid]:
            if rng.random() < 0.15:
                continue  # missed detection
            x, y, w, h = a["bbox"]
            j = rng.normal(0, jitter, 4)
            b = [x + j[0], y + j[1], x + w + j[2], y + h + j[3]]
            c = a["category_id"] - 1
            if rng.random() < 0.10:
                c = int(rng.integers(0, n_cat))  # misclassification
            rows.append((c, float(rng.uniform(0.30, 0.99)), b))
        W, H = wh[iid]
        for _ in range(int(rng.integers(0, fp_per_img))):
            w = float(rng.uniform(4, W * 0.5))
            h = float(rng.uniform(4, H * 0.5))
            x = float(rng.uniform(0, max(W - w, 1)))
            y = float(rng.uniform(0, max(H - h, 1)))
            rows.append((int(rng.integers(0, n_cat)), float(rng.uniform(0.001, 0.55)),
                         [x, y, x + w, y + h]))
        rows.sort(key=lambda r: -r[1])
        counts.append(len(rows))
        for c, s, b in rows:
            cls.append(c)
            score.append(s)
            xyxy.append(b)

    return ShiftDets(
        counts=np.array(counts, np.int32),
        cls=np.array(cls, np.int16),
        score=np.array(score, np.float32),
        xyxy=np.array(xyxy, np.float32).reshape(-1, 4),
        shift=(0, 0),
    )
