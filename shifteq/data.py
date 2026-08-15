"""COCO ground truth + a one-shot preprocessing cache.

The single most important performance decision in this repo lives here.

A naive sweep re-decodes and re-letterboxes every image once per shift. At 32x32 / step 2
that is 256 shifts x N images of JPEG decode + two cv2.resize calls -- the sweep becomes
CPU-bound on image loading and the GPU idles. Instead we letterbox each image **once**
into a uint8 [3, imgsz, imgsz] tensor and persist it to a memmap. Applying a shift then
costs one device-side slice-assign into a padded canvas, so every later pass over the
shift grid is pure GPU work.

Cost: N * 3 * 640 * 640 bytes = 1.2 GB per 1000 images. That is the price of turning a
day of dataloader into minutes of GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .modelzoo import ModelSpec
from .utils import LOGGER, read_json, write_json

# COCO "area range" edges, in original-image pixels^2 (COCOeval convention).
AREA_RNG = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}
AREA_KEYS = ("all", "small", "medium", "large")


@dataclass
class ImageMeta:
    image_id: int
    file: str
    ori_h: int
    ori_w: int
    # letterbox geometry, exactly as Ultralytics reports it to ops.scale_boxes:
    #   gain = ratio_pad[0][0], (pad_x, pad_y) = ratio_pad[1]
    gain_y: float
    gain_x: float
    pad_x: int
    pad_y: int


@dataclass
class CocoGT:
    """Ground truth in the shape COCOeval needs, with nothing we do not use."""

    image_ids: np.ndarray                 # [N] int64, the evaluation image order
    boxes: dict[int, np.ndarray]          # image_id -> [G,4] xyxy float32 (original coords)
    cats: dict[int, np.ndarray]           # image_id -> [G]  int64 contiguous cat index
    areas: dict[int, np.ndarray]          # image_id -> [G]  float32 (COCO 'area', segm area)
    iscrowd: dict[int, np.ndarray]        # image_id -> [G]  bool
    cat_ids: np.ndarray                   # [K] the COCO category ids, sorted
    cat_id_to_index: dict[int, int]

    @property
    def num_images(self) -> int:
        return len(self.image_ids)

    @property
    def num_cats(self) -> int:
        return len(self.cat_ids)


def load_coco_gt(ann_json: str | Path, image_ids: list[int] | None = None) -> CocoGT:
    """Parse instances_val2017.json without pulling in pycocotools.

    We reproduce COCOeval._prepare exactly on one point that is easy to get wrong:
    for bbox evaluation, `ignore` is *overwritten* by `iscrowd` -- an annotation is
    ignored iff it is a crowd region. Nothing else in the file marks ignore.
    """
    with open(ann_json) as f:
        d = json.load(f)

    cat_ids = np.array(sorted(c["id"] for c in d["categories"]), dtype=np.int64)
    cat_id_to_index = {int(c): i for i, c in enumerate(cat_ids)}

    all_ids = sorted(im["id"] for im in d["images"])
    keep = set(all_ids if image_ids is None else image_ids)
    ids = np.array([i for i in all_ids if i in keep], dtype=np.int64)

    boxes: dict[int, list] = {int(i): [] for i in ids}
    cats: dict[int, list] = {int(i): [] for i in ids}
    areas: dict[int, list] = {int(i): [] for i in ids}
    crowd: dict[int, list] = {int(i): [] for i in ids}

    for a in d["annotations"]:
        iid = int(a["image_id"])
        if iid not in boxes:
            continue
        x, y, w, h = a["bbox"]
        boxes[iid].append((x, y, x + w, y + h))
        cats[iid].append(cat_id_to_index[int(a["category_id"])])
        areas[iid].append(float(a["area"]))
        crowd[iid].append(bool(a.get("iscrowd", 0)))

    def _stack(dd, dtype, ncol=None):
        out = {}
        for k, v in dd.items():
            if len(v) == 0:
                out[k] = np.zeros((0, ncol) if ncol else (0,), dtype=dtype)
            else:
                out[k] = np.asarray(v, dtype=dtype)
        return out

    gt = CocoGT(
        image_ids=ids,
        boxes=_stack(boxes, np.float32, 4),
        cats=_stack(cats, np.int64),
        areas=_stack(areas, np.float32),
        iscrowd=_stack(crowd, bool),
        cat_ids=cat_ids,
        cat_id_to_index=cat_id_to_index,
    )
    LOGGER.info(
        f"COCO GT: {gt.num_images} images, "
        f"{sum(len(v) for v in gt.cats.values())} annotations, {gt.num_cats} categories"
    )
    return gt


def select_subset(all_image_ids: list[int], n: int | None, seed: int = 0) -> list[int]:
    """Deterministic image subset. Returned sorted so downstream order never depends
    on the RNG implementation."""
    if n is None or n >= len(all_image_ids):
        return sorted(all_image_ids)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_image_ids), size=n, replace=False)
    return sorted(int(all_image_ids[i]) for i in idx)


# --------------------------------------------------------------------------------------
# preprocessing cache
# --------------------------------------------------------------------------------------


def _load_and_resize(path: str, imgsz: int, preproc: str):
    """Byte-for-byte replica of Ultralytics' val-time image load.

    letterbox: BaseDataset.load_image(rect_mode=True) then LetterBox(scaleup=False).
    stretch:   RTDETRDataset.load_image(rect_mode=False), a single resize to a square.
    """
    import cv2

    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    h0, w0 = im.shape[:2]

    if preproc == "stretch":
        if not (h0 == w0 == imgsz):
            im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), (imgsz, imgsz), (0, 0)

    # --- rect_mode=True: long side to imgsz ---
    r = imgsz / max(h0, w0)
    if r != 1:
        import math

        w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    h1, w1 = im.shape[:2]

    # --- LetterBox(new_shape=(imgsz, imgsz), scaleup=False, center=True) ---
    r2 = min(imgsz / h1, imgsz / w1)
    r2 = min(r2, 1.0)  # scaleup=False
    new_unpad = (round(w1 * r2), round(h1 * r2))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w1, h1) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        h1, w1 = im.shape[:2]
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    assert im.shape[:2] == (imgsz, imgsz), im.shape
    return im, (h0, w0), (h1, w1), (left, top)


class PreprocCache:
    """uint8 [N, 3, imgsz, imgsz] RGB memmap + per-image letterbox geometry."""

    def __init__(self, root: str | Path, spec: ModelSpec, image_ids: list[int], img_dir: str | Path,
                 pad_value: int = 114):
        self.root = Path(root)
        self.spec = spec
        self.image_ids = list(image_ids)
        self.img_dir = Path(img_dir)
        self.imgsz = spec.imgsz
        # pad_value is part of the key: a cache built with gray borders is a different
        # dataset from one built with black borders, and silently reusing it would
        # invalidate the padding ablation.
        tag = f"{spec.preproc}_{self.imgsz}_{len(self.image_ids)}_pad{pad_value}"
        self.pad_value = pad_value
        self.arr_path = self.root / f"imgs_{tag}.u8"
        self.meta_path = self.root / f"meta_{tag}.json"
        self._arr: np.ndarray | None = None
        self.meta: list[ImageMeta] = []

    @property
    def shape(self):
        return (len(self.image_ids), 3, self.imgsz, self.imgsz)

    def build(self, pad_value: int | None = None, workers: int = 8, force: bool = False) -> "PreprocCache":
        pad_value = self.pad_value if pad_value is None else pad_value
        if self.arr_path.exists() and self.meta_path.exists() and not force:
            self._load_meta()
            LOGGER.info(f"preproc cache hit: {self.arr_path}")
            return self

        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.arr_path.with_suffix(".partial")
        # Raw memmap, no .npy header -- `self.array` reads it back the same way.
        arr = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=self.shape)

        from concurrent.futures import ThreadPoolExecutor

        metas: list[ImageMeta | None] = [None] * len(self.image_ids)

        def work(k: int):
            iid = self.image_ids[k]
            path = str(self.img_dir / f"{iid:012d}.jpg")
            im, (h0, w0), (h1, w1), (left, top) = _load_and_resize(path, self.imgsz, self.spec.preproc)
            if pad_value != 114 and self.spec.preproc == "letterbox":
                im = _repad(im, left, top, h1, w1, pad_value)
            arr[k] = np.ascontiguousarray(im.transpose(2, 0, 1)[::-1])  # BGR HWC -> RGB CHW
            metas[k] = ImageMeta(
                image_id=int(iid), file=path, ori_h=h0, ori_w=w0,
                gain_y=h1 / h0, gain_x=w1 / w0, pad_x=int(left), pad_y=int(top),
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, _ in enumerate(ex.map(work, range(len(self.image_ids)))):
                if (k + 1) % 500 == 0:
                    LOGGER.info(f"  preprocessed {k + 1}/{len(self.image_ids)}")
        arr.flush()
        del arr
        tmp.replace(self.arr_path)

        self.meta = [m for m in metas if m is not None]
        write_json(self.meta_path, {"pad_value": pad_value, "meta": [m.__dict__ for m in self.meta]})
        LOGGER.info(f"preproc cache written: {self.arr_path} ({np.prod(self.shape) / 1e9:.2f} GB)")
        return self

    def _load_meta(self):
        d = read_json(self.meta_path)
        self.meta = [ImageMeta(**m) for m in d["meta"]]

    @property
    def array(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.memmap(self.arr_path, dtype=np.uint8, mode="r", shape=self.shape)
        return self._arr


def _repad(im, left, top, h1, w1, pad_value: int):
    """Replace the letterbox border with a different constant (padding-value ablation).

    Padding acts as an implicit absolute positional cue (Kayhan & van Gemert, CVPR'20):
    a network can read distance-to-border out of a constant frame. Swapping the constant
    is the cheapest probe of how much of the shift response is really border-driven.
    """
    import cv2

    core = im[top : top + h1, left : left + w1]
    bottom = im.shape[0] - top - h1
    right = im.shape[1] - left - w1
    if pad_value == -1:  # replicate
        return cv2.copyMakeBorder(core, top, bottom, left, right, cv2.BORDER_REPLICATE)
    if pad_value == -2:  # reflect
        return cv2.copyMakeBorder(core, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    v = int(pad_value)
    return cv2.copyMakeBorder(core, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(v, v, v))
