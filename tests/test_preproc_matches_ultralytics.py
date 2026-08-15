"""Gate B1, run locally on synthetic images: our letterbox cache must be byte-identical
to what Ultralytics' val dataloader produces.

This is the highest-risk piece of the harness. The val path is two resizes with different
rounding rules (`ceil` in BaseDataset.load_image, `round(dh - 0.1)` / `round(dh + 0.1)` in
LetterBox), and being one pixel off would shift every box by one pixel -- an error the
size of the entire effect under study, applied uniformly, and invisible in the AP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

from shifteq.data import PreprocCache, _load_and_resize  # noqa: E402
from shifteq.modelzoo import ModelSpec  # noqa: E402

IMGSZ = 640
SHAPES = [(480, 640), (640, 480), (333, 500), (640, 640), (427, 640), (500, 333), (200, 900)]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A minimal but real Ultralytics YOLODataset over synthetic images."""
    import cv2

    root = tmp_path_factory.mktemp("uld")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    rng = np.random.default_rng(0)
    ids = []
    for k, (h, w) in enumerate(SHAPES):
        iid = 900000 + k
        ids.append(iid)
        cv2.imwrite(str(root / "images" / f"{iid:012d}.jpg"),
                    rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        (root / "labels" / f"{iid:012d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG, overrides=dict(mode="val", task="detect", imgsz=IMGSZ, rect=False))
    ds = YOLODataset(
        img_path=str(root / "images"), imgsz=IMGSZ, batch_size=1, augment=False, hyp=cfg,
        rect=False, cache=None, single_cls=False, stride=32, pad=0.5, prefix="test: ",
        task="detect", classes=None, data={"names": {0: "a"}, "nc": 1, "channels": 3},
        fraction=1.0,
    )
    return root, ds, ids


def test_pixels_and_geometry_match(dataset):
    root, ds, ids = dataset
    by_stem = {Path(f).stem: k for k, f in enumerate(ds.im_files)}

    for iid in ids:
        item = ds[by_stem[f"{iid:012d}"]]
        ref = item["img"].numpy()                       # uint8 [3,640,640], RGB

        im, (h0, w0), (h1, w1), (left, top) = _load_and_resize(
            str(root / "images" / f"{iid:012d}.jpg"), IMGSZ, "letterbox"
        )
        ours = np.ascontiguousarray(im.transpose(2, 0, 1)[::-1])

        assert ref.shape == ours.shape == (3, IMGSZ, IMGSZ), (ref.shape, ours.shape)
        n_diff = int((ref != ours).sum())
        assert n_diff == 0, f"image {iid}: {n_diff} differing pixels vs Ultralytics dataloader"

        rp = item["ratio_pad"]
        assert abs(float(rp[0][0]) - h1 / h0) < 1e-12, (iid, rp, h1 / h0)
        assert (int(rp[1][0]), int(rp[1][1])) == (left, top), (iid, rp, (left, top))
        assert tuple(item["ori_shape"]) == (h0, w0)


def test_cache_reproduces_the_same_bytes(dataset, tmp_path):
    root, ds, ids = dataset
    spec = ModelSpec(name="t", weights="none", imgsz=IMGSZ, stride=32)
    cache = PreprocCache(tmp_path, spec, ids, root / "images").build(workers=2)
    by_stem = {Path(f).stem: k for k, f in enumerate(ds.im_files)}
    for i, iid in enumerate(ids):
        ref = ds[by_stem[f"{iid:012d}"]]["img"].numpy()
        assert np.array_equal(np.asarray(cache.array[i]), ref), f"cache mismatch for {iid}"

    # a second build must hit the cache and yield identical bytes
    again = PreprocCache(tmp_path, spec, ids, root / "images").build(workers=2)
    assert np.array_equal(np.asarray(again.array), np.asarray(cache.array))
    assert [m.image_id for m in again.meta] == [m.image_id for m in cache.meta]


def test_pad_value_ablation_only_changes_the_border(dataset, tmp_path):
    root, ds, ids = dataset
    spec = ModelSpec(name="t", weights="none", imgsz=IMGSZ, stride=32)
    a = PreprocCache(tmp_path / "a", spec, ids, root / "images", pad_value=114).build(workers=2)
    b = PreprocCache(tmp_path / "b", spec, ids, root / "images", pad_value=0).build(workers=2)
    assert a.arr_path != b.arr_path, "pad_value must be part of the cache key"
    for i, m in enumerate(a.meta):
        h1 = int(round(m.gain_y * m.ori_h))
        w1 = int(round(m.gain_x * m.ori_w))
        core = (slice(None), slice(m.pad_y, m.pad_y + h1), slice(m.pad_x, m.pad_x + w1))
        assert np.array_equal(np.asarray(a.array[i])[core], np.asarray(b.array[i])[core])
        if m.pad_y > 0:
            assert (np.asarray(b.array[i])[:, : m.pad_y, :] == 0).all()
        if m.pad_x > 0:
            assert (np.asarray(b.array[i])[:, :, : m.pad_x] == 0).all()
