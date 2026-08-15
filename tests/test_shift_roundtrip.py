"""The harness geometry must be exactly invertible, independently of any network.

If de-shifting is even slightly wrong, it injects a delta-dependent bias into every box
and the study measures our arithmetic instead of the detector's shift response.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.data import ImageMeta  # noqa: E402
from shifteq.shift import ShiftGeometry, canvas_size, deshift_boxes, embed_numpy, shift_boxes_forward  # noqa: E402

IMGSZ = 640


def _meta(h0, w0, imgsz=IMGSZ):
    """Letterbox geometry as Ultralytics computes it for a val image."""
    import math

    r = imgsz / max(h0, w0)
    w1 = min(math.ceil(w0 * r), imgsz)
    h1 = min(math.ceil(h0 * r), imgsz)
    left = round((imgsz - w1) / 2 - 0.1)
    top = round((imgsz - h1) / 2 - 0.1)
    return ImageMeta(1, "x.jpg", h0, w0, gain_y=h1 / h0, gain_x=w1 / w0, pad_x=left, pad_y=top)


@pytest.mark.parametrize("preproc", ["letterbox", "stretch"])
@pytest.mark.parametrize("hw", [(480, 640), (640, 480), (333, 500), (640, 640), (427, 640)])
def test_roundtrip(preproc, hw):
    m = _meta(*hw)
    rng = np.random.default_rng(0)
    b = np.stack([rng.uniform(0, hw[1], 500), rng.uniform(0, hw[0], 500),
                  rng.uniform(0, hw[1], 500), rng.uniform(0, hw[0], 500)], axis=1)
    b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1)
    b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1)
    for dy in (0, 1, 7, 31):
        for dx in (0, 1, 16, 31):
            fwd = shift_boxes_forward(b, dy, dx, m, IMGSZ, preproc)
            back = deshift_boxes(fwd, dy, dx, m, IMGSZ, preproc, clip=False)
            assert np.abs(back - b).max() < 1e-3, (preproc, hw, dy, dx, np.abs(back - b).max())


@pytest.mark.parametrize("preproc", ["letterbox", "stretch"])
def test_translated_box_deshifts_to_same_place(preproc):
    """The equivariance target: a canvas box translated by exactly delta must land on the
    same original-image box after de-shifting. Any residual in the real experiment is
    therefore the network, not the harness."""
    m = _meta(480, 640)
    base = np.array([[10.0, 20.0, 100.0, 140.0], [0.0, 0.0, 4.0, 4.0], [600.0, 440.0, 640.0, 480.0]])
    c0 = shift_boxes_forward(base, 0, 0, m, IMGSZ, preproc)
    ref = deshift_boxes(c0, 0, 0, m, IMGSZ, preproc, clip=False)
    for dy in range(0, 32, 5):
        for dx in range(0, 32, 7):
            c = c0.copy()
            c[:, [0, 2]] += dx
            c[:, [1, 3]] += dy
            got = deshift_boxes(c, dy, dx, m, IMGSZ, preproc, clip=False)
            assert np.abs(got - ref).max() < 1e-3


def test_canvas_alignment():
    for ms in range(0, 64):
        n = canvas_size(640, ms, 32)
        assert n % 32 == 0 and n >= 640 + ms and n - 32 < 640 + ms
    assert canvas_size(640, 0, 32) == 640      # identity gate collapses to the baseline
    assert canvas_size(640, 31, 32) == 672
    assert canvas_size(640, 1, 32) == 672


def test_embed_places_content_exactly():
    geom = ShiftGeometry(8, 3, 1)
    img = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    for dy in range(4):
        for dx in range(4):
            out = embed_numpy(img, dy, dx, geom, pad_value=114)
            assert out.shape == (3, geom.canvas, geom.canvas)
            assert np.array_equal(out[:, dy : dy + 8, dx : dx + 8], img)
            mask = np.ones(out.shape, bool)
            mask[:, dy : dy + 8, dx : dx + 8] = False
            assert (out[mask] == 114).all()


def test_geometry_rejects_out_of_range_shift():
    g = ShiftGeometry(640, 4, 32)
    g.check(0, 0)
    g.check(4, 4)
    with pytest.raises(ValueError):
        g.check(5, 0)
    with pytest.raises(ValueError):
        g.check(0, -1)
