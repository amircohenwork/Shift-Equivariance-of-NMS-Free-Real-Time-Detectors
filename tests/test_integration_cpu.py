"""End-to-end plumbing test on CPU: fake images, a randomly-initialised YOLO26 head.

Detections from random weights are meaningless, and that is fine -- this test exists to
catch the wiring bugs that only appear once real tensors flow through: canvas shapes,
NMS output layout, de-shift indexing, cache round-trips, the sweep loop, and the oracle
built on genuine (not synthetic) match arrays.

No weights download, no COCO, no GPU. Runs in a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

from shifteq.cocolite import build_matches, evaluate_detections, npig_per_cat  # noqa: E402
from shifteq.data import PreprocCache, load_coco_gt  # noqa: E402
from shifteq.detstore import DetStore, RunManifest, to_coco_json  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner, infer_geometry  # noqa: E402
from shifteq.modelzoo import ModelSpec, assert_unfused, head_is, set_head, weight_fingerprint  # noqa: E402
from shifteq.oracle import build_union, delta_ap, greedy, make_accumulator  # noqa: E402
from shifteq.stability import StabilityAccumulator  # noqa: E402
from shifteq.sweep import curve_stats, run_sweep  # noqa: E402
from synth import make_gt_json  # noqa: E402

IMGSZ = 160
MAXSHIFT = 4
N_IMG = 6


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import cv2
    import torch
    from ultralytics.nn.tasks import DetectionModel

    tmp = tmp_path_factory.mktemp("integration")
    ann, ids, cat_ids = make_gt_json(tmp / "gt.json", n_img=N_IMG, n_cat=80, seed=11)
    gt = load_coco_gt(ann, ids)

    img_dir = tmp / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    import json

    meta = {im["id"]: (im["height"], im["width"]) for im in json.loads(Path(ann).read_text())["images"]}
    for iid in ids:
        h, w = meta[iid]
        cv2.imwrite(str(img_dir / f"{iid:012d}.jpg"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))

    spec = ModelSpec(name="yolo26n-test", weights="yolo26n.yaml", imgsz=IMGSZ, stride=32,
                     heads=("e2e", "nms"))
    torch.manual_seed(0)
    net = DetectionModel("yolo26n.yaml", nc=80, verbose=False).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)

    cache = PreprocCache(tmp / "cache", spec, ids, img_dir).build(workers=2)
    return dict(tmp=tmp, ann=ann, ids=ids, gt=gt, spec=spec, net=net, cache=cache,
                cmi=np.arange(80, dtype=np.int64))


def test_preproc_cache_shapes(world):
    c = world["cache"]
    assert c.array.shape == (N_IMG, 3, IMGSZ, IMGSZ)
    assert c.array.dtype == np.uint8
    assert len(c.meta) == N_IMG
    for m in c.meta:
        assert 0 < m.gain_y <= 1.0
        assert m.pad_x >= 0 and m.pad_y >= 0


def test_no_fusion_and_head_switch_preserves_weights(world):
    net = world["net"]
    fp0 = weight_fingerprint(net)
    for head in ("e2e", "nms", "e2e"):
        set_head(net, head)
        assert head_is(net) == head
        ev = assert_unfused(net, require_both_heads=True)
        assert ev["n_batchnorm"] > 10
        assert weight_fingerprint(net) == fp0, "head switch must not touch a single weight"


@pytest.mark.parametrize("head", ["e2e", "nms"])
def test_run_shift_and_deshift(world, head):
    spec, net, cache, gt = world["spec"], world["net"], world["cache"], world["gt"]
    set_head(net, head)
    geom = infer_geometry(spec, MAXSHIFT)
    assert geom.canvas == 192
    r = ShiftRunner(net, spec, cache, geom, PostprocCfg(max_det=50), device="cpu", batch=3)
    d = r.run_shift(2, 3)
    assert d.n_images == N_IMG
    assert d.counts.sum() == len(d.score) == len(d.cls) == len(d.xyxy)
    if d.n_dets:
        for i, m in enumerate(cache.meta):
            _, _, b = d.per_image(i)
            if len(b):
                # de-shifted boxes must live in the ORIGINAL image frame
                assert b[:, 0].min() >= -1e-3 and b[:, 1].min() >= -1e-3
                assert b[:, 2].max() <= m.ori_w + 1e-3
                assert b[:, 3].max() <= m.ori_h + 1e-3


def test_identity_geometry_is_deterministic_and_evaluable(world):
    from shifteq.shift import ShiftGeometry

    spec, net, cache, gt, cmi = (world[k] for k in ("spec", "net", "cache", "gt", "cmi"))
    set_head(net, "e2e")
    geom = ShiftGeometry(spec.imgsz, 0, spec.stride)
    assert geom.canvas == spec.imgsz and geom.is_identity
    r = ShiftRunner(net, spec, cache, geom, PostprocCfg(max_det=50), device="cpu", batch=4)
    a, b = r.run_shift(0, 0), r.run_shift(0, 0)
    assert np.array_equal(a.counts, b.counts)
    assert np.array_equal(a.score, b.score)
    assert np.array_equal(a.xyxy, b.xyxy)
    res = evaluate_detections(gt, a, cmi)
    assert set(res) == {"AP", "AP50", "AP_small", "AP_medium", "AP_large"}
    assert all(0.0 <= v <= 1.0 for v in res.values())


def test_detstore_roundtrip(world, tmp_path):
    spec, net, cache = world["spec"], world["net"], world["cache"]
    set_head(net, "nms")
    geom = infer_geometry(spec, MAXSHIFT)
    r = ShiftRunner(net, spec, cache, geom, PostprocCfg(max_det=30), device="cpu", batch=6)
    d = r.run_shift(1, 1)
    st = DetStore(tmp_path, "run")
    st.write_manifest(RunManifest(
        run_id="run", model=spec.name, head="nms", weights_sha256="x", weight_fingerprint="y",
        imgsz=spec.imgsz, stride=spec.stride, preproc=spec.preproc, max_shift=MAXSHIFT, step=1,
        canvas=geom.canvas, pad_value=114, conf=0.001, iou=0.7, max_det=30, multi_label=True,
        device="cpu", dtype="float32", fused=False, fusion_evidence={},
        image_ids=[int(i) for i in world["ids"]],
    ))
    st.save(1, 1, d.counts, d.cls, d.score, d.xyxy)
    back = st.load(1, 1)
    assert np.array_equal(d.counts, back.counts)
    assert np.array_equal(d.xyxy, back.xyxy)
    assert st.available_shifts() == [(1, 1)]
    assert st.manifest.max_shift == MAXSHIFT
    js = to_coco_json(back, world["ids"], np.arange(1, 81))
    assert len(js) == back.n_dets
    if js:
        assert set(js[0]) == {"image_id", "category_id", "bbox", "score"}


def test_sweep_and_stability(world, tmp_path):
    spec, net, cache, gt, cmi = (world[k] for k in ("spec", "net", "cache", "gt", "cmi"))
    set_head(net, "e2e")
    geom = infer_geometry(spec, MAXSHIFT)
    r = ShiftRunner(net, spec, cache, geom, PostprocCfg(max_det=40), device="cpu", batch=6)
    stab = StabilityAccumulator(gt, cmi, tau=0.0)
    rows = run_sweep(r, gt, cmi, tmp_path / "s.jsonl", max_shift=MAXSHIFT, step=2, stability=stab)
    assert len(rows) == 9                       # (0,2,4)^2
    assert all("AP" in x and "AP_small" in x for x in rows)
    cs = curve_stats(rows, "AP")
    assert cs["AP_range"] >= 0 and cs["n_shifts"] == 9
    s = stab.summary()
    assert s["n_shifts"] == 9 and s["n_gt"] > 0
    for k in ("flicker_rate", "dropout_rate", "class_flip_rate"):
        assert 0.0 <= s[k] <= 1.0
    assert s["jitter_px"] >= 0.0

    # resume: rows already present are not recomputed
    rows2 = run_sweep(r, gt, cmi, tmp_path / "s.jsonl", max_shift=MAXSHIFT, step=2)
    assert len(rows2) == 9


def test_oracle_on_real_matches(world):
    spec, net, cache, gt, cmi = (world[k] for k in ("spec", "net", "cache", "gt", "cmi"))
    set_head(net, "e2e")
    geom = infer_geometry(spec, 1)
    r = ShiftRunner(net, spec, cache, geom, PostprocCfg(max_det=40), device="cpu", batch=6)
    shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    per_shift = [build_matches(gt, r.run_shift(*s), cmi, "all") for s in shifts]
    npig = npig_per_cat(gt, "all")
    u = build_union(per_shift, npig, gt.num_images, gt.num_cats)
    acc = make_accumulator(u, "numpy")
    base = acc.ap(acc.new_mask(np.zeros(gt.num_images, dtype=np.int64)))[0]
    best = greedy(acc, shifts, "best", "AP", log_every=0)
    worst = greedy(acc, shifts, "worst", "AP", log_every=0)
    assert best.ap >= base - 1e-12
    assert worst.ap <= base + 1e-12
    assert delta_ap(best, worst)["dAP"] >= 0
    assert best.n_ap_calls >= 4 * gt.num_images
