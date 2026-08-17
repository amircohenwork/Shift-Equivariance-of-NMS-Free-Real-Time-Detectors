"""The oracle's fast path must equal the slow path, and Algorithm 1 must bracket the
baseline.

The union-table trick (sort once over all shifts, then select an assignment with a
boolean mask) is the only reason 20,000 full-dataset AP evaluations are affordable. It is
also the only place where a subtle indexing bug would produce plausible-looking numbers,
so it is checked against a from-scratch `cocolite.accumulate` on the same selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.cocolite import accumulate, build_matches, npig_per_cat  # noqa: E402
from shifteq.data import load_coco_gt  # noqa: E402
from shifteq.oracle import (  # noqa: E402
    NumpyAccumulator, TorchAccumulator, build_union, delta_ap, greedy,
)
from synth import make_dets, make_gt_json  # noqa: E402

N_CAT = 10
N_SHIFT = 4


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("oracle")
    ann, ids, cat_ids = make_gt_json(tmp / "gt.json", n_img=25, n_cat=N_CAT, seed=3)
    gt = load_coco_gt(ann, ids)
    cmi = np.arange(N_CAT, dtype=np.int64)
    # Four "shifts": same ground truth, four independently perturbed detection sets.
    per_shift = [build_matches(gt, make_dets(ann, ids, N_CAT, seed=200 + s), cmi, "all")
                 for s in range(N_SHIFT)]
    npig = npig_per_cat(gt, "all")
    u = build_union(per_shift, npig, gt.num_images, gt.num_cats)
    return gt, per_shift, npig, u


def _from_scratch(per_shift, assign, npig, n_cats):
    parts = {k: [] for k in ("cat", "match", "ign", "score", "img")}
    for i, s in enumerate(assign):
        m = per_shift[int(s)]
        sel = m.img == i
        for k, v in (("cat", m.cat), ("match", m.match), ("ign", m.ign),
                     ("score", m.score), ("img", m.img)):
            parts[k].append(v[sel])
    c = {k: np.concatenate(v) for k, v in parts.items()}
    return accumulate(c["cat"], c["match"], c["ign"], npig, n_cats,
                      presorted=False, score=c["score"], img=c["img"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_union_select_equals_from_scratch(setup, seed):
    gt, per_shift, npig, u = setup
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, N_SHIFT, size=u.n_images)
    acc = NumpyAccumulator(u)
    ap, ap50 = acc.ap(acc.new_mask(assign))
    ref = _from_scratch(per_shift, assign, npig, gt.num_cats)
    assert abs(ap - ref["AP"]) < 1e-12, (ap, ref["AP"])
    assert abs(ap50 - ref["AP50"]) < 1e-12, (ap50, ref["AP50"])


def test_torch_matches_numpy(setup):
    pytest.importorskip("torch")
    _, _, _, u = setup
    a, b = NumpyAccumulator(u), TorchAccumulator(u, device="cpu")
    rng = np.random.default_rng(9)
    for _ in range(4):
        assign = rng.integers(0, N_SHIFT, size=u.n_images)
        x = a.ap(a.new_mask(assign))
        y = b.ap(b.new_mask(assign))
        assert abs(x[0] - y[0]) < 1e-9 and abs(x[1] - y[1]) < 1e-9, (x, y)


def test_set_image_is_incremental(setup):
    """Editing one image's rows must give the same mask as rebuilding it wholesale."""
    _, _, _, u = setup
    acc = NumpyAccumulator(u)
    assign = np.zeros(u.n_images, dtype=np.int64)
    mask = acc.new_mask(assign)
    rng = np.random.default_rng(4)
    for _ in range(20):
        i = int(rng.integers(0, u.n_images))
        s = int(rng.integers(0, N_SHIFT))
        assign[i] = s
        acc.set_image(mask, i, s)
        assert np.array_equal(mask, acc.new_mask(assign))


def test_greedy_brackets_baseline(setup):
    gt, per_shift, npig, u = setup
    shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    acc = NumpyAccumulator(u)
    base_ap, base_ap50 = acc.ap(acc.new_mask(np.zeros(u.n_images, dtype=np.int64)))

    best = greedy(acc, shifts, "best", "AP", iters=1, log_every=0)
    worst = greedy(acc, shifts, "worst", "AP", iters=1, log_every=0)

    assert best.ap >= base_ap - 1e-12, (best.ap, base_ap)
    assert worst.ap <= base_ap + 1e-12, (worst.ap, base_ap)
    d = delta_ap(best, worst)
    assert d["dAP"] >= 0
    # The whole premise of the metric: per-image selection moves AP much further than any
    # single uniform choice can.
    assert d["dAP"] > 0, "no envelope at all -- the four detection sets are identical?"
    assert best.n_ap_calls > 0


def test_greedy_second_iteration_changes_little(setup):
    """The paper's justification for K=1. Worth re-checking on our detectors before
    quoting a one-iteration number."""
    _, _, _, u = setup
    shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    acc = NumpyAccumulator(u)
    r = greedy(acc, shifts, "best", "AP", iters=3, log_every=0)
    aps = [it["AP"] for it in r.iterations]
    assert aps == sorted(aps), "greedy AP must be non-decreasing across iterations"
    if len(aps) > 1:
        assert (aps[-1] - aps[0]) <= 0.05 * max(aps[0], 1e-9) + 1e-6


# --------------------------------------------------------------------------------------
# Regression: CUDA true division vs reciprocal multiply at exact recall thresholds
# --------------------------------------------------------------------------------------
def _exact_threshold_npigs(rec_thrs=None):
    """Category sizes for which some m/npig lands *exactly* on a COCO recall threshold.

    COCO interpolates precision at 101 thresholds 0.00, 0.01, ... 1.00. Whenever a
    category's non-ignored GT count `npig` makes m/npig equal one of them, the recall
    array touches a threshold exactly -- and a one-ULP error in computing m/npig moves it
    to the other side, flipping `searchsorted(..., side="left")` by a whole position.
    """
    from shifteq.cocolite import REC_THRS

    rec = REC_THRS if rec_thrs is None else rec_thrs
    out = []
    for npig in range(2, 120):
        ms = np.arange(npig + 1, dtype=np.float64) / np.float64(npig)
        if np.isin(ms, rec).sum() > 2:
            out.append(npig)
    return out


def test_exact_recall_thresholds_are_actually_reachable():
    """The premise of the regression test below: such category sizes exist and are common."""
    ns = _exact_threshold_npigs()
    assert len(ns) > 10, ns
    assert 55 in ns, "npig=55 gives 11/55=0.2, 22/55=0.4, 44/55=0.8 exactly"


def test_torch_accumulator_matches_numpy_on_every_device(setup):
    """`tpc / npig` with a Python-int divisor becomes a reciprocal multiply on CUDA, one
    ULP below true division. Wherever a recall value sits exactly on a threshold that flips
    the interpolation index and changes AP -- 0.0013 AP over COCO val, 7 of 80 categories.

    torch/cpu never had the bug, so a CPU-only run of this test cannot fail; it is here so
    that a GPU machine catches a regression. The `npig_f` device tensor in
    `TorchAccumulator` is the fix.
    """
    torch = pytest.importorskip("torch")
    _, _, _, u = setup
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    ref = NumpyAccumulator(u)
    rng = np.random.default_rng(11)
    for dev in devices:
        acc = TorchAccumulator(u, device=dev)
        for _ in range(3):
            assign = rng.integers(0, N_SHIFT, size=u.n_images)
            x = ref.ap(ref.new_mask(assign))
            y = acc.ap(acc.new_mask(assign))
            assert abs(x[0] - y[0]) < 1e-12 and abs(x[1] - y[1]) < 1e-12, (dev, x, y)


def test_divisor_is_a_tensor_not_a_python_int():
    """Pin the fix itself. `tensor / int` and `tensor / tensor` lower to different CUDA
    kernels, and only the second is IEEE true division -- a future refactor that 'simplifies'
    `self.npig_f[k]` back to `npig` would silently reintroduce the bug on GPU only."""
    import inspect

    src = inspect.getsource(TorchAccumulator.ap)
    assert "rc = tpc / self.npig_f[k]" in src, (
        "TorchAccumulator.ap must divide by the device tensor `npig_f`, not a Python int"
    )
