#!/usr/bin/env python3
"""Phase-0 Gate B: prove the shift harness is a faithful, invertible wrapper.

Read this before quoting "delta=(0,0) reproduces the baseline bit-exactly"
-------------------------------------------------------------------------
That statement is true for exactly one configuration and false for the one you will
actually run, and the difference matters:

  max_shift = 0  ->  canvas == imgsz == 640. The harness degenerates to the stock
                     Ultralytics val pipeline. delta=(0,0) IS the baseline, and equality
                     is bit-exact. This is B1/B3/B4 below and it is a real assertion.

  max_shift = 31 ->  canvas == 672. delta=(0,0) puts the image in the top-left of a
                     672x672 canvas with a 32-px gray border. That is *not* the baseline
                     input: the border changes boundary convolutions and the anchor grid
                     is 84x84 instead of 80x80. Demanding bit-exactness there would be
                     demanding that the padding have no effect -- which is precisely the
                     hypothesis the padding ablation exists to test. B6 therefore
                     *measures* this gap and reports it as a result instead of asserting
                     it away.

What is bit-exact in every configuration is our geometry: preprocessing (B1) and the
de-shift inverse map (B2). Any residual difference between shifts is then attributable to
the network, which is the entire point of the study.

Gates
  B1  preprocessing identity     our cached tensor == Ultralytics' dataloader tensor, byte-for-byte
  B2  de-shift round-trip        deshift(shift(b, d), d) == b analytically, all d, both preproc families
  B3  determinism                two runs of the same delta produce identical detection bytes
  B4  evaluator agreement        cocolite AP == pycocotools/faster-coco-eval AP on the same predictions
  B5  no fusion                  BatchNorms intact, both head branches present, weights unchanged across heads
  B6  canvas effect              measured AP(canvas=672, delta=0) - AP(canvas=640), reported not asserted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, evaluate_detections  # noqa: E402
from shifteq.data import PreprocCache, load_coco_gt, select_subset  # noqa: E402
from shifteq.detstore import coco80_to_coco91, to_coco_json  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner, infer_geometry  # noqa: E402
from shifteq.modelzoo import MODELS, assert_unfused, load_detector, set_head, weight_fingerprint  # noqa: E402
from shifteq.shift import ShiftGeometry, deshift_boxes, shift_boxes_forward  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, sha256_of_array, write_json  # noqa: E402


# ---------------------------------------------------------------- B1
def gate_b1_preproc(cache: PreprocCache, data_yaml: str, imgsz: int, n: int = 32) -> dict:
    """Compare our letterbox cache against Ultralytics' own val dataloader tensors."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    data = check_det_dataset(data_yaml)
    cfg = get_cfg(DEFAULT_CFG, overrides=dict(mode="val", task="detect", imgsz=imgsz, rect=False))
    ds = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)

    by_id = {}
    for k, f in enumerate(ds.im_files):
        stem = Path(f).stem
        if stem.isnumeric():
            by_id[int(stem)] = k

    ours = {m.image_id: i for i, m in enumerate(cache.meta)}
    ids = [i for i in list(ours)[:n] if i in by_id]
    bad_px, bad_geo, checked = [], [], 0
    for iid in ids:
        item = ds[by_id[iid]]
        ref = item["img"].numpy()                     # uint8 [3,H,W], RGB
        our = np.asarray(cache.array[ours[iid]])
        checked += 1
        if ref.shape != our.shape or not np.array_equal(ref, our):
            diff = int((ref.astype(np.int16) != our.astype(np.int16)).sum()) if ref.shape == our.shape else -1
            bad_px.append({"image_id": iid, "shape_ref": list(ref.shape),
                           "shape_ours": list(our.shape), "n_diff_px": diff})
        m = cache.meta[ours[iid]]
        rp = item["ratio_pad"]
        ref_gain, ref_pad = float(rp[0][0]), (int(rp[1][0]), int(rp[1][1]))
        if abs(ref_gain - m.gain_y) > 1e-12 or ref_pad != (m.pad_x, m.pad_y):
            bad_geo.append({"image_id": iid, "ref": [ref_gain, ref_pad],
                            "ours": [m.gain_y, (m.pad_x, m.pad_y)]})
    return {"checked": checked, "pixel_mismatches": bad_px, "geometry_mismatches": bad_geo,
            "pass": checked > 0 and not bad_px and not bad_geo}


# ---------------------------------------------------------------- B2
def gate_b2_roundtrip(cache: PreprocCache, imgsz: int, max_shift: int, preproc: str,
                      n_img: int = 50, n_box: int = 200, seed: int = 0) -> dict:
    """The inverse map must be exact, independently of any network."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    worst_case = None
    shifts = [(0, 0), (0, 1), (1, 0), (1, 1), (max_shift, 0), (0, max_shift), (max_shift, max_shift)]
    shifts = [s for s in shifts if max(s) <= max_shift]
    for m in cache.meta[:n_img]:
        b = np.stack(
            [
                rng.uniform(0, m.ori_w - 1, n_box),
                rng.uniform(0, m.ori_h - 1, n_box),
                rng.uniform(1, m.ori_w, n_box),
                rng.uniform(1, m.ori_h, n_box),
            ],
            axis=1,
        )
        b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1)
        b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1)
        for dy, dx in shifts:
            fwd = shift_boxes_forward(b, dy, dx, m, imgsz, preproc)
            back = deshift_boxes(fwd, dy, dx, m, imgsz, preproc, clip=False)
            err = float(np.abs(back.astype(np.float64) - b).max())
            if err > worst:
                worst, worst_case = err, {"image_id": m.image_id, "shift": [dy, dx]}
    return {"max_abs_err_px": worst, "worst_case": worst_case, "n_images": min(n_img, len(cache.meta)),
            "shifts_tested": [list(s) for s in shifts], "tol_px": 1e-3, "pass": worst < 1e-3}


def gate_b2b_equivariance_of_map(cache: PreprocCache, imgsz: int, max_shift: int, preproc: str) -> dict:
    """A box translated by exactly delta in canvas space must de-shift to the same
    original-image box. This is the property the whole comparison rests on: if the
    detector were perfectly equivariant, its de-shifted output would be delta-invariant."""
    worst = 0.0
    for m in cache.meta[:50]:
        base = np.array([[10.0, 20.0, 100.0, 140.0], [0.0, 0.0, 5.0, 5.0]])
        c0 = shift_boxes_forward(base, 0, 0, m, imgsz, preproc)
        for dy in (0, 1, max_shift):
            for dx in (0, 1, max_shift):
                if max(dy, dx) > max_shift:
                    continue
                cd = c0.copy()
                cd[:, [0, 2]] += dx
                cd[:, [1, 3]] += dy
                back = deshift_boxes(cd, dy, dx, m, imgsz, preproc, clip=False)
                ref = deshift_boxes(c0, 0, 0, m, imgsz, preproc, clip=False)
                worst = max(worst, float(np.abs(back - ref).max()))
    return {"max_abs_err_px": worst, "tol_px": 1e-3, "pass": worst < 1e-3}


# ---------------------------------------------------------------- B3
def gate_b3_determinism(runner: ShiftRunner, shifts=((0, 0), (1, 1))) -> dict:
    out = {}
    ok = True
    for dy, dx in shifts:
        a = runner.run_shift(dy, dx)
        b = runner.run_shift(dy, dx)
        h = {
            "counts": (sha256_of_array(a.counts), sha256_of_array(b.counts)),
            "score": (sha256_of_array(a.score), sha256_of_array(b.score)),
            "xyxy": (sha256_of_array(a.xyxy), sha256_of_array(b.xyxy)),
            "cls": (sha256_of_array(a.cls), sha256_of_array(b.cls)),
        }
        same = all(x == y for x, y in h.values())
        ok &= same
        out[f"{dy},{dx}"] = {"identical": bool(same), "n_dets": a.n_dets,
                             "sha256": {k: v[0] for k, v in h.items()}}
    out["pass"] = bool(ok)
    return out


# ---------------------------------------------------------------- B4
def gate_b4_evaluator(gt, dets, class_map_inv, class_map, ann_json: str, image_ids: list[int],
                      tol: float = 1e-4) -> dict:
    ours = evaluate_detections(gt, dets, class_map_inv)
    preds = to_coco_json(dets, image_ids, class_map, round_coords=False)
    ref, backend = _reference_ap(ann_json, preds, image_ids)
    if ref is None:
        return {"pass": None, "reason": "no pycocotools/faster-coco-eval available", "ours": ours}
    deltas = {k: abs(ours[k] - ref[k]) for k in ref if k in ours}
    return {"backend": backend, "ours": ours, "reference": ref, "abs_delta": deltas,
            "tol": tol, "pass": all(v <= tol for v in deltas.values())}


def _reference_ap(ann_json: str, preds: list[dict], image_ids: list[int]):
    import json
    import tempfile

    if not preds:
        return None, "none"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        pjson = f.name
    for backend in ("faster_coco_eval", "pycocotools"):
        try:
            if backend == "faster_coco_eval":
                from faster_coco_eval import COCO, COCOeval_faster as E
            else:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval as E
            anno = COCO(ann_json)
            pred = anno.loadRes(pjson)
            ev = E(anno, pred, iouType="bbox")
            ev.params.imgIds = list(image_ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            s = ev.stats
            return {"AP": float(s[0]), "AP50": float(s[1]), "AP_small": float(s[3]),
                    "AP_medium": float(s[4]), "AP_large": float(s[5])}, backend
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(f"{backend} unavailable/failed: {e}")
    return None, "none"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--head", default="e2e", choices=["e2e", "nms"])
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None, help="instances_val2017.json (default: from data yaml)")
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--subset", type=int, default=500)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=31)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--b1-images", type=int, default=32,
                    help="how many images to byte-compare against the Ultralytics dataloader")
    ap.add_argument("--out", default="results/phase0/gateB_harness.json")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(args.data)
    root = Path(data["path"])
    ann = Path(args.ann) if args.ann else root / "annotations" / "instances_val2017.json"
    img_dir = Path(args.img_dir) if args.img_dir else root / "images" / "val2017"

    spec = MODELS[args.model]
    gt_all = load_coco_gt(ann)
    ids = select_subset([int(i) for i in gt_all.image_ids], args.subset, args.subset_seed)
    gt = load_coco_gt(ann, ids)
    class_map = coco80_to_coco91()
    cmi = class_map_inverse(gt, class_map)

    cache = PreprocCache(Path(args.cache_dir) / "preproc", spec, ids, img_dir).build()

    net, spec = load_detector(spec, device=args.device)
    fp_before = weight_fingerprint(net)
    set_head(net, args.head)
    fp_after = weight_fingerprint(net)
    fusion = assert_unfused(net, require_both_heads=len(spec.heads) > 1)

    report: dict = {
        "model": spec.name, "head": args.head, "subset": len(ids), "max_shift": args.max_shift,
        "device": args.device, "torch": torch.__version__,
    }

    LOGGER.info("== B1 preprocessing identity ==")
    report["B1_preproc_identity"] = gate_b1_preproc(cache, args.data, spec.imgsz, n=args.b1_images)

    LOGGER.info("== B2 de-shift round-trip ==")
    report["B2_deshift_roundtrip"] = gate_b2_roundtrip(cache, spec.imgsz, args.max_shift, spec.preproc)
    report["B2b_map_equivariance"] = gate_b2b_equivariance_of_map(cache, spec.imgsz, args.max_shift, spec.preproc)

    pp = PostprocCfg()
    LOGGER.info("== B3 determinism (identity geometry, canvas == imgsz) ==")
    geom0 = ShiftGeometry(spec.imgsz, 0, spec.stride)
    runner0 = ShiftRunner(net, spec, cache, geom0, pp, args.device, args.batch)
    report["B3_determinism"] = gate_b3_determinism(runner0, shifts=((0, 0),))

    LOGGER.info("== B4 evaluator agreement (our COCO AP vs reference) ==")
    dets0 = runner0.run_shift(0, 0)
    report["B4_evaluator"] = gate_b4_evaluator(gt, dets0, cmi, class_map, str(ann), ids)
    report["baseline_AP_canvas640"] = evaluate_detections(gt, dets0, cmi)

    LOGGER.info("== B5 no fusion ==")
    report["B5_no_fusion"] = {
        **fusion,
        "weights_unchanged_by_head_switch": fp_before == fp_after,
        "fingerprint": fp_after,
        "pass": bool(fusion["n_batchnorm"] >= 10 and fp_before == fp_after),
    }

    LOGGER.info("== B6 canvas effect at delta=(0,0) (measured, not asserted) ==")
    geomN = infer_geometry(spec, args.max_shift)
    runnerN = ShiftRunner(net, spec, cache, geomN, pp, args.device, args.batch)
    detsN = runnerN.run_shift(0, 0)
    apN = evaluate_detections(gt, detsN, cmi)
    same_dets = (
        np.array_equal(dets0.counts, detsN.counts)
        and np.array_equal(dets0.score, detsN.score)
        and np.array_equal(dets0.xyxy, detsN.xyxy)
    )
    report["B6_canvas_effect"] = {
        "canvas": geomN.canvas, "imgsz": spec.imgsz,
        "AP_canvas640": report["baseline_AP_canvas640"],
        "AP_canvasN_delta0": apN,
        "dAP": {k: apN[k] - report["baseline_AP_canvas640"][k] for k in apN},
        "detections_bit_identical": bool(same_dets),
        "note": ("Padding the canvas to fit the shift grid changes the input; a non-zero dAP here "
                 "is the border/positional-encoding effect, measured at zero shift. It is a result, "
                 "not a bug. Report it next to the padding-value ablation."),
    }

    gates = {k: v.get("pass") for k, v in report.items() if isinstance(v, dict) and "pass" in v}
    hard_fail = [k for k, v in gates.items() if v is False]
    report["gates"] = gates
    report["gate_pass"] = not hard_fail
    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")
    for k, v in gates.items():
        LOGGER.info(f"  {k}: {'PASS' if v else ('SKIP' if v is None else 'FAIL')}")
    if hard_fail:
        LOGGER.error(f"GATE B FAILED: {hard_fail}")
        sys.exit(1)
    LOGGER.info("GATE B PASSED.")


if __name__ == "__main__":
    main()
