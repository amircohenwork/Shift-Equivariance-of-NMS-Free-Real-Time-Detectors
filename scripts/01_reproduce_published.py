#!/usr/bin/env python3
"""Phase-0 Gate A: reproduce the published COCO val AP for both heads.

Runs Ultralytics' own validator -- not ours -- so this gate answers exactly one
question: is our environment (weights, dataset, package version) the one the numbers
were published from? Gate B (02_harness_gates.py) then asks whether *our* pipeline
agrees with that validator.

Targets for yolo26n (Ultralytics model card / arXiv:2606.03748 Tab. 7):
    one-to-many + NMS   mAP50-95 = 40.9
    one-to-one  (e2e)   mAP50-95 = 40.1
Gate: both within --tol (default 0.1) AP.

The published protocol is `rect=True`
-------------------------------------
`cfg/default.yaml` says `rect: False` but comments "rectangular batching for val when
mode='val'", so which one produced the table is genuinely ambiguous. Measured, on
yolo26n / COCO val:

                    e2e              nms
    rect=False   40.255 (+0.16)   40.867 (-0.03)
    rect=True    40.044 (-0.06)   40.813 (-0.09)      <- both inside 0.1

So the table is rect=True, and this script gates on that. It *also* runs rect=False,
because that square-letterbox number -- not the published one -- is the baseline our shift
harness must match: the canvas protocol needs a fixed input size, and rectangular batching
would make it depend on which images share a batch.

Note the square-vs-rect gap is 4x larger for the one-to-one head (+0.21) than for the
one-to-many head (+0.05). The heads already disagree about how much gray padding costs
them, before any shift is applied. That is worth carrying into the discussion.

Ruled out as explanations for the rect=False residual, each with its own script:
`max_det` (01b -- both heads respond identically, +0.6 AP over 50..500, saturating at
300), confidence threshold and fp16 (01_ --probe), and exact-vs-grouped top-k selection
(01c -- byte-identical selections, 0/64 images differ).

A fresh YOLO() is built for each head, on purpose. `model.val(end2end=True)` routes
through AutoBackend(fuse=True), and BaseModel.fuse() calls Detect.fuse() on an
end-to-end head, which sets `cv2 = cv3 = None` -- it deletes the one-to-many branch from
the live object. Reusing the same handle for the second head would evaluate a mutilated
model, or crash. This is the single most expensive mistake available in this project, so
the script also verifies it happened, rather than trusting the comment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.modelzoo import MODELS  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, sha256_of_file, write_json  # noqa: E402


def coco_eval_json(pred_json: str, ann_json: str, image_ids=None) -> dict:
    """Evaluate a predictions.json the way Ultralytics does -- faster-coco-eval, bbox.

    This is the number the published table reports. `model.val()` hands back a DetMetrics
    whose `results_dict` is Ultralytics' *internal* AP implementation, and the COCO number
    only ever lives in the `stats` dict that `model.val()` discards. Reading the wrong one
    is worth ~0.6 AP here, which is six times the gate tolerance.
    """
    from faster_coco_eval import COCO, COCOeval_faster

    anno = COCO(str(ann_json))
    pred = anno.loadRes(str(pred_json))
    ev = COCOeval_faster(anno, pred, iouType="bbox")
    if image_ids is not None:
        ev.params.imgIds = list(image_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    d = ev.stats_as_dict
    return {
        "mAP50-95": float(d["AP_all"]) * 100,
        "mAP50": float(d["AP_50"]) * 100,
        "mAP75": float(d["AP_75"]) * 100,
        "mAP_small": float(d["AP_small"]) * 100,
        "mAP_medium": float(d["AP_medium"]) * 100,
        "mAP_large": float(d["AP_large"]) * 100,
    }


def val_one(weights: str, head: str, data: str, imgsz: int, batch: int, device: str,
            workers: int, ann_json: str, max_det: int = 300, quantize=None,
            conf: float = 0.001, iou: float = 0.7, rect: bool = False) -> dict:
    """One head, one fresh model handle.

    Fresh on purpose: `model.val(end2end=True)` routes through AutoBackend(fuse=True), and
    BaseModel.fuse() calls Detect.fuse() on an end-to-end head, which sets
    `cv2 = cv3 = None` -- it deletes the one-to-many branch from the live object.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    m = model.val(
        data=data, imgsz=imgsz, batch=batch, device=device, workers=workers,
        conf=conf, iou=iou, max_det=max_det,
        rect=rect,           # False = square letterbox, the documented val default
        quantize=quantize,   # None = fp32; 16 = fp16
        save_json=True, plots=False, verbose=False,
        end2end=(head == "e2e"),
    )
    save_dir = Path(m.save_dir)
    pj = save_dir / "predictions.json"
    out = {"save_dir": str(save_dir), "max_det": max_det, "rect": rect,
           "quantize": quantize, "conf": conf, "iou": iou}
    out["coco"] = coco_eval_json(pj, ann_json)
    rd = m.results_dict
    out["ultralytics_internal"] = {
        "mAP50-95": float(rd.get("metrics/mAP50-95(B)", float("nan"))) * 100,
        "mAP50": float(rd.get("metrics/mAP50(B)", float("nan"))) * 100,
    }
    out["mAP50-95"] = out["coco"]["mAP50-95"]
    return out


def probe_fuse_destroys_o2m(weights: str, device: str) -> dict:
    """Demonstrate, on this exact package version, that fusing an end2end model removes
    the one-to-many head. If a future release changes this, we want the run manifest to
    say so rather than silently relying on a stale assumption."""
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import load_checkpoint

    from shifteq.modelzoo import resolve_device

    net, _ = load_checkpoint(weights, device=resolve_device(device), fuse=False)
    head = [m for m in net.modules() if isinstance(m, Detect)][-1]
    before = head.cv2 is not None and head.cv3 is not None
    net.end2end = True
    net.fuse(verbose=False)
    head = [m for m in net.modules() if isinstance(m, Detect)][-1]
    after = head.cv2 is not None and head.cv3 is not None
    return {"one2many_before_fuse": bool(before), "one2many_after_fuse": bool(after),
            "fuse_destroys_one2many": bool(before and not after)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.1, help="AP tolerance for the gate")
    ap.add_argument("--ann", default=None, help="instances_val2017.json (default: from data yaml)")
    ap.add_argument("--probe", action="store_true",
                    help="also sweep max_det/precision to explain any residual gap")
    ap.add_argument("--gate-rect", type=lambda v: v.lower() != "false", default=True,
                    help="protocol the gate is checked against (published table is rect=True)")
    ap.add_argument("--out", default="results/phase0/gateA_reproduce.json")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    spec = MODELS[args.model]

    import torch
    import ultralytics

    from shifteq.modelzoo import resolve_weights

    wpath = resolve_weights(spec)

    from ultralytics.data.utils import check_det_dataset

    ann = Path(args.ann) if args.ann else (
        Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"
    )
    report = {
        "model": spec.name,
        "weights": str(wpath),
        "weights_sha256": sha256_of_file(wpath),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "data": args.data,
        "imgsz": args.imgsz,
        "published": spec.published_map,
        "tol": args.tol,
        "heads": {},
        "fuse_probe": probe_fuse_destroys_o2m(str(wpath), args.device),
    }
    LOGGER.info(f"fuse probe: {report['fuse_probe']}")

    report["gate_rect"] = args.gate_rect
    report["harness_baseline"] = {}
    failures = []
    for head in spec.heads:
        LOGGER.info(f"=== validating {spec.name} head={head} (gate protocol rect={args.gate_rect}) ===")
        r = val_one(str(wpath), head, args.data, args.imgsz, args.batch, args.device,
                    args.workers, str(ann), rect=args.gate_rect)
        target = spec.published_map.get(head)
        r["published"] = target
        if target is not None:
            r["delta"] = r["mAP50-95"] - target
            r["pass"] = abs(r["delta"]) <= args.tol
            if not r["pass"]:
                failures.append(f"{head}: got {r['mAP50-95']:.3f}, published {target}, "
                                f"delta {r['delta']:+.3f} > tol {args.tol}")
        report["heads"][head] = r
        LOGGER.info(
            f"  {head}: COCO mAP50-95={r['coco']['mAP50-95']:.3f} "
            f"mAP50={r['coco']['mAP50']:.3f} | ultralytics-internal "
            f"{r['ultralytics_internal']['mAP50-95']:.3f} | published {target} "
            f"| delta {r.get('delta', float('nan')):+.3f}"
        )

        # The square-letterbox number is what the shift harness has to reproduce.
        b = val_one(str(wpath), head, args.data, args.imgsz, args.batch, args.device,
                    args.workers, str(ann), rect=False)
        report["harness_baseline"][head] = b["coco"]
        report["harness_baseline"][head]["minus_gate_protocol"] = (
            b["coco"]["mAP50-95"] - r["coco"]["mAP50-95"]
        )
        LOGGER.info(
            f"  {head}: square-letterbox baseline (harness protocol) "
            f"AP={b['coco']['mAP50-95']:.3f} "
            f"({report['harness_baseline'][head]['minus_gate_protocol']:+.3f} vs gate protocol)"
        )

    if {"e2e", "nms"} <= set(report["heads"]):
        report["head_gap_ours"] = report["heads"]["nms"]["mAP50-95"] - report["heads"]["e2e"]["mAP50-95"]
        report["head_gap_published"] = spec.published_map["nms"] - spec.published_map["e2e"]

    if args.probe:
        LOGGER.info("=== sensitivity probe: which knob explains the gap? ===")
        report["probe"] = {}
        for head in spec.heads:
            for label, kw in (("max_det100", dict(max_det=100)),
                              ("fp16", dict(quantize=16)),
                              ("conf0.01", dict(conf=0.01))):
                r = val_one(str(wpath), head, args.data, args.imgsz, args.batch,
                            args.device, args.workers, str(ann), **kw)
                report["probe"][f"{head}/{label}"] = r["coco"]
                LOGGER.info(f"  {head}/{label}: mAP50-95={r['coco']['mAP50-95']:.3f}")

    report["gate_pass"] = not failures
    report["failures"] = failures
    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")

    if failures:
        LOGGER.error("GATE A FAILED:")
        for f in failures:
            LOGGER.error("  " + f)
        sys.exit(1)
    LOGGER.info("GATE A PASSED: both published numbers reproduced within tolerance.")


if __name__ == "__main__":
    main()
