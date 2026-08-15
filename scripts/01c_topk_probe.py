#!/usr/bin/env python3
"""Is the deployed end-to-end path the same function as the PyTorch one?

`Detect.get_topk_index` picks its top-k implementation from the *export* state:

    groups = 8 if self.export and self.format == "engine" and not self.dynamic else 1
    ... self._grouped_topk(scores.max(dim=-1)[0], k, groups)

With `groups=1` it is an exact `topk`. With `groups=8` the anchor axis is split into 8
chunks, top-k is taken within each chunk, and the winners are re-ranked -- exact only if
no chunk holds more than k of the true top-k. So the TensorRT-exported end-to-end model
can select a *different* set of 300 boxes than the PyTorch model with identical weights.

That matters twice over:

1. It is the leading candidate for Gate A's residual: our PyTorch e2e run scores 40.26
   against a published 40.1, and the published figure describes the edge-deployment path.
2. It matters for the paper's own question. "What does end-to-end inference cost?" is a
   question about the thing people deploy. If the deployed selection is an approximation
   of the one we measure, every stability number we report is about the PyTorch variant
   and must say so.

This probe runs the same weights through both selection paths and reports the AP gap and
how often the selected index sets actually differ.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.modelzoo import MODELS, load_detector, resolve_weights, set_head  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402

_A = import_module("01_reproduce_published")


def _set_topk_mode(grouped: bool) -> None:
    """Flip the class-level switches `get_topk_index` reads."""
    from ultralytics.nn.modules.head import Detect

    Detect.export = bool(grouped)
    Detect.format = "engine" if grouped else None
    Detect.dynamic = False


def selection_disagreement(model: str, device: str, n_batches: int = 8, batch: int = 8,
                           canvas: int = 640, seed: int = 0) -> dict:
    """How often do exact and grouped top-k disagree on random inputs?

    Random inputs, deliberately: this measures the selection rule, not the dataset.
    """
    from ultralytics.nn.modules.head import Detect

    net, spec = load_detector(model, device=device)
    set_head(net, "e2e")
    head = [m for m in net.modules() if isinstance(m, Detect)][-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = next(net.parameters()).device

    n_diff = n_tot = 0
    max_iou_gap = 0.0
    with torch.inference_mode():
        for _ in range(n_batches):
            x = torch.rand((batch, 3, canvas, canvas), generator=g).to(dev)
            _set_topk_mode(False)
            a = net(x)
            a = a[0] if isinstance(a, (list, tuple)) else a
            _set_topk_mode(True)
            b = net(x)
            b = b[0] if isinstance(b, (list, tuple)) else b
            _set_topk_mode(False)
            # compare the selected (box, score, class) sets per image
            for i in range(a.shape[0]):
                sa = a[i].float().cpu().numpy()
                sb = b[i].float().cpu().numpy()
                n_tot += 1
                if sa.shape != sb.shape or not np.allclose(sa, sb, atol=1e-5):
                    n_diff += 1
                    m = min(len(sa), len(sb))
                    if m:
                        max_iou_gap = max(max_iou_gap, float(np.abs(sa[:m, 4] - sb[:m, 4]).max()))
    return {"images": n_tot, "images_with_different_selection": n_diff,
            "frac_different": n_diff / max(n_tot, 1), "max_score_delta": max_iou_gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/phase0/topk_probe.json")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    spec = MODELS[args.model]
    weights = str(resolve_weights(spec))

    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    report: dict = {"model": spec.name, "published": spec.published_map}

    LOGGER.info("=== selection disagreement on random inputs ===")
    report["disagreement"] = selection_disagreement(args.model, args.device)
    LOGGER.info(f"  {report['disagreement']}")

    for label, grouped in (("exact_topk", False), ("grouped_topk_engine", True)):
        _set_topk_mode(grouped)
        LOGGER.info(f"=== COCO val, e2e head, {label} ===")
        r = _A.val_one(weights, "e2e", args.data, args.imgsz, args.batch, args.device,
                       args.workers, str(ann))
        report[label] = r["coco"]
        LOGGER.info(f"  {label}: AP={r['coco']['mAP50-95']:.3f} AP50={r['coco']['mAP50']:.3f}")
    _set_topk_mode(False)

    d = report["exact_topk"]["mAP50-95"] - report["grouped_topk_engine"]["mAP50-95"]
    report["exact_minus_grouped_AP"] = d
    report["published_e2e"] = spec.published_map.get("e2e")
    LOGGER.info(f"exact - grouped = {d:+.3f} AP  (published e2e = {report['published_e2e']})")

    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
