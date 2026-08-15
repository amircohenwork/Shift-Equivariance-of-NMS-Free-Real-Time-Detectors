#!/usr/bin/env python3
"""Uniform-shift curve AP(delta) over the 32-px grid, plus flicker and box jitter.

Every image gets the same delta, so each point is an ordinary COCO AP. This is the metric
family to quote for "what does a translation cost in practice"; the oracle envelope in
04_oracle.py answers a different question and lives in a different table.

Cost is dominated by forward passes: |grid| x N. The preprocessing cache means the
dataloader is not in that loop, so at step 2 (16x16 = 256 shifts) on 1000 images this is
256k forwards -- minutes on a modern GPU for a nano model, not hours.

    # headline sweep
    python scripts/05_sweep.py --model yolo26n --head e2e --subset 1000 --step 2
    # fidelity check the proposal promises: dense grid on a small image set
    python scripts/05_sweep.py --model yolo26n --head e2e --subset 100 --step 1 --tag dense
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse  # noqa: E402
from shifteq.detstore import DetStore, coco80_to_coco91  # noqa: E402
from shifteq.stability import StabilityAccumulator  # noqa: E402
from shifteq.stabindex import SIAccumulator  # noqa: E402
from shifteq.sweep import curve_stats, run_sweep, subsample_fidelity  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_b = import_module("03_build_cache")


def main():
    ap = argparse.ArgumentParser()
    _b.add_common_args(ap)
    ap.set_defaults(max_shift=31, step=2, subset=1000)
    ap.add_argument("--tau", type=float, default=0.25, help="score threshold for flicker/jitter")
    ap.add_argument("--persist-dets", action="store_true",
                    help="also write per-shift detections (~30 MB/shift on full val)")
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--si-mode", default="ref", choices=["ref", "adjacent", "off"],
                    help="Stability Index pairing: vs delta=(0,0), vs the previous grid point, or off")
    ap.add_argument("--workers", type=int, default=None, help="CPU workers for match+stability (default: cores-2, cap 16)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/sweep")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    gt, ids, spec, cache, net, geom, pp, runner, ann = _b.prepare(args)
    cmi = class_map_inverse(gt, coco80_to_coco91())

    rid = _b.run_id_for(args.model, args.head, args.max_shift, args.step, args.subset,
                        args.pad_value, geom.embed_mode)
    rid = f"{rid}_{args.tag}" if args.tag else rid
    out_dir = Path(args.out_dir)
    jsonl = out_dir / f"{rid}.jsonl"

    store = None
    if args.persist_dets:
        store = DetStore(args.store_dir, rid)
        store.write_manifest(_b.make_manifest(args, spec, net, geom, ids))

    stab = None if args.no_stability else StabilityAccumulator(gt, cmi, tau=args.tau)
    si = None if (stab is None or args.si_mode == 'off') else SIAccumulator(gt, stab.idx, mode=args.si_mode)

    rows = run_sweep(
        runner, gt, cmi, jsonl,
        max_shift=args.max_shift, step=args.step,
        stability=stab, stab_index=si, store=store, workers=args.workers,
    )

    report = {
        "run_id": rid, "model": spec.name, "head": args.head, "n_images": gt.num_images,
        "max_shift": args.max_shift, "step": args.step, "pad_value": args.pad_value,
        "canvas": geom.canvas, "n_shifts": len(rows),
        "curve": {k: v for key in ("AP", "AP50", "AP_small", "AP_medium", "AP_large")
                  for k, v in curve_stats(rows, key).items()},
        "note": ("AP_range here is the spread over UNIFORM shifts. It is not dAP: the oracle "
                 "envelope selects a shift per image and is systematically wider."),
    }
    if args.step == 1:
        report["subsample_fidelity"] = {str(s): subsample_fidelity(rows, s) for s in (2, 4)}
    if si is not None:
        report["stability_index"] = si.summary()
        LOGGER.info(f"SI: {report['stability_index']}")
    if stab is not None:
        report["stability"] = stab.summary()
        LOGGER.info(f"stability: {report['stability']}")

    write_json(out_dir / f"{rid}_summary.json", report)
    LOGGER.info(f"wrote {out_dir / f'{rid}_summary.json'}")
    LOGGER.info(
        f"AP over uniform shifts: min={report['curve']['AP_min']:.4f} "
        f"max={report['curve']['AP_max']:.4f} range={report['curve']['AP_range']:.4f} "
        f"(at delta=0: {report['curve']['AP_at_0']:.4f})"
    )


if __name__ == "__main__":
    main()
