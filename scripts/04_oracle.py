#!/usr/bin/env python3
"""Oracle envelope dAP = AP_best - AP_worst (Manfredi & Wang, Algorithm 1).

Reads the detection cache built by 03_build_cache.py, so it is pure CPU/GPU accumulation
with no forward passes. Run `--dry-run` first: it times a single AP evaluation and
extrapolates the K x (M+1)^2 x N budget, which is the number that decides whether you run
this on 5000 images or 1000.

Comparability note: max_shift=1 is the setting Manfredi & Wang report (their Table 1), so
that is the only setting whose dAP can be put next to their RetinaNet / FasterRCNN /
CenterNet numbers. Larger max_shift changes the metric's meaning -- the envelope widens
mechanically with the number of shifts available per image -- so those runs must be
labelled with their M and never compared across M.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.cocolite import class_map_inverse, npig_per_cat  # noqa: E402
from shifteq.data import AREA_KEYS, load_coco_gt  # noqa: E402
from shifteq.detstore import DetStore, coco80_to_coco91  # noqa: E402
from shifteq.pipeline import match_shifts_parallel  # noqa: E402
from shifteq.stability import build_gt_index  # noqa: E402
from shifteq.oracle import (  # noqa: E402
    assignment_histogram, build_union, delta_ap, estimate_cost, greedy,
    make_accumulator, summary_at_assignment,
)
from shifteq.utils import LOGGER, set_deterministic, setup_logging, shift_grid, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default="cache/dets")
    ap.add_argument("--run-id", required=True, help="e.g. yolo26n_e2e_m1s1_full_pad114")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--objective", default="AP50", choices=["AP", "AP50"],
                    help="scalar the greedy search optimises. AP50 is the paper's choice "
                         "(Sec. 3.1: 'In practice, we use AP50 as the metric'); AP is then "
                         "read off at that assignment. Changing this makes dAP incomparable "
                         "to their Table 1.")
    ap.add_argument("--area", default="all", choices=list(AREA_KEYS),
                    help="area range the objective is computed on; 'small' gives a genuine "
                         "small-object oracle at the cost of a second full pass")
    ap.add_argument("--iters", type=int, default=1, help="K in Algorithm 1")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="time one AP call, print the budget, exit")
    ap.add_argument("--directions", default="best,worst")
    ap.add_argument("--no-size-summary", action="store_true",
                    help="skip matching the small/medium/large ranges (saves ~3x match time and RAM)")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for the matching stage (default: cores-2, cap 16)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for every artifact this run writes. Without it a K=2 run "
                         "would overwrite the committed K=1 assignment .npy files, which "
                         "are evidence for the published numbers.")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)

    store = DetStore(args.store_dir, args.run_id)
    man = store.manifest
    if man.max_shift != 1:
        LOGGER.warning(
            f"max_shift={man.max_shift}: dAP is only comparable to Manfredi & Wang at max_shift=1. "
            "Label this run with its M and do not compare it across M."
        )

    if args.ann:
        ann = Path(args.ann)
    else:
        from ultralytics.data.utils import check_det_dataset

        ann = Path(check_det_dataset(args.data)["path"]) / "annotations" / "instances_val2017.json"

    gt = load_coco_gt(ann, man.image_ids)
    cmi = class_map_inverse(gt, coco80_to_coco91())
    grid = shift_grid(man.max_shift, man.step)
    missing = [s for s in grid if not store.has(*s)]
    if missing:
        raise SystemExit(f"detection cache incomplete, missing shifts: {missing}")

    areas = (args.area,) if args.no_size_summary else tuple(dict.fromkeys((args.area, *AREA_KEYS)))
    npig = {a: npig_per_cat(gt, a) for a in areas}
    LOGGER.info(f"matching {len(grid)} shifts x {gt.num_images} images over areas {areas} ...")
    t0 = time.perf_counter()
    per_shift = match_shifts_parallel(
        gt, build_gt_index(gt), cmi, npig, store, grid, areas, workers=args.workers
    )
    LOGGER.info(f"matching done in {(time.perf_counter() - t0) / 60:.1f} min")
    u = build_union(per_shift[args.area], npig[args.area], gt.num_images, gt.num_cats)
    acc = make_accumulator(u, args.backend, args.device)
    LOGGER.info(f"accumulator backend: {acc.name}")

    # --- cost probe -------------------------------------------------------
    mask = acc.new_mask(np.zeros(gt.num_images, dtype=np.int64))
    acc.ap(mask)                       # warm up
    t = time.perf_counter()
    reps = 5
    for _ in range(reps):
        acc.ap(mask)
    spa = (time.perf_counter() - t) / reps
    cost = estimate_cost(gt.num_images, len(grid), args.iters, spa)
    LOGGER.info(
        f"one AP call = {spa * 1000:.1f} ms -> {cost['ap_calls']:,} calls per direction "
        f"= {cost['hours']:.2f} h ({2 * cost['hours']:.2f} h for best+worst)"
    )
    if args.dry_run:
        write_json(args.out or f"results/oracle/{args.run_id}_dryrun.json",
                   {"run_id": args.run_id, "area": args.area, "objective": args.objective, **cost})
        return

    baseline_ap, baseline_ap50 = acc.ap(mask)
    LOGGER.info(f"baseline (all images at delta=(0,0)): AP={baseline_ap:.5f} AP50={baseline_ap50:.5f}")

    out_dir = Path("results/oracle")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Artifact names must not collide across runs that differ only in a parameter.
    # `iters` is the live case: --iters 2 reuses the same run_id and would silently
    # overwrite the K=1 assignments the published Delta-AP rests on.
    tag = args.tag or ("" if args.iters == 1 else f"K{args.iters}")
    stem = f"{args.run_id}_{tag}" if tag else args.run_id
    if tag:
        LOGGER.info(f"artifacts tagged '{tag}' -> {stem}_*")
    results = {}
    for direction in args.directions.split(","):
        LOGGER.info(f"=== greedy {direction} ({args.objective}, area={args.area}) ===")
        res = greedy(
            acc, grid, direction=direction, objective=args.objective, iters=args.iters,
            checkpoint=out_dir / f"{stem}_{direction}_ckpt.json",
        )
        results[direction] = res
        LOGGER.info(
            f"  -> AP={res.ap:.5f} AP50={res.ap50:.5f} in {res.seconds / 60:.1f} min "
            f"({res.n_ap_calls:,} AP calls)"
        )

    report = {
        "run_id": args.run_id, "model": man.model, "head": man.head,
        "max_shift": man.max_shift, "step": man.step, "n_images": gt.num_images,
        "objective": args.objective, "area": args.area, "iters": args.iters,
        "backend": acc.name, "seconds_per_ap": spa,
        # Content hash of the detections consumed, so this result can be traced to its
        # exact input rather than to a run_id that may have been rebuilt since.
        "detections": store.fingerprint(),
        "baseline": {"AP": baseline_ap, "AP50": baseline_ap50},
        "shifts": [list(s) for s in grid],
        # Their Table 1 is max_shift=1 over ALL areas. A small-object envelope sits on a
        # different AP scale entirely (PROTOCOL section 12.1), so M=1 alone does not make
        # a run comparable to them -- quoting dAP_small against their table would be wrong.
        "comparable_to_manfredi_wang": man.max_shift == 1 and args.area == "all",
    }
    for d, r in results.items():
        report[d] = {
            "AP": r.ap, "AP50": r.ap50, "seconds": r.seconds, "n_ap_calls": r.n_ap_calls,
            "iterations": r.iterations, "shift_histogram": assignment_histogram(r),
        }
        if not args.no_size_summary:
            report[d]["summary_at_assignment"] = summary_at_assignment(
                gt, per_shift, np.asarray(r.assignment), npig
            )
        np.save(out_dir / f"{stem}_{d}_assignment.npy", np.asarray(r.assignment, dtype=np.int16))
    if {"best", "worst"} <= set(results):
        report["delta"] = delta_ap(results["best"], results["worst"])
        LOGGER.info(
            f"dAP = {report['delta']['dAP'] * 100:.2f} AP   "
            f"dAP50 = {report['delta']['dAP50'] * 100:.2f} AP50   "
            f"(best {report['delta']['AP_best'] * 100:.2f} / worst {report['delta']['AP_worst'] * 100:.2f})"
        )

    out = args.out or f"results/oracle/{stem}_{args.objective}_{args.area}.json"
    write_json(out, report)
    LOGGER.info(f"wrote {out}")


if __name__ == "__main__":
    main()
