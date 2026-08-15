#!/usr/bin/env python3
"""Run inference over a shift grid once and persist de-shifted detections.

Everything downstream reads this cache, so a (model, head) pair is only ever forwarded
once per shift. The oracle's max_shift=1 grid is a strict subset of the 32-px sweep grid,
which is why building the small grid here is never wasted work.

Typical Phase-0/1 use:
    # oracle grid (4 shifts) -- cheap, do it on the full val set
    python scripts/03_build_cache.py --model yolo26n --head e2e --max-shift 1 --step 1
    python scripts/03_build_cache.py --model yolo26n --head nms --max-shift 1 --step 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from shifteq.data import PreprocCache, load_coco_gt, select_subset  # noqa: E402
from shifteq.detstore import DetStore, RunManifest  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner, infer_geometry, log_plan  # noqa: E402
from shifteq.modelzoo import MODELS, assert_unfused, load_detector, resolve_weights, set_head, weight_fingerprint  # noqa: E402
from shifteq.protocol import SHIFT as PROTOCOL_SHIFT, protocol_hash  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, sha256_of_file, shift_grid, write_json  # noqa: E402


def run_id_for(model: str, head: str, max_shift: int, step: int, subset: int | None, pad: int,
               embed_mode: str = "grow") -> str:
    sub = "full" if subset is None else f"n{subset}"
    rid = f"{model}_{head}_m{max_shift}s{step}_{sub}_pad{pad}"
    # "grow" is the frozen default, so its run ids stay as published; anything else is
    # tagged, because a cache built under a different embedding is a different experiment.
    return rid if embed_mode == "grow" else f"{rid}_{embed_mode}"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--head", default="e2e", choices=["e2e", "nms"])
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--subset", type=int, default=None, help="images; omit for all 5000")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--max-shift", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--embed-mode", default=None, choices=["grow", "fit"],
                    help="grow: letterbox to imgsz, canvas imgsz+M (full resolution, larger "
                         "input). fit: letterbox to imgsz-M, canvas imgsz (validated input "
                         "size, slightly lower resolution). Default from shifteq.protocol.")
    ap.add_argument("--pad-value", type=int, default=114,
                    help="canvas/letterbox fill; -1 = replicate, -2 = reflect")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--store-dir", default="cache/dets")


def resolve_paths(args):
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(args.data)
    root = Path(data["path"])
    ann = Path(args.ann) if args.ann else root / "annotations" / "instances_val2017.json"
    img_dir = Path(args.img_dir) if args.img_dir else root / "images" / "val2017"
    return ann, img_dir


def prepare(args):
    """Shared setup for 03/04/05: GT subset, preproc cache, unfused model, runner."""
    ann, img_dir = resolve_paths(args)
    gt_all = load_coco_gt(ann)
    ids = select_subset([int(i) for i in gt_all.image_ids], args.subset, args.subset_seed)
    gt = load_coco_gt(ann, ids)

    spec = MODELS[args.model]
    mode = args.embed_mode or PROTOCOL_SHIFT["embed_mode"]
    geom = infer_geometry(spec, args.max_shift, mode)
    cache = PreprocCache(Path(args.cache_dir) / "preproc", spec, ids, img_dir,
                         pad_value=args.pad_value, tile=geom.tile).build()
    net, spec = load_detector(spec, device=args.device)
    set_head(net, args.head, max_det=args.max_det)
    pp = PostprocCfg(conf=args.conf, iou=args.iou, max_det=args.max_det)
    # The letterbox border and the canvas border are set by the same flag, so the
    # padding ablation changes one thing rather than two.
    pad_mode = {-1: "replicate", -2: "reflect"}.get(args.pad_value, "constant")
    pad = 114.0 if args.pad_value < 0 else float(args.pad_value)
    runner = ShiftRunner(net, spec, cache, geom, pp, args.device, args.batch,
                         pad_value=pad, pad_mode=pad_mode)
    return gt, ids, spec, cache, net, geom, pp, runner, ann


def make_manifest(args, spec, net, geom, ids) -> RunManifest:
    import ultralytics

    w = resolve_weights(spec)
    return RunManifest(
        run_id=run_id_for(args.model, args.head, args.max_shift, args.step, args.subset,
                          args.pad_value, args.embed_mode or PROTOCOL_SHIFT["embed_mode"]),
        model=spec.name, head=args.head,
        weights_sha256=sha256_of_file(w), weight_fingerprint=weight_fingerprint(net),
        imgsz=spec.imgsz, stride=spec.stride, preproc=spec.preproc,
        max_shift=args.max_shift, step=args.step, canvas=geom.canvas, pad_value=args.pad_value,
        conf=args.conf, iou=args.iou, max_det=args.max_det, multi_label=True,
        device=str(args.device), dtype="float32", fused=False,
        fusion_evidence=assert_unfused(net, require_both_heads=False),
        image_ids=list(map(int, ids)),
        ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
        notes=f"protocol=v{__import__('shifteq.protocol', fromlist=['VERSION']).VERSION} hash={protocol_hash()} embed_mode={geom.embed_mode} tile={geom.tile}",
    )


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    gt, ids, spec, cache, net, geom, pp, runner, ann = prepare(args)

    store = DetStore(args.store_dir, run_id_for(args.model, args.head, args.max_shift,
                                                args.step, args.subset, args.pad_value,
                                                geom.embed_mode))
    store.write_manifest(make_manifest(args, spec, net, geom, ids))

    grid = shift_grid(args.max_shift, args.step)
    log_plan(spec, geom, len(ids), len(grid), args.batch)

    t0 = time.perf_counter()
    for k, (dy, dx) in enumerate(grid):
        if store.has(dy, dx) and not args.force:
            LOGGER.info(f"  [{k + 1}/{len(grid)}] delta=({dy},{dx}) cached")
            continue
        t = time.perf_counter()
        d = runner.run_shift(dy, dx)
        store.save(dy, dx, d.counts, d.cls, d.score, d.xyxy)
        el = time.perf_counter() - t0
        LOGGER.info(
            f"  [{k + 1}/{len(grid)}] delta=({dy},{dx}) {d.n_dets:,} dets "
            f"{time.perf_counter() - t:.1f}s | eta {el / (k + 1) * (len(grid) - k - 1) / 60:.1f}m"
        )

    m = store.manifest
    m.shifts_done = [list(s) for s in store.available_shifts()]
    store.write_manifest(m)
    write_json(Path(args.store_dir) / store.run_id / "done.json",
               {"seconds": time.perf_counter() - t0, "shifts": len(grid),
                "mb": store.size_on_disk() / 1e6})
    LOGGER.info(f"cache ready: {store.dir} ({store.size_on_disk() / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
