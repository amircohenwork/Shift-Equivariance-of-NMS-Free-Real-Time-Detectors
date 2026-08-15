#!/usr/bin/env python3
"""What is the preprocessing cache actually worth? Measured, not assumed.

The concern this addresses: if you re-decode and re-letterbox every image once per shift,
the 32-px sweep does 256x the JPEG work, the GPU idles behind libjpeg, and you conclude the
study needs more compute than you have -- expensively wrong.

`shifteq/data.py` decodes once into a uint8 memmap and shifts on the GPU, so decode is
outside the shift loop by construction. This script measures both paths **end to end on the
same images and the same model**, rather than extrapolating from a decode microbenchmark.
An earlier version of this script did extrapolate, compared decode-alone against the full
cached pipeline, and produced a meaningless speedup; if you touch the arithmetic here,
compare like with like.

Both paths are timed with the *same* GPU work (forward + NMS + de-shift). The only
difference is where the pixels come from:

  naive   ThreadPool of `--decode-workers` doing cv2 decode + letterbox, prefetching one
          batch ahead so the comparison is against a competent dataloader, not a strawman
  cached  slice a uint8 memmap, pin, H2D, shift on device
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.data import PreprocCache, _load_and_resize, load_coco_gt, select_subset  # noqa: E402
from shifteq.infer import PostprocCfg, ShiftRunner, infer_geometry  # noqa: E402
from shifteq.modelzoo import MODELS, load_detector, set_head  # noqa: E402
from shifteq.shift import embed  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_cached(runner: ShiftRunner, shifts: list[tuple[int, int]]) -> float:
    runner.run_shift(*shifts[0])          # warm CUDA + page cache
    _sync()
    t = time.perf_counter()
    for s in shifts:
        runner.run_shift(*s)
    _sync()
    return (time.perf_counter() - t) / len(shifts)


def bench_naive(runner: ShiftRunner, files: list[str], imgsz: int, preproc: str,
                shifts: list[tuple[int, int]], workers: int, batch: int) -> float:
    """Same GPU work, but every batch is decoded from JPEG on every shift."""
    geom, net, pp = runner.geom, runner.net, runner.pp
    dev = runner.device

    def load_batch(chunk: list[str]) -> torch.Tensor:
        ims = [_load_and_resize(f, imgsz, preproc)[0] for f in chunk]
        arr = np.stack([np.ascontiguousarray(im.transpose(2, 0, 1)[::-1]) for im in ims])
        return torch.from_numpy(arr)

    chunks = [files[i : i + batch] for i in range(0, len(files), batch)]

    def one_shift(dy: int, dx: int) -> None:
        buf = None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # prefetch one batch ahead, as any real dataloader would
            fut = ex.submit(load_batch, chunks[0])
            for k in range(len(chunks)):
                cpu = fut.result()
                if k + 1 < len(chunks):
                    fut = ex.submit(load_batch, chunks[k + 1])
                imgs = cpu.to(dev, non_blocking=False)
                buf = embed(imgs, dy, dx, geom, runner.pad_value, out=buf)
                with torch.inference_mode():
                    raw = net(buf)
                runner._postprocess(raw, "e2e" if getattr(net, "end2end", False) else "nms")

    one_shift(*shifts[0])                 # warm
    _sync()
    t = time.perf_counter()
    for s in shifts:
        one_shift(*s)
    _sync()
    return (time.perf_counter() - t) / len(shifts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=sorted(MODELS))
    ap.add_argument("--head", default="e2e", choices=["e2e", "nms"])
    ap.add_argument("--data", default="coco.yaml")
    ap.add_argument("--ann", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument("--timed-shifts", type=int, default=3)
    ap.add_argument("--sweep-shifts", type=int, default=256)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="results/phase0/cache_benefit.json")
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
    ids = select_subset([int(i) for i in gt_all.image_ids], args.subset, 0)
    files = [str(img_dir / f"{i:012d}.jpg") for i in ids]

    cache = PreprocCache(Path(args.cache_dir) / "preproc", spec, ids, img_dir).build()
    net, spec = load_detector(spec, device=args.device)
    set_head(net, args.head)
    geom = infer_geometry(spec, 31)
    runner = ShiftRunner(net, spec, cache, geom, PostprocCfg(), args.device, args.batch)

    shifts = [(k % 3, (k + 1) % 3) for k in range(args.timed_shifts)]
    N, S, W = len(ids), args.sweep_shifts, args.decode_workers

    LOGGER.info(f"timing {args.timed_shifts} shifts x {N} images, both paths ...")
    cached_s = bench_cached(runner, shifts)
    LOGGER.info(f"  cached: {cached_s:.2f} s/shift ({cached_s / N * 1000:.3f} ms/image)")
    naive_s = bench_naive(runner, files, spec.imgsz, spec.preproc, shifts, W, args.batch)
    LOGGER.info(f"  naive : {naive_s:.2f} s/shift ({naive_s / N * 1000:.3f} ms/image, "
                f"{W} decode workers)")

    # one-off cost of building the cache, measured the same way it is built
    t = time.perf_counter()
    PreprocCache(Path(args.cache_dir) / "preproc_bench", spec, ids[: min(200, N)],
                 img_dir).build(workers=W, force=True)
    build_s_per_img = (time.perf_counter() - t) / min(200, N)

    cached_h = cached_s * S / 3600
    naive_h = naive_s * S / 3600
    build_h = build_s_per_img * N / 3600

    report = {
        "model": spec.name, "head": args.head, "n_images": N, "sweep_shifts": S,
        "decode_workers": W, "batch": args.batch,
        "s_per_shift_cached": cached_s, "s_per_shift_naive": naive_s,
        "ms_per_image_cached": cached_s / N * 1000,
        "ms_per_image_naive": naive_s / N * 1000,
        "sweep_hours_cached": cached_h, "sweep_hours_naive": naive_h,
        "cache_build_hours": build_h,
        "speedup": naive_s / cached_s,
        "overhead_pct": 100 * (naive_s - cached_s) / cached_s,
        "cache_gb": float(np.prod(cache.shape)) / 1e9,
    }

    LOGGER.info("")
    LOGGER.info(f"full sweep, {N} images x {S} shifts, per (model, head):")
    LOGGER.info(f"  cached : {cached_h:.2f} h  (+ {build_h * 60:.1f} min one-off cache build)")
    LOGGER.info(f"  naive  : {naive_h:.2f} h  ({report['overhead_pct']:+.0f}%)")
    LOGGER.info(f"  speedup: {report['speedup']:.2f}x   cache size {report['cache_gb']:.2f} GB")
    LOGGER.info("")
    LOGGER.info(f"Against a 24 GPU-hour budget: cached fits {int(24 / max(cached_h, 1e-9))} "
                f"configurations, naive fits {int(24 / max(naive_h, 1e-9))}.")

    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
