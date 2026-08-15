#!/usr/bin/env python3
"""1-D shift sweep at full 1-px resolution, and the FFT that decides grid-locked vs noise.

The 2-D grid in `05_sweep.py` runs at step 2 to keep 256 shifts affordable, which is fine
for a range statistic but weak for a periodicity claim. A 1-D line is cheap enough to run at
step 1 -- 64 shifts instead of 1024 -- so this is where the spectral question gets answered.

Design: sweep delta = (0, dx) and (dy, 0) for 0..63, i.e. **two full 32-px strides at 1-px
resolution**. With 64 samples the FFT bins land exactly on the periods of interest --
period 32 is bin 2, 16 is bin 4, 8 is bin 8 -- so a stride-locked response produces power in
a single bin with no leakage.

Everything is broken out by object size, because that is where the two heads differ most and
because small objects have the most to gain or lose from grid alignment.

Significance is a permutation null over delta (see `shifteq/spectrum.py`), which preserves
the marginal AP distribution and destroys only the ordering -- so it controls for the
finite-image-sample noise floor rather than assuming it away.

    python scripts/10_sweep1d_fft.py --model yolo26n --head e2e --subset 1000
    python scripts/10_sweep1d_fft.py --model yolo26n --head nms --subset 1000
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shifteq.cocolite import class_map_inverse  # noqa: E402
from shifteq.detstore import coco80_to_coco91  # noqa: E402
from shifteq.spectrum import NETWORK_STRIDES, analyse, parity_effect, stride_report  # noqa: E402
from shifteq.sweep import run_sweep  # noqa: E402
from shifteq.utils import LOGGER, set_deterministic, setup_logging, write_json  # noqa: E402

_b = import_module("03_build_cache")

KEYS = ("AP", "AP50", "AP_small", "AP_medium", "AP_large")


def main():
    ap = argparse.ArgumentParser()
    _b.add_common_args(ap)
    ap.set_defaults(max_shift=63, step=1, subset=1000)
    ap.add_argument("--axis", default="both", choices=["x", "y", "both"])
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results/sweep1d")
    args = ap.parse_args()

    setup_logging()
    set_deterministic(0)
    gt, ids, spec, cache, net, geom, pp, runner, ann = _b.prepare(args)
    cmi = class_map_inverse(gt, coco80_to_coco91())

    n = args.max_shift + 1
    LOGGER.info(f"1-D sweep: {n} shifts per axis at step 1 "
                f"({n / spec.stride:.1f} strides), canvas={geom.canvas}, N={gt.num_images}")

    out_dir = Path(args.out_dir)
    base = _b.run_id_for(args.model, args.head, args.max_shift, args.step, args.subset,
                         args.pad_value, geom.embed_mode)
    base = f"{base}_{args.tag}" if args.tag else base

    axes = ["x", "y"] if args.axis == "both" else [args.axis]
    report: dict = {"run_id": base, "model": spec.name, "head": args.head,
                    "n_images": gt.num_images, "max_shift": args.max_shift,
                    "canvas": geom.canvas, "stride": spec.stride,
                    "permutations": args.permutations, "axes": {}}

    for axis in axes:
        grid = [(0, d) if axis == "x" else (d, 0) for d in range(n)]
        rows = run_sweep(runner, gt, cmi, out_dir / f"{base}_{axis}.jsonl",
                         stability=None, workers=args.workers, grid=grid)
        # rows come back sorted by (dy, dx); recover the sweep coordinate
        by_d = {(r["dx"] if axis == "x" else r["dy"]): r for r in rows}
        deltas = np.array(sorted(by_d))
        if not np.array_equal(deltas, np.arange(n)):
            LOGGER.warning(f"axis {axis}: expected 0..{n - 1}, got {len(deltas)} points")

        ax_out: dict = {"n_points": len(deltas), "curves": {}, "spectra": {}}
        for key in KEYS:
            v = np.array([by_d[int(d)][key] for d in deltas], dtype=np.float64) * 100
            spec_res = analyse(deltas, v, n_permutations=args.permutations)
            sr = stride_report(spec_res, strides=NETWORK_STRIDES)
            par = parity_effect(v, n_permutations=args.permutations)
            ax_out["curves"][key] = {
                "delta": deltas.tolist(), "values": v.tolist(),
                "min": float(v.min()), "max": float(v.max()),
                "range": float(v.max() - v.min()), "std": float(v.std(ddof=1)),
                "at_0": float(v[0]),
            }
            ax_out["spectra"][key] = {
                "peak": sr["peak"], "strides": sr["strides"],
                "stride_power_total": sr["stride_power_total"],
                "interpretation": sr["interpretation"],
                "enrichment": sr["enrichment"],
                "expected_by_chance": sr["expected_by_chance"],
                "min_stride_p_uncorrected": sr["min_stride_p_uncorrected"],
                "p_stride_fwer": sr["p_stride_fwer"],
                "p_peak_fwer": sr["p_peak_fwer"],
                "parity": par,
                "power": spec_res.power.tolist(),
                "periods": [None if not np.isfinite(p) else float(p) for p in spec_res.periods],
                "p_values": spec_res.p_values.tolist(),
            }
            pw = "/".join(f"{sr['strides'][str(k)]['power_frac']:.2f}" for k in NETWORK_STRIDES)
            LOGGER.info(
                "{:>3s} {:<10s} range={:5.3f} peak={:5.1f}px p_fwer={:.4f} | 2/4/8/16/32 = {} "
                "sum={:.2f} enrich={:.1f}x p_stride={:.4f} | even-odd={:+.3f} p={:.4f} -> {}".format(
                    axis, key, ax_out["curves"][key]["range"],
                    sr["peak"]["period_px"], sr["peak"]["p_value"], pw,
                    sr["stride_power_total"], sr["enrichment"], sr["p_stride_fwer"],
                    par["even_minus_odd_AP"], par["p_value"], sr["interpretation"],
                )
            )
        report["axes"][axis] = ax_out

    write_json(out_dir / f"{base}_fft.json", report)
    LOGGER.info(f"wrote {out_dir / f'{base}_fft.json'}")


if __name__ == "__main__":
    main()
