#!/usr/bin/env python3
"""The gate: is there a measurable, consistent gap between the two heads?

Reads the 1-D sweeps from `10_sweep1d_fft.py` for both heads and asks two questions.

**Is the response grid-locked?** Reported per curve from the spectral test, with the
family-wise correction -- the peak bin was selected by being the largest of 32, so its
naive p-value overstates significance and must not be quoted.

**Is one head measurably less stable?** A range statistic over 64 shifts is noisy, so
"e2e's range is bigger" is not evidence on its own. The test here is **paired across shift
positions**: both heads saw the same images at the same deltas, so for each delta we take
each head's absolute deviation from its own mean and compare the two heads with a paired
permutation test (sign-flip null). That controls for the fact that some deltas are simply
hard for both heads, which an unpaired comparison would count as signal.

Also emits the FFT figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.spectrum import NETWORK_STRIDES  # noqa: E402
from shifteq.utils import LOGGER, read_json, setup_logging, write_json  # noqa: E402

KEYS = ("AP", "AP50", "AP_small", "AP_medium", "AP_large")


def paired_dispersion_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
                           seed: int = 0) -> dict:
    """Is curve `a` more dispersed than curve `b`, paired by shift position?

    Statistic: mean over deltas of (|a - mean(a)| - |b - mean(b)|). Null: the sign of each
    per-delta difference is exchangeable, which is what "the two heads are equally stable"
    implies once each curve is centred on its own mean.
    """
    da = np.abs(a - a.mean())
    db = np.abs(b - b.mean())
    d = da - db
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1.0) / (n_perm + 1.0)
    return {"mean_abs_dev_a": float(da.mean()), "mean_abs_dev_b": float(db.mean()),
            "difference": obs, "p_value": float(p), "n_perm": n_perm}


def figure(data: dict, out: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"matplotlib unavailable: {e}")
        return None

    axes_present = [a for a in ("x", "y") if all(a in d["axes"] for d in data.values())]
    fig, axs = plt.subplots(2, len(axes_present), figsize=(6.2 * len(axes_present), 8))
    axs = np.atleast_2d(axs).reshape(2, len(axes_present))
    colors = {"e2e": "#c1440e", "nms": "#1f6feb"}

    for j, axis in enumerate(axes_present):
        # top: AP(delta)
        for head, d in data.items():
            c = d["axes"][axis]["curves"]["AP"]
            axs[0, j].plot(c["delta"], c["values"], "-o", ms=2.5, lw=1.1,
                           color=colors.get(head, None),
                           label=f"{head} (range {c['range']:.2f})")
        for s in (32, 64):
            axs[0, j].axvline(s, color="k", ls=":", lw=0.8)
        axs[0, j].set_xlabel(f"shift {axis} (px)")
        axs[0, j].set_ylabel("AP (%)")
        axs[0, j].set_title(f"AP(δ{axis}) at 1-px resolution, two full strides")
        axs[0, j].legend(fontsize=8)
        axs[0, j].grid(alpha=0.3)

        # bottom: power spectrum
        for head, d in data.items():
            sp = d["axes"][axis]["spectra"]["AP"]
            per = np.array([p if p is not None else np.inf for p in sp["periods"]])
            pw = np.array(sp["power"])
            m = np.isfinite(per) & (per >= 2)
            axs[1, j].plot(per[m], pw[m], "-o", ms=3, lw=1.1, color=colors.get(head, None),
                           label=f"{head} (stride power {sp['stride_power_total']:.2f})")
        for s in NETWORK_STRIDES:
            axs[1, j].axvline(s, color="g", ls="--", lw=0.9, alpha=0.6)
        axs[1, j].axhline(1.0 / 32, color="k", ls=":", lw=0.8, label="uniform (chance)")
        axs[1, j].set_xscale("log", base=2)
        axs[1, j].set_xlabel("period (px)  — green lines = network strides 2/4/8/16/32")
        axs[1, j].set_ylabel("power fraction")
        axs[1, j].set_title("power spectrum of AP(δ)")
        axs[1, j].legend(fontsize=8)
        axs[1, j].grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n")
    ap.add_argument("--dir", default="results/sweep1d")
    ap.add_argument("--subset", type=int, default=1000)
    ap.add_argument("--max-shift", type=int, default=63)
    ap.add_argument("--pad-value", type=int, default=114)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--out", default="results/head_gap.json")
    ap.add_argument("--fig", default="results/figures/fft_head_comparison.png")
    args = ap.parse_args()

    setup_logging()
    d = Path(args.dir)
    data = {}
    for head in ("e2e", "nms"):
        f = d / f"{args.model}_{head}_m{args.max_shift}s1_n{args.subset}_pad{args.pad_value}_fft.json"
        if not f.exists():
            raise SystemExit(f"missing {f}; run scripts/10_sweep1d_fft.py --head {head} first")
        data[head] = read_json(f)

    report: dict = {"model": args.model, "n_images": data["e2e"]["n_images"],
                    "max_shift": args.max_shift, "grid_locked": {}, "head_gap": {}}

    LOGGER.info("=== is the response grid-locked? (family-wise corrected) ===")
    n_locked = n_tot = 0
    for head, dd in data.items():
        for axis, ax in dd["axes"].items():
            for key in KEYS:
                sp = ax["spectra"][key]
                n_tot += 1
                locked = sp["interpretation"] == "grid-locked"
                n_locked += locked
                report["grid_locked"][f"{head}/{axis}/{key}"] = {
                    "peak_period_px": sp["peak"]["period_px"],
                    "p_peak_fwer": sp["p_peak_fwer"],
                    "stride_power": sp["stride_power_total"],
                    "enrichment": sp["enrichment"],
                    "p_stride_fwer": sp["p_stride_fwer"],
                    "parity_even_minus_odd": sp["parity"]["even_minus_odd_AP"],
                    "parity_p": sp["parity"]["p_value"],
                    "grid_locked": locked,
                }
    report["n_curves_grid_locked"] = n_locked
    report["n_curves"] = n_tot
    LOGGER.info(f"  {n_locked}/{n_tot} curves grid-locked "
                f"(stride bins hold {100 / 32 * len(NETWORK_STRIDES):.0f}% by chance)")

    LOGGER.info("")
    LOGGER.info("=== head gap: is e2e measurably less stable? (paired across δ) ===")
    wins = losses = 0
    for axis in ("x", "y"):
        for key in KEYS:
            a = np.array(data["e2e"]["axes"][axis]["curves"][key]["values"])
            b = np.array(data["nms"]["axes"][axis]["curves"][key]["values"])
            t = paired_dispersion_test(a, b, n_perm=args.permutations)
            ra = data["e2e"]["axes"][axis]["curves"][key]["range"]
            rb = data["nms"]["axes"][axis]["curves"][key]["range"]
            sig = t["p_value"] < 0.05
            direction = "e2e worse" if t["difference"] > 0 else "nms worse"
            wins += sig and t["difference"] > 0
            losses += sig and t["difference"] < 0
            report["head_gap"][f"{axis}/{key}"] = {
                "range_e2e": ra, "range_nms": rb, **t,
                "significant": bool(sig), "direction": direction,
            }
            LOGGER.info(
                "  {:>1s} {:<10s} range {:5.3f} vs {:5.3f} | mean|dev| {:.4f} vs {:.4f} "
                "(Δ{:+.4f}, p={:.4f}) {}".format(
                    axis, key, ra, rb, t["mean_abs_dev_a"], t["mean_abs_dev_b"],
                    t["difference"], t["p_value"], "**" + direction + "**" if sig else "n.s."
                )
            )
    report["n_significant_e2e_worse"] = wins
    report["n_significant_nms_worse"] = losses

    f = figure(data, Path(args.fig))
    if f:
        report["figure"] = f
        LOGGER.info(f"wrote {f}")

    LOGGER.info("")
    LOGGER.info(f"GATE: {n_locked}/{n_tot} curves grid-locked; "
                f"e2e significantly less stable on {wins}/{len(report['head_gap'])} curves, "
                f"nms on {losses}.")
    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
