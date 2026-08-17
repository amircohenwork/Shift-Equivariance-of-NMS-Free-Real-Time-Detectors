#!/usr/bin/env python3
"""Padding ablation: is the shift response border-reading, or aliasing?

Kayhan & van Gemert (CVPR'20) showed that a constant padding border is an implicit
*absolute position* cue -- a convolution can read distance-to-frame out of it, which is
exactly the kind of absolute-position information a translation-equivariant model is not
supposed to have. That gives a competing explanation for everything measured so far: maybe
AP(delta) wobbles because the object's distance to a constant gray frame changed, not
because strided downsampling aliases.

The two explanations make opposite predictions about this ablation, which is why it is
worth the GPU time:

    aliasing      -> the stride periodicity survives when the border stops being constant
    border cue    -> the stride periodicity weakens or disappears

The four arms split into two groups, and the grouping is the design:

    constant border (an absolute-position cue is available)
        114   gray -- the Ultralytics val default, and the frozen protocol
          0   black -- Manfredi & Wang's canvas; controls for the specific *value*
    content-derived border (no absolute-position cue)
         -1   replicate -- edge pixels extended outward
         -2   reflect   -- mirrored content

If 114 and 0 agree with each other and both differ from replicate/reflect, the cue is doing
the work. If all four agree, the constant is not the mechanism. Comparing 114 against 0
alone cannot distinguish those, because both are constants.

Note the padding flag sets *both* borders -- the letterbox bars when the preprocessing cache
is built, and the canvas frame at embed time -- so this changes one thing, not two.

    python scripts/13_pad_ablation.py --model yolo26n --subset 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.spectrum import dispersion, paired_dispersion_test  # noqa: E402
from shifteq.utils import LOGGER, read_json, setup_logging, write_json  # noqa: E402

KEYS = ("AP", "AP50", "AP_small", "AP_medium", "AP_large")
PAD_NAMES = {114: "gray 114 (protocol)", 0: "black 0", -1: "replicate", -2: "reflect"}
#: which arms still hand the network a constant frame to measure distance against
CONSTANT_BORDER = (114, 0)


def load(d: Path, model: str, head: str, max_shift: int, subset: str, pad: int) -> dict | None:
    sub = "full" if str(subset).lower() in ("full", "none", "all") else f"n{subset}"
    f = d / f"{model}_{head}_m{max_shift}s1_{sub}_pad{pad}_fft.json"
    if not f.exists():
        LOGGER.warning(f"missing {f.name} -- pad {pad} / head {head} skipped")
        return None
    r = read_json(f)
    r["_file"] = str(f)
    return r


def curve(arm: dict, axis: str, key: str) -> np.ndarray | None:
    ax = arm["axes"].get(axis)
    return None if ax is None else np.array(ax["curves"][key]["values"], dtype=np.float64)


def figure(data: dict, pads: list[int], out: Path, axis: str = "x") -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"matplotlib unavailable: {e}")
        return None

    heads = [h for h in ("e2e", "nms") if any((h, p) in data for p in pads)]
    if not heads:
        return None
    fig, axs = plt.subplots(2, len(heads), figsize=(6.4 * len(heads), 8.4), squeeze=False)
    cmap = {114: "#555555", 0: "#000000", -1: "#c1440e", -2: "#1f6feb"}

    for j, head in enumerate(heads):
        for p in pads:
            arm = data.get((head, p))
            if arm is None or axis not in arm["axes"]:
                continue
            v = curve(arm, axis, "AP")
            ls = "-" if p in CONSTANT_BORDER else "--"
            axs[0][j].plot(np.arange(len(v)), v - v.mean(), ls, lw=1.2, color=cmap.get(p),
                           label=f"{PAD_NAMES.get(p, p)} (AP {v.mean():.2f})")
            sp = arm["axes"][axis]["spectra"]["AP"]
            axs[1][j].bar(pads.index(p), sp["stride_power_total"], 0.6, color=cmap.get(p),
                          hatch=None if p in CONSTANT_BORDER else "//")
        axs[0][j].set_xlabel(f"shift {axis} (px)")
        axs[0][j].set_ylabel("AP - mean(AP) (points)")
        axs[0][j].set_title(f"{head}: AP(δ{axis}) shape, AP level removed")
        axs[0][j].legend(fontsize=8)
        axs[0][j].grid(alpha=0.3)
        axs[1][j].axhline(5 / 32, color="k", ls=":", lw=1.0, label="uniform (chance)")
        axs[1][j].set_xticks(range(len(pads)))
        axs[1][j].set_xticklabels([PAD_NAMES.get(p, str(p)) for p in pads],
                                  rotation=20, ha="right", fontsize=8)
        axs[1][j].set_ylabel("power fraction in stride bins")
        axs[1][j].set_title(f"{head}: grid-locking vs padding (hatched = no constant cue)")
        axs[1][j].legend(fontsize=8)
        axs[1][j].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n")
    ap.add_argument("--dir", default="results/sweep1d")
    ap.add_argument("--subset", default="1000")
    ap.add_argument("--max-shift", type=int, default=63)
    ap.add_argument("--pads", type=int, nargs="+", default=[114, 0, -1, -2])
    ap.add_argument("--baseline", type=int, default=114)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--out", default="results/pad_ablation.json")
    ap.add_argument("--fig", default="results/figures/pad_ablation.png")
    args = ap.parse_args()

    setup_logging()
    d = Path(args.dir)
    data: dict[tuple[str, int], dict] = {}
    for head in ("e2e", "nms"):
        for p in args.pads:
            r = load(d, args.model, head, args.max_shift, args.subset, p)
            if r is not None:
                data[(head, p)] = r
    if not data:
        raise SystemExit("no padding runs found -- see `make pad-ablation`")

    report: dict = {"model": args.model, "subset": args.subset, "baseline_pad": args.baseline,
                    "pads": args.pads, "arms": {}, "gridlock": {}, "vs_baseline": {}}

    LOGGER.info("=== does the padding change the grid-locking? ===")
    LOGGER.info(f"  {'head':<5s} {'padding':<20s} {'AP@δ0':>7s} {'meanAP':>7s} {'mad':>7s} "
                f"{'cv%':>6s} {'stride pw':>10s} {'enrich':>7s} {'p_fwer':>8s} {'locked':>7s}")
    for head in ("e2e", "nms"):
        for p in args.pads:
            arm = data.get((head, p))
            if arm is None:
                continue
            n_lock = n_cur = 0
            per_curve = {}
            for axis, ax in arm["axes"].items():
                for key in KEYS:
                    s = ax["spectra"][key]
                    n_cur += 1
                    lock = s["interpretation"] == "grid-locked"
                    n_lock += lock
                    per_curve[f"{axis}/{key}"] = {
                        "stride_power": s["stride_power_total"], "enrichment": s["enrichment"],
                        "p_stride_fwer": s["p_stride_fwer"], "grid_locked": lock,
                        "parity_even_minus_odd": s["parity"]["even_minus_odd_AP"],
                        "parity_p": s["parity"]["p_value"],
                        **dispersion(np.array(ax["curves"][key]["values"])),
                    }
            k = f"{head}/pad{p}"
            report["arms"][k] = {"file": arm["_file"], "n_images": arm["n_images"],
                                 "constant_border": p in CONSTANT_BORDER, "curves": per_curve}
            report["gridlock"][k] = {"n_locked": n_lock, "n_curves": n_cur}

            ax0 = "x" if "x" in arm["axes"] else next(iter(arm["axes"]))
            s0 = arm["axes"][ax0]["spectra"]["AP"]
            dd = dispersion(curve(arm, ax0, "AP"))
            LOGGER.info(
                f"  {head:<5s} {PAD_NAMES.get(p, str(p)):<20s} "
                f"{arm['axes'][ax0]['curves']['AP']['at_0']:>7.2f} {dd['mean']:>7.2f} "
                f"{dd['mean_abs_dev']:>7.4f} {dd['cv_percent']:>6.3f} "
                f"{s0['stride_power_total']:>10.2f} {s0['enrichment']:>6.1f}x "
                f"{s0['p_stride_fwer']:>8.4f} {n_lock:>3d}/{n_cur:<3d}"
            )

    LOGGER.info("")
    LOGGER.info(f"=== each padding vs the pad-{args.baseline} protocol (paired across δ) ===")
    for head in ("e2e", "nms"):
        base = data.get((head, args.baseline))
        if base is None:
            continue
        for p in args.pads:
            arm = data.get((head, p))
            if arm is None or p == args.baseline:
                continue
            row = {"constant_border": p in CONSTANT_BORDER}
            for axis in ("x", "y"):
                a, b = curve(arm, axis, "AP"), curve(base, axis, "AP")
                if a is None or b is None:
                    continue
                # relative: the padding shifts the AP level, so absolute points would
                # conflate "wobbles less" with "scores lower"
                row[f"{axis}/dispersion_vs_baseline"] = paired_dispersion_test(
                    a, b, n_perm=args.permutations, relative=True)
                sa = arm["axes"][axis]["spectra"]["AP"]["stride_power_total"]
                sb = base["axes"][axis]["spectra"]["AP"]["stride_power_total"]
                row[f"{axis}/stride_power"] = {"this": sa, "baseline": sb, "delta": sa - sb}
                row[f"{axis}/AP_level"] = {"this": float(a.mean()), "baseline": float(b.mean()),
                                           "delta": float(a.mean() - b.mean())}
            report["vs_baseline"][f"{head}/pad{p}"] = row
            x = row.get("x/dispersion_vs_baseline")
            sp = row.get("x/stride_power")
            lv = row.get("x/AP_level")
            if x and sp and lv:
                LOGGER.info(
                    f"  {head:<5s} {PAD_NAMES.get(p, str(p)):<20s} AP level {lv['delta']:+6.3f} | "
                    f"stride power {sp['baseline']:.2f} -> {sp['this']:.2f} ({sp['delta']:+.2f}) | "
                    f"relative dispersion Δ{x['difference']:+.4f} p={x['p_value']:.4f} "
                    f"{'(differs)' if x['p_value'] < 0.05 else '(n.s.)'}"
                )

    # The headline: does the constant-border group behave differently from the
    # content-border group?
    LOGGER.info("")
    grp = {True: [], False: []}
    for k, v in report["arms"].items():
        if "x/AP" in v["curves"]:
            grp[v["constant_border"]].append(v["curves"]["x/AP"]["stride_power"])
    if grp[True] and grp[False]:
        mt, mf = float(np.mean(grp[True])), float(np.mean(grp[False]))
        report["group_summary"] = {
            "constant_border_mean_stride_power": mt,
            "content_border_mean_stride_power": mf,
            "delta": mt - mf,
            "n_constant": len(grp[True]), "n_content": len(grp[False]),
        }
        LOGGER.info(
            f"stride power, constant border ({len(grp[True])} arms): {mt:.3f} | "
            f"content-derived border ({len(grp[False])} arms): {mf:.3f} | Δ{mt - mf:+.3f}"
        )
        verdict = ("border cue contributes" if abs(mt - mf) > 0.10
                   else "grid-locking survives without a constant border -> aliasing, not "
                        "border-reading")
        report["verdict"] = verdict
        LOGGER.info(f"VERDICT: {verdict}")

    n_lock = sum(v["n_locked"] for v in report["gridlock"].values())
    n_cur = sum(v["n_curves"] for v in report["gridlock"].values())
    report["n_curves_grid_locked"] = n_lock
    report["n_curves"] = n_cur
    LOGGER.info(f"across all padding arms: {n_lock}/{n_cur} curves grid-locked")

    f = figure(data, list(args.pads), Path(args.fig))
    if f:
        report["figure"] = f
        LOGGER.info(f"wrote {f}")
    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
