#!/usr/bin/env python3
"""Cross-model check: is the story about YOLO26, or about detectors?

`11_head_gap.py` answers the paper's question on one checkpoint. That leaves two ways to be
wrong, and this script is aimed at both.

**Is grid-locking a YOLO26 quirk?** If the periodic response comes from the strided
downsampling cascade, every detector built on one should show it -- including a transformer
detector with a completely different head. YOLOv8-N, YOLO11-N and RT-DETR-L are one-head
models and contribute nothing to the head ablation; they are here purely as a universality
check on the spectral claim.

**Does the head gap replicate?** YOLOv10-N is the only other released checkpoint that
carries both a one-to-one and a one-to-many branch trained together, so it is a genuine
second instance of the paper's ablation -- same design, different weights, different
training recipe. A head effect that appears on yolo26n and vanishes on yolov10n is a
statement about one checkpoint, not about NMS-free detection. This is the closest thing to
an independent replication the released model zoo allows.

Absolute vs relative dispersion
-------------------------------
Arms sit at different AP levels (RT-DETR-L at ~53, YOLOv8-N at ~37), so "whose AP curve
moves by more points" partly measures where a model sits on the AP scale. Every cross-model
comparison here is therefore reported twice: in AP points, and as a coefficient of variation
(mean |deviation| / mean AP). Within-checkpoint head comparisons stay absolute -- the two
heads are within a point of each other and the AP cost is what a practitioner actually pays.

    python scripts/12_crosscheck.py --subset full \
        --arms yolo26n:e2e yolo26n:nms yolov10n:e2e yolov10n:nms yolov8n:nms yolo11n:nms rtdetr-l:e2e
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.spectrum import NETWORK_STRIDES, dispersion, paired_dispersion_test  # noqa: E402
from shifteq.utils import LOGGER, read_json, setup_logging, write_json  # noqa: E402

KEYS = ("AP", "AP50", "AP_small", "AP_medium", "AP_large")
DUAL_HEAD = ("yolo26n", "yolov10n")


def load_arm(d: Path, model: str, head: str, max_shift: int, subset: str, pad: int) -> dict | None:
    sub = "full" if str(subset).lower() in ("full", "none", "all") else f"n{subset}"
    f = d / f"{model}_{head}_m{max_shift}s1_{sub}_pad{pad}_fft.json"
    if not f.exists():
        cands = sorted(d.glob(f"{model}_{head}_m{max_shift}s1_*_pad{pad}_fft.json"))
        if not cands:
            LOGGER.warning(f"missing {f.name} -- arm {model}:{head} skipped")
            return None
        f = cands[-1]
        LOGGER.warning(f"{sub} not found for {model}:{head}; using {f.name}")
    r = read_json(f)
    r["_file"] = str(f)
    return r


def curve(arm: dict, axis: str, key: str) -> np.ndarray | None:
    ax = arm["axes"].get(axis)
    return None if ax is None else np.array(ax["curves"][key]["values"], dtype=np.float64)


def figure(arms: dict, out: Path, axis: str = "x") -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"matplotlib unavailable: {e}")
        return None

    names = [k for k in arms if axis in arms[k]["axes"]]
    if not names:
        return None
    cmap = plt.get_cmap("tab10")
    color = {n: cmap(i % 10) for i, n in enumerate(names)}

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

    # A: every curve centred on its own mean -- the shape, with the AP level removed
    for n in names:
        v = curve(arms[n], axis, "AP")
        axs[0].plot(np.arange(len(v)), v - v.mean(), lw=1.2, color=color[n], label=n)
    for s in (32, 64):
        axs[0].axvline(s, color="k", ls=":", lw=0.8)
    axs[0].set_xlabel(f"shift {axis} (px)")
    axs[0].set_ylabel("AP - mean(AP)  (points)")
    axs[0].set_title("AP(δ) with the AP level removed")
    axs[0].legend(fontsize=7, ncol=2)
    axs[0].grid(alpha=0.3)

    # B: how much of each spectrum sits in the stride bins
    sp = [arms[n]["axes"][axis]["spectra"]["AP"]["stride_power_total"] for n in names]
    bars = axs[1].bar(range(len(names)), sp,
                      color=[color[n] for n in names])
    axs[1].axhline(len(NETWORK_STRIDES) / 32, color="k", ls="--", lw=1.0,
                   label="uniform spectrum (chance)")
    for b, n in zip(bars, names):
        lock = arms[n]["axes"][axis]["spectra"]["AP"]["interpretation"] == "grid-locked"
        axs[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    "✓" if lock else "×", ha="center", fontsize=13)
    axs[1].set_xticks(range(len(names)))
    axs[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axs[1].set_ylabel("power fraction in stride bins 2/4/8/16/32")
    axs[1].set_title("grid-locking is not model-specific")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3, axis="y")

    # C: relative dispersion, so the AP level does not do the talking
    cv = [dispersion(curve(arms[n], axis, "AP"))["cv_percent"] for n in names]
    cvs = [dispersion(curve(arms[n], axis, "AP_small"))["cv_percent"] for n in names]
    w = 0.4
    axs[2].bar(np.arange(len(names)) - w / 2, cv, w, label="AP", color="#1f6feb")
    axs[2].bar(np.arange(len(names)) + w / 2, cvs, w, label="AP_small", color="#c1440e")
    axs[2].set_xticks(range(len(names)))
    axs[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axs[2].set_ylabel("mean |AP - mean| / mean AP  (%)")
    axs[2].set_title("relative shift dispersion")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["yolo26n:e2e", "yolo26n:nms", "yolov10n:e2e", "yolov10n:nms",
                             "yolov8n:nms", "yolo11n:nms", "rtdetr-l:e2e"])
    ap.add_argument("--dir", default="results/sweep1d")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--max-shift", type=int, default=63)
    ap.add_argument("--pad-value", type=int, default=114)
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--reference", default="yolo26n:nms",
                    help="arm every other arm is compared against in the cross-model table")
    ap.add_argument("--out", default="results/crosscheck.json")
    ap.add_argument("--fig", default="results/figures/crosscheck.png")
    args = ap.parse_args()

    setup_logging()
    d = Path(args.dir)
    arms: dict[str, dict] = {}
    for a in args.arms:
        model, _, head = a.partition(":")
        r = load_arm(d, model, head or "e2e", args.max_shift, args.subset, args.pad_value)
        if r is not None:
            arms[a] = r
    if not arms:
        raise SystemExit("no arms found -- run scripts/10_sweep1d_fft.py first")

    report: dict = {"arms": {}, "gridlock": {}, "parity": {}, "dispersion": {},
                    "replication": {}, "cross_model": {},
                    "subset": args.subset, "max_shift": args.max_shift,
                    "pad_value": args.pad_value}

    # ---------------------------------------------------------------- universality
    LOGGER.info("=== is grid-locking universal across detectors? ===")
    LOGGER.info(f"  {'arm':<16s} {'N':>5s} {'locked':>8s} {'stride pw':>10s} {'enrich':>7s} "
                f"{'p_fwer':>8s} {'even-odd':>9s} {'AP@δ0':>7s}")
    tot_locked = tot_curves = 0
    for name, r in arms.items():
        report["arms"][name] = {"file": r["_file"], "n_images": r["n_images"],
                                "canvas": r.get("canvas"), "stride": r.get("stride")}
        n_lock = n_cur = 0
        per_curve = {}
        for axis, ax in r["axes"].items():
            for key in KEYS:
                s = ax["spectra"][key]
                n_cur += 1
                lock = s["interpretation"] == "grid-locked"
                n_lock += lock
                per_curve[f"{axis}/{key}"] = {
                    "peak_period_px": s["peak"]["period_px"],
                    "stride_power": s["stride_power_total"],
                    "enrichment": s["enrichment"],
                    "p_stride_fwer": s["p_stride_fwer"],
                    "grid_locked": lock,
                }
                report["parity"][f"{name}/{axis}/{key}"] = {
                    "even_minus_odd_AP": s["parity"]["even_minus_odd_AP"],
                    "p_value": s["parity"]["p_value"],
                }
                report["dispersion"][f"{name}/{axis}/{key}"] = dispersion(
                    np.array(ax["curves"][key]["values"])
                )
        report["gridlock"][name] = {"n_locked": n_lock, "n_curves": n_cur, "curves": per_curve}
        tot_locked += n_lock
        tot_curves += n_cur
        ax0 = "x" if "x" in r["axes"] else next(iter(r["axes"]))
        s0 = r["axes"][ax0]["spectra"]["AP"]
        c0 = r["axes"][ax0]["curves"]["AP"]
        LOGGER.info(
            f"  {name:<16s} {r['n_images']:>5d} {n_lock:>4d}/{n_cur:<3d} "
            f"{s0['stride_power_total']:>10.2f} {s0['enrichment']:>6.1f}x "
            f"{s0['p_stride_fwer']:>8.4f} {s0['parity']['even_minus_odd_AP']:>+9.3f} "
            f"{c0['at_0']:>7.2f}"
        )
    report["n_curves_grid_locked"] = tot_locked
    report["n_curves"] = tot_curves
    LOGGER.info(f"  TOTAL {tot_locked}/{tot_curves} curves grid-locked across "
                f"{len(arms)} model/head arms")

    # ---------------------------------------------------------------- replication
    LOGGER.info("")
    LOGGER.info("=== does the head gap replicate on a second dual-head checkpoint? ===")
    for model in DUAL_HEAD:
        e, n = f"{model}:e2e", f"{model}:nms"
        if e not in arms or n not in arms:
            continue
        res, wins, losses = {}, 0, 0
        for axis in ("x", "y"):
            for key in KEYS:
                a, b = curve(arms[e], axis, key), curve(arms[n], axis, key)
                if a is None or b is None:
                    continue
                t = paired_dispersion_test(a, b, n_perm=args.permutations)
                sig = t["p_value"] < 0.05
                wins += sig and t["difference"] > 0
                losses += sig and t["difference"] < 0
                res[f"{axis}/{key}"] = {**t, "significant": bool(sig),
                                        "direction": "e2e worse" if t["difference"] > 0 else "nms worse"}
                LOGGER.info(
                    "  {:<9s} {:>1s} {:<10s} mean|dev| {:.4f} vs {:.4f} (Δ{:+.4f}, p={:.4f}) {}".format(
                        model, axis, key, t["mean_abs_dev_a"], t["mean_abs_dev_b"],
                        t["difference"], t["p_value"],
                        ("**e2e worse**" if t["difference"] > 0 else "**nms worse**") if sig else "n.s.")
                )
        report["replication"][model] = {"curves": res, "n_e2e_worse": wins, "n_nms_worse": losses,
                                        "n_curves": len(res)}
        LOGGER.info(f"  -> {model}: e2e less stable on {wins}/{len(res)}, nms on {losses}")

    present = [m for m in DUAL_HEAD if m in report["replication"]]
    if len(present) == 2:
        agree = sum(
            1 for k in report["replication"][present[0]]["curves"]
            if k in report["replication"][present[1]]["curves"]
            and np.sign(report["replication"][present[0]]["curves"][k]["difference"])
            == np.sign(report["replication"][present[1]]["curves"][k]["difference"])
        )
        n = len(report["replication"][present[0]]["curves"])
        # Sign agreement across two independent checkpoints under "no head effect" is a
        # coin flip per curve; the curves are not independent of each other, so this is a
        # descriptive concordance, not a p-value.
        report["sign_agreement"] = {"n_agree": int(agree), "n_curves": int(n),
                                    "models": present}
        LOGGER.info(f"  sign agreement between {present[0]} and {present[1]}: "
                    f"{agree}/{n} curves point the same way")

    # ---------------------------------------------------------------- cross-model
    LOGGER.info("")
    LOGGER.info(f"=== cross-model dispersion, vs {args.reference} (relative: CV) ===")
    ref = arms.get(args.reference)
    for name, r in arms.items():
        row = {}
        for axis in ("x", "y"):
            a = curve(r, axis, "AP")
            if a is None:
                continue
            row[f"{axis}/AP"] = dispersion(a)
            row[f"{axis}/AP_small"] = dispersion(curve(r, axis, "AP_small"))
            if ref is not None and name != args.reference and axis in ref["axes"]:
                row[f"{axis}/AP_vs_ref"] = paired_dispersion_test(
                    a, curve(ref, axis, "AP"), n_perm=args.permutations, relative=True
                )
        report["cross_model"][name] = row
        if "x/AP" in row:
            v = row["x/AP"]
            vs = row.get("x/AP_vs_ref")
            tail = (f" | vs ref Δcv{vs['difference']:+.4f} p={vs['p_value']:.4f}"
                    if vs else " | (reference)")
            LOGGER.info(f"  {name:<16s} AP {v['mean']:6.2f} ± {v['mean_abs_dev']:.4f} "
                        f"(cv {v['cv_percent']:.3f}%) small cv "
                        f"{row['x/AP_small']['cv_percent']:.3f}%{tail}")

    f = figure(arms, Path(args.fig))
    if f:
        report["figure"] = f
        LOGGER.info(f"wrote {f}")

    LOGGER.info("")
    LOGGER.info(f"CROSS-CHECK: {tot_locked}/{tot_curves} curves grid-locked over "
                f"{len(arms)} arms / {len({a.split(':')[0] for a in arms})} models.")
    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
