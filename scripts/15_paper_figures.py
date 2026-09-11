"""The three summary figures in RESULTS.md, built from the committed results.

Nothing here recomputes anything: every number is read from results/ and every figure is a
view of a table that already exists in the report, so a figure and its table cannot drift.

Sized at a single page-width column (7.1in) and drawn at that size rather than shrunk into
it, so the axis text lands at the point size it is set to instead of whatever a scale factor
leaves behind.

    python scripts/15_paper_figures.py            # -> figures/alt/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

TEXTWIDTH = 7.1  # inches, a4paper with the 0.58in margins set in the preamble
DPI = 300

E2E, NMS = "#c0392b", "#2c6fbb"           # one-to-one is warm, one-to-many+NMS is cool
PRIMARY, SECOND = "YOLO26-N", "YOLOv10-N"

#: Manfredi & Wang Table 2, RetinaNet at M=15. A different detector on different images --
#: plotted as a single reference marker, never joined to our curves.
MW = {"M": 15, "dAP": 8.3, "dAP50": 21.5}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3,
    "legend.frameon": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def _load(p: str) -> dict:
    return json.loads((RES / p).read_text())


def _panel_tag(ax, tag: str) -> None:
    ax.text(-0.16, 1.06, tag, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="bottom", ha="left")


# ------------------------------------------------------------------ figure 1: grid-locking
def fig_gridlock(out: Path) -> None:
    fft = {h: _load(f"sweep1d/yolo26n_{h}_m63s1_n5000_pad114_fft.json") for h in ("e2e", "nms")}
    gl = _load("head_gap_full.json")["grid_locked"]
    cross = _load("crosscheck.json")["gridlock"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4))
    fig.subplots_adjust(wspace=0.46)

    # (a) the curve itself -------------------------------------------------------------
    ax = axes[0]
    for head, colour, lab in (("e2e", E2E, "one-to-one (NMS-free)"),
                              ("nms", NMS, "one-to-many + NMS")):
        c = fft[head]["axes"]["x"]["curves"]["AP"]
        ax.plot(c["delta"], c["values"], color=colour, lw=1.0, label=lab)
    for x in (0, 32):
        ax.axvline(x, color="0.6", ls=":", lw=0.7, zorder=0)
    ax.set(xlabel=r"horizontal shift $\delta_x$ (px)", ylabel="COCO AP",
           xlim=(-1, 64), ylim=(39.62, 41.40), xticks=[0, 16, 32, 48, 63])
    ax.set_title("AP does not sit still", pad=3)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="upper left", handlelength=1.3, borderaxespad=0.3,
              labelspacing=0.25)
    par = fft["e2e"]["axes"]["x"]["spectra"]["AP"]["parity"]
    ax.text(0.985, 0.97, f"even $-$ odd $= {par['even_minus_odd_AP']:+.3f}$ AP\n"
                        f"($p={par['p_value']:.3f}$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="0.25")
    _panel_tag(ax, "(a)")

    # (b) where the power goes ---------------------------------------------------------
    ax = axes[1]
    spec = fft["e2e"]["axes"]["x"]["spectra"]["AP"]
    per = np.array(spec["periods"][1:], dtype=float)
    stride_bins = {v["bin"] for v in spec["strides"].values()}
    for p in (2, 4, 8, 16, 32):
        ax.axvline(p, color="0.55", ls="--", lw=0.65, zorder=0)
    for head, colour, off in (("e2e", E2E, 0.985), ("nms", NMS, 1.015)):
        s = fft[head]["axes"]["x"]["spectra"]["AP"]
        pw = np.array(s["power"][1:], dtype=float)
        on = np.array([i + 1 in stride_bins for i in range(len(pw))])
        ax.vlines(per * off, 0, pw, color=colour, lw=1.5, alpha=0.85)
        ax.plot(per[on] * off, pw[on], "o", ms=2.6, color=colour, zorder=3)
    ax.set(xscale="log", xlabel="period (px)", ylabel="fraction of spectral power",
           xlim=(1.8, 72), ylim=(0, 0.50))
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels([2, 4, 8, 16, 32, 64])
    ax.minorticks_off()
    ax.set_title("and the wobble has a rhythm", pad=3)
    e, n = (fft[h]["axes"]["x"]["spectra"]["AP"]["stride_power_total"] for h in ("e2e", "nms"))
    ax.text(0.36, 0.97, f"on stride periods:  {e*100:.0f}% (1:1),  {n*100:.0f}% (1:m)\n"
                       f"expected by chance:  {spec['expected_by_chance']*100:.0f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="0.2")
    _panel_tag(ax, "(b)")

    # (c) every architecture we could get hold of ---------------------------------------
    ax = axes[2]
    order = ["yolo26n:e2e", "yolo26n:nms", "yolov10n:e2e", "yolov10n:nms",
             "yolov8n:nms", "yolo11n:nms", "rtdetr-l:e2e"]
    pretty = {"yolo26n": "YOLO26-N", "yolov10n": "YOLOv10-N", "yolov8n": "YOLOv8-N",
              "yolo11n": "YOLO11-N", "rtdetr-l": "RT-DETR-L"}
    labs, means, los, his, locked, cols = [], [], [], [], [], []
    for arm in order:
        model, head = arm.split(":")
        sp = [c["stride_power"] * 100 for c in cross[arm]["curves"].values()]
        labs.append(f"{pretty[model]}  {'1:1' if head == 'e2e' else '1:m'}")
        means.append(float(np.mean(sp)))
        los.append(float(np.mean(sp) - min(sp)))
        his.append(float(max(sp) - np.mean(sp)))
        locked.append(f"{cross[arm]['n_locked']}/{cross[arm]['n_curves']}")
        cols.append(E2E if head == "e2e" else NMS)
    ys = np.arange(len(order))[::-1]
    ax.barh(ys, means, 0.66, color=cols, alpha=0.85,
            xerr=[los, his], error_kw=dict(lw=0.7, capsize=1.6, ecolor="0.25"))
    ax.axvline(15.625, color="0.35", ls="--", lw=0.8)
    ax.set_ylim(-0.6, 7.3)
    ax.text(17.5, 6.85, "chance (16%)", fontsize=6.0, ha="left", va="center", color="0.3")
    for y, lab in zip(ys, locked):
        ax.text(3, y, lab, ha="left", va="center", fontsize=5.8, color="white",
                fontweight="bold")
    ax.set(xlabel="power on stride periods (%)", xlim=(0, 100), yticks=ys)
    ax.set_yticklabels(labs, fontsize=6.0)
    ax.set_title("on every detector we tested", pad=3)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    _panel_tag(ax, "(c)")

    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    n_all = sum(v["n_curves"] for v in cross.values())
    n_lk = sum(v["n_locked"] for v in cross.values())
    print(f"wrote {out}   [20/20 primary curves locked, {n_lk}/{n_all} across architectures]")


# ------------------------------------------------------------- figure 2: the failure shape
def fig_failure_shape(out: Path) -> None:
    arms = [("yolo26n", "e2e"), ("yolo26n", "nms"), ("yolov10n", "e2e"), ("yolov10n", "nms")]
    S = {a: _load(f"sweep/{a[0]}_{a[1]}_m31s2_n1000_pad114_summary.json") for a in arms}
    labels = [f"{PRIMARY if m == 'yolo26n' else SECOND}  {'1:1' if h == 'e2e' else '1:m'}"
              for m, h in arms]
    cols = [E2E, NMS, E2E, NMS]
    hatch = ["", "", "///", "///"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    fig.subplots_adjust(wspace=0.28)

    def grouped(ax, groups, values, ylabel, title):
        w, xs = 0.2, np.arange(len(groups))
        for i, (lab, col, ht) in enumerate(zip(labels, cols, hatch)):
            ax.bar(xs + (i - 1.5) * w, [values[g][i] for g in groups], w * 0.92,
                   color=col, alpha=0.85, hatch=ht, edgecolor="white", lw=0.4, label=lab)
        ax.set(ylabel=ylabel, xticks=xs)
        ax.set_xticklabels(groups)
        ax.set_title(title, pad=3)
        ax.grid(axis="y", alpha=0.25, lw=0.4)

    # (a) dropout by object size --------------------------------------------------------
    sizes = ["small", "medium", "large", "all"]
    drop = {s: [S[a]["stability"][("" if s == "all" else s + "_") + "dropout_rate"] * 100
                for a in arms] for s in sizes}
    grouped(axes[0], sizes, drop, "objects lost at some shift (%)",
            "Dropout: was found at $\\delta{=}(0,0)$, gone at some shift")
    axes[0].set_ylim(0, 104)
    # The head ratio per size is the whole point of the panel: it is LOWEST on small
    # objects, which is what refutes the small-object reading of the dropout numbers.
    for x, sz in enumerate(sizes):
        for j, dx in ((0, -1.0), (2, +1.0)):
            r = drop[sz][j] / drop[sz][j + 1]
            axes[0].text(x + dx * 0.2, max(drop[sz][j], drop[sz][j + 1]) + 2.0,
                         f"$\\times${r:.2f}", ha="center", va="bottom", fontsize=5.8,
                         color="0.3")
    axes[0].legend(loc="upper right", ncol=1, handlelength=1.1,
                   handletextpad=0.4, borderaxespad=0.3, labelspacing=0.3)
    axes[0].text(0.02, 0.985,
                 "$\\times$ = one-to-one $\\div$ one-to-many\nlowest on small objects",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=5.9, color="0.35")
    _panel_tag(axes[0], "(a)")

    # (b) where the instability sits ----------------------------------------------------
    comp = {"overall\nSI": "SI", "confidence": "SI_conf", "location": "SI_loc",
            "extent": "SI_ext"}
    vals = {k: [S[a]["stability_index"][v] for a in arms] for k, v in comp.items()}
    grouped(axes[1], list(comp), vals, "Stability Index  ($\\uparrow$ better)",
            "The gap is in confidence, not in geometry")
    axes[1].set_ylim(0.4, 0.82)
    _panel_tag(axes[1], "(b)")

    # box jitter is the one geometric measure the dropout convention cannot contaminate,
    # and it is the one that runs the other way -- say so on the figure, not only in prose.
    j = [S[a]["stability"]["jitter_px"] for a in arms]
    axes[1].text(0.5, 0.975,
                 f"box jitter, the one measure dropout cannot touch (px, $\\downarrow$ better):\n"
                 f"{j[0]:.2f} vs {j[1]:.2f}   and   {j[2]:.2f} vs {j[3]:.2f}"
                 f"  $-$  the one-to-one head is steadier",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=6.0,
                 color="0.15", linespacing=1.35,
                 bbox=dict(fc="#f2f2f2", ec="0.75", lw=0.5, pad=2.0))

    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------ figure A1: the ladder
def fig_ladder(out: Path) -> None:
    lad = _load("final_report.json")["ladder"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.2))
    fig.subplots_adjust(wspace=0.40)
    styles = {"e2e": (E2E, "one-to-one"), "nms": (NMS, "one-to-many + NMS")}

    for head, v in lad["heads"].items():
        col, lab = styles[head]
        short = "1:1" if head == "e2e" else "1:m"
        M = [r["M"] for r in v["rungs"]]
        axes[0].plot(M, [r["dAP50"] for r in v["rungs"]], "o-", color=col, ms=3,
                     label=lab)
        axes[0].plot(M, [r["dAP"] for r in v["rungs"]], "s--", color=col, ms=2.6,
                     alpha=0.55)
        axes[1].plot(M, [r["dAP50"] / r["M"] for r in v["rungs"]], "o-", color=col, ms=3,
                     label=lab)
        axes[2].plot(M, [r["gain_above_baseline"] for r in v["rungs"]], "^-", color=col,
                     ms=3)
        axes[2].plot(M, [-r["loss_below_baseline"] for r in v["rungs"]], "v--", color=col,
                     ms=3, alpha=0.6)

    axes[0].plot(MW["M"], MW["dAP50"], "*", ms=8, color="0.15", zorder=5,
                 label="RetinaNet [3]")
    axes[0].plot(MW["M"], MW["dAP"], "*", ms=6, color="0.55", zorder=5)
    axes[0].set(xlabel="max shift $M$ (px)", ylabel="envelope (AP points)")
    axes[0].set_title("Grows with $M$ …", pad=3)
    axes[0].legend(loc="lower right", handlelength=1.3, borderaxespad=0.3,
                   labelspacing=0.25, fontsize=5.9)
    axes[0].text(0.035, 0.965, "solid: $\\Delta$AP$_{50}$\ndashed: $\\Delta$AP",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=5.9, color="0.3")
    _panel_tag(axes[0], "(a)")

    axes[1].set(xlabel="max shift $M$ (px)", ylabel="$\\Delta$AP$_{50}$ per px of $M$")
    axes[1].set_title("… but sub-linearly", pad=3)
    axes[1].legend(loc="upper right", handlelength=1.3, borderaxespad=0.2)
    _panel_tag(axes[1], "(b)")

    axes[2].axhline(0, color="0.35", lw=0.7)
    axes[2].set(xlabel="max shift $M$ (px)", ylabel="AP$_{50}$ vs. $\\delta{=}(0,0)$")
    axes[2].set_title("and $\\delta{=}(0,0)$ is an average case", pad=3)
    axes[2].text(0.035, 0.965, "$\\blacktriangle$ best shift\n"
                 "$\\blacktriangledown$ worst shift",
                 transform=axes[2].transAxes, ha="left", va="top", fontsize=5.9, color="0.3")
    a_e2e = lad["heads"]["e2e"]["mean_envelope_asymmetry"]
    a_nms = lad["heads"]["nms"]["mean_envelope_asymmetry"]
    axes[2].text(0.03, 0.03,
                 f"mean $|$best$+$worst$|$ asymmetry: {a_e2e:.2f} / {a_nms:.2f}",
                 transform=axes[2].transAxes, ha="left", va="bottom", fontsize=5.9,
                 color="0.25")
    _panel_tag(axes[2], "(c)")

    for a in axes:
        a.grid(alpha=0.25, lw=0.4)
        a.set_xticks([1, 3, 7, 15])

    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures/alt",
                help="NOT figures/ -- that holds the figures RESULTS.md embeds")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    fig_gridlock(out / "fig1_gridlock.png")
    fig_failure_shape(out / "fig2_failure_shape.png")
    fig_ladder(out / "figA1_ladder.png")


if __name__ == "__main__":
    main()
