#!/usr/bin/env python3
"""Collect every results/*.json into the tables and figures the paper needs.

Deliberately keeps the two metric families in separate tables. The oracle envelope and
the uniform-shift range are both "how much does AP move under a 1-px shift", and putting
them in one table invites exactly the conflation this project is trying to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.sweep import load_rows, to_grid  # noqa: E402
from shifteq.utils import LOGGER, read_json, setup_logging  # noqa: E402


def md_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    if not rows:
        return "_(no data)_\n"
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            f = fmt.get(c)
            cells.append(f.format(v) if (f and isinstance(v, (int, float))) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def collect_oracle(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name.endswith("_ckpt.json") or p.name.endswith("_dryrun.json"):
            continue
        r = read_json(p)
        if "delta" not in r:
            continue
        out.append({
            "model": r["model"], "head": r["head"], "M": r["max_shift"], "N": r["n_images"],
            "objective": r["objective"], "area": r["area"],
            "AP_worst": r["delta"]["AP_worst"] * 100, "AP_best": r["delta"]["AP_best"] * 100,
            "dAP": r["delta"]["dAP"] * 100,
            "AP50_worst": r["delta"]["AP50_worst"] * 100, "AP50_best": r["delta"]["AP50_best"] * 100,
            "dAP50": r["delta"]["dAP50"] * 100,
            "baseline_AP": r["baseline"]["AP"] * 100,
            "comparable": r.get("comparable_to_manfredi_wang", False),
            "file": p.name,
        })
    return out


def collect_sweep(d: Path) -> tuple[list[dict], list[dict]]:
    curves, stab = [], []
    for p in sorted(d.glob("*_summary.json")):
        r = read_json(p)
        c = r["curve"]
        curves.append({
            "model": r["model"], "head": r["head"], "N": r["n_images"], "step": r["step"],
            "shifts": r["n_shifts"], "pad": r["pad_value"],
            "AP@0": c["AP_at_0"] * 100, "AP_min": c["AP_min"] * 100, "AP_max": c["AP_max"] * 100,
            "AP_range": c["AP_range"] * 100, "AP_std": c["AP_std"] * 100,
            "APs_range": c.get("AP_small_range", float("nan")) * 100,
            "APm_range": c.get("AP_medium_range", float("nan")) * 100,
            "APl_range": c.get("AP_large_range", float("nan")) * 100,
            "argmin": c["AP_argmin"], "argmax": c["AP_argmax"],
        })
        s = r.get("stability")
        if s:
            stab.append({
                # run_id/N/step are what make a row identifiable. Without them the n=100
                # dense control and the n=1000 sweep appear as two `yolo26n,e2e,0.25` rows
                # that a reader can only tell apart by guessing from `n_gt`.
                "run_id": r["run_id"], "model": r["model"], "head": r["head"],
                "N": r["n_images"], "step": r["step"], "shifts": r["n_shifts"],
                "tau": s["tau"], "n_gt": s["n_gt"],
                "flicker": s["flicker_rate"] * 100, "dropout": s["dropout_rate"] * 100,
                "class_flip": s["class_flip_rate"] * 100, "jitter_px": s["jitter_px"],
                "jitter_rel": s["jitter_rel"],
                "flicker_small": s.get("small_flicker_rate", float("nan")) * 100,
                "flicker_med": s.get("medium_flicker_rate", float("nan")) * 100,
                "flicker_large": s.get("large_flicker_rate", float("nan")) * 100,
                "jitter_small": s.get("small_jitter_px", float("nan")),
            })
    return curves, stab


def figures(sweep_dir: Path, out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"matplotlib unavailable, skipping figures: {e}")
        return []

    made = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(sweep_dir.glob("*.jsonl")):
        rows = load_rows(p)
        if len(rows) < 4:
            continue
        g, dys, dxs = to_grid(rows, "AP")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        im = axes[0].imshow(g * 100, origin="lower", aspect="equal",
                            extent=[min(dxs), max(dxs), min(dys), max(dys)])
        axes[0].set_xlabel("shift x (px)")
        axes[0].set_ylabel("shift y (px)")
        axes[0].set_title(f"AP(delta) - {p.stem}")
        fig.colorbar(im, ax=axes[0], label="AP (%)")

        row0 = [r for r in rows if r["dy"] == min(dys)]
        col0 = [r for r in rows if r["dx"] == min(dxs)]
        axes[1].plot([r["dx"] for r in row0], [r["AP"] * 100 for r in row0], "o-", label="dy=0")
        axes[1].plot([r["dy"] for r in col0], [r["AP"] * 100 for r in col0], "s-", label="dx=0")
        ap0 = next((r["AP"] * 100 for r in rows if r["dy"] == 0 and r["dx"] == 0), None)
        if ap0 is not None:
            axes[1].axhline(ap0, ls="--", c="k", lw=0.8, label="delta=(0,0)")
        axes[1].set_xlabel("shift (px)")
        axes[1].set_ylabel("AP (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_title("uniform-shift profile")
        fig.tight_layout()
        f = out_dir / f"sweep_{p.stem}.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        made.append(str(f))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/REPORT.md")
    args = ap.parse_args()
    setup_logging()

    res = Path(args.results)
    oracle = collect_oracle(res / "oracle")
    curves, stab = collect_sweep(res / "sweep")
    figs = figures(res / "sweep", res / "figures")

    f2 = "{:.2f}"
    f3 = "{:.3f}"
    parts = ["# Shift equivariance of NMS-free real-time detectors\n"]

    parts.append("\n## Phase-0 gates\n")
    grows = []
    for p in sorted((res / "phase0").glob("*.json")):
        r = read_json(p)
        grows.append({"gate": p.stem, "pass": r.get("gate_pass"),
                      "detail": json.dumps(r.get("gates", r.get("failures", "")))[:110]})
    parts.append(md_table(grows, ["gate", "pass", "detail"]))

    parts.append(
        "\n## Table 1 - Oracle envelope (Manfredi & Wang dAP)\n\n"
        "Per-image adversarial shift selection, greedy Algorithm 1. Comparable to prior work "
        "only at M=1. **This is not the range of AP over uniform shifts.**\n\n"
        "Reference values from Manfredi & Wang (COCO val, M=1): RetinaNet dAP 2.2 / dAP50 5.1, "
        "FasterRCNN 2.9 / 6.4, CenterNet 3.4 / 8.1.\n\n"
    )
    parts.append(md_table(
        oracle,
        ["model", "head", "M", "N", "area", "baseline_AP", "AP_worst", "AP_best", "dAP", "dAP50", "comparable"],
        {"baseline_AP": f2, "AP_worst": f2, "AP_best": f2, "dAP": f2, "dAP50": f2},
    ))

    parts.append(
        "\n## Table 2 - Uniform-shift curve AP(delta)\n\n"
        "Every image shifted by the same delta; each point is an ordinary COCO AP. "
        "`AP_range` is the honest cost of a translation in deployment.\n\n"
    )
    parts.append(md_table(
        curves,
        ["model", "head", "N", "step", "shifts", "pad", "AP@0", "AP_min", "AP_max", "AP_range",
         "AP_std", "APs_range", "APm_range", "APl_range", "argmin", "argmax"],
        {k: f2 for k in ("AP@0", "AP_min", "AP_max", "AP_range", "AP_std",
                         "APs_range", "APm_range", "APl_range")},
    ))

    parts.append("\n## Table 3 - Instance-level instability\n\n")
    parts.append(md_table(
        stab,
        ["model", "head", "tau", "n_gt", "flicker", "dropout", "class_flip", "jitter_px", "jitter_rel",
         "flicker_small", "flicker_med", "flicker_large", "jitter_small"],
        {k: f2 for k in ("flicker", "dropout", "class_flip", "flicker_small",
                         "flicker_med", "flicker_large")} | {"jitter_px": f3, "jitter_rel": f3,
                                                              "jitter_small": f3},
    ))

    if figs:
        parts.append("\n## Figures\n\n")
        for f in figs:
            parts.append(f"![{Path(f).stem}]({Path(f).relative_to(res)})\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(parts))
    for name, rows in (("oracle", oracle), ("sweep_curves", curves), ("stability", stab)):
        if rows:
            import csv

            p = res / f"{name}.csv"
            with open(p, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    LOGGER.info(f"wrote {args.out} ({len(oracle)} oracle rows, {len(curves)} sweep rows, {len(figs)} figures)")


if __name__ == "__main__":
    main()
