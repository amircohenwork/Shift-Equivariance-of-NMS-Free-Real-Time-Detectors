#!/usr/bin/env python3
"""The last two experiments, each read against the prediction that was written down first.

Both are pre-registered in `PROTOCOL.md` §12 (v1.2.0), declared before either was run.
This script does not decide what counts as success -- §12 already did -- it only reports
whether the measured numbers meet the stated criterion, including when they do not.

**Small-object oracle** (§12.1). §4 fixes the greedy objective's area range to `all`, and
the `AP_small` figure printed beside it is `summary_at_assignment`: AP_small read off the
assignment that maximised *overall* AP50. That describes an assignment; it does not search
over small objects. Finding 1's strongest sub-claim is about small objects, so it needs a
real oracle. Prediction: the head gap is *wider* on small objects than on all objects,
primary quantity the ΔAP ratio.

**Max-shift ladder** (§12.2). Manfredi & Wang Table 2, M ∈ {1,3,7,15} on 100 images -- the
one place this project meets the prior work on its protocol rather than its own. Three
predictions from their Sec. 4.1: monotone growth, sub-linear growth, and an envelope roughly
symmetric about the unshifted baseline. Read as a shape; the levels are not comparable to
anything else here, because ΔAP moves with both M and N.

    python scripts/14_final_report.py                  # both, whatever is on disk
    python scripts/14_final_report.py --what ladder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.protocol import LADDER, SMALL_ORACLE  # noqa: E402
from shifteq.utils import LOGGER, read_json, setup_logging, write_json  # noqa: E402

ORACLE_DIR = Path("results/oracle")

#: Manfredi & Wang Table 2, RetinaNet. Quoted for shape only -- a different detector on a
#: different image set, so the levels are not ours to match.
MW_RETINANET = {"baseline_AP": 43.8, "baseline_AP50": 64.2, "dAP_at_15": 8.3,
                "dAP50_at_15": 21.5}


def _load(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def _delta(r: dict, key: str) -> float:
    return r["delta"][key] * 100.0


# ---------------------------------------------------------------------------------------
# 12.1 small-object oracle
# ---------------------------------------------------------------------------------------
def small_oracle(tag: str = "small") -> dict:
    """Is the head gap wider when the oracle actually searches over small objects?"""
    out: dict = {"arms": {}, "checkpoints": {}, "prediction": SMALL_ORACLE["prediction"],
                 "primary_quantity": SMALL_ORACLE["primary_quantity"]}

    for arm in SMALL_ORACLE["arms"]:
        model, head = arm.split(":")
        rid = f"{model}_{head}_m1s1_full_pad114"
        small = _load(ORACLE_DIR / f"{rid}_{tag}_AP50_small.json")
        allr = _load(ORACLE_DIR / f"{rid}_AP50_all.json")
        if small is None or allr is None:
            LOGGER.warning(f"{arm}: missing "
                           f"{'small' if small is None else ''}{' and ' if small is None and allr is None else ''}"
                           f"{'all-area' if allr is None else ''} oracle -- skipped")
            continue
        out["arms"][arm] = {
            "small": {"dAP": _delta(small, "dAP"), "dAP50": _delta(small, "dAP50"),
                      "AP_best": _delta(small, "AP_best"), "AP_worst": _delta(small, "AP_worst"),
                      "baseline_AP": small["baseline"]["AP"] * 100},
            "all": {"dAP": _delta(allr, "dAP"), "dAP50": _delta(allr, "dAP50")},
        }

    for model in dict.fromkeys(a.split(":")[0] for a in out["arms"]):
        e, n = f"{model}:e2e", f"{model}:nms"
        if e not in out["arms"] or n not in out["arms"]:
            LOGGER.warning(f"{model}: need both heads to state a gap -- skipped")
            continue
        row: dict = {}
        for area in ("small", "all"):
            de, dn = out["arms"][e][area]["dAP"], out["arms"][n][area]["dAP"]
            de50, dn50 = out["arms"][e][area]["dAP50"], out["arms"][n][area]["dAP50"]
            row[area] = {
                "dAP_e2e": de, "dAP_nms": dn, "dAP_gap": de - dn,
                "dAP_ratio_pct": (de / dn - 1) * 100 if dn else float("nan"),
                "dAP50_e2e": de50, "dAP50_nms": dn50,
                "dAP50_ratio_pct": (de50 / dn50 - 1) * 100 if dn50 else float("nan"),
            }
        # The pre-registered test, in the pre-registered order: direction first, then
        # whether the gap widened. A direction failure is the serious one -- it would say
        # small objects are not where the head difference lives.
        direction_ok = row["small"]["dAP_e2e"] > row["small"]["dAP_nms"]
        widened = row["small"]["dAP_ratio_pct"] > row["all"]["dAP_ratio_pct"]
        row["direction_holds"] = bool(direction_ok)
        row["gap_widened_vs_all_area"] = bool(widened)
        row["verdict"] = (
            "confirmed" if direction_ok and widened
            else "partial -- e2e worse on small objects, but the gap did not widen"
            if direction_ok else "REFUTED -- e2e's small-object envelope is not wider"
        )
        out["checkpoints"][model] = row

    return out


# ---------------------------------------------------------------------------------------
# 12.2 max-shift ladder
# ---------------------------------------------------------------------------------------
def ladder(model: str = "yolo26n") -> dict:
    """Does our envelope grow with max shift the way Manfredi & Wang's did?"""
    out: dict = {"model": model, "reference": MW_RETINANET, "heads": {},
                 "n_images": LADDER["n_images"], "max_shifts": LADDER["max_shifts"]}

    for head in LADDER["heads"]:
        rungs = []
        for M in LADDER["max_shifts"]:
            r = _load(ORACLE_DIR / f"{model}_{head}_m{M}s1_n{LADDER['n_images']}_pad114_AP50_all.json")
            if r is None:
                LOGGER.warning(f"{model}/{head} M={M}: not produced -- skipped")
                continue
            base50 = r["baseline"]["AP50"] * 100
            best50, worst50 = _delta(r, "AP50_best"), _delta(r, "AP50_worst")
            rungs.append({
                "M": M, "n_shifts": len(r["shifts"]),
                "dAP": _delta(r, "dAP"), "dAP50": _delta(r, "dAP50"),
                "baseline_AP": r["baseline"]["AP"] * 100, "baseline_AP50": base50,
                "AP50_best": best50, "AP50_worst": worst50,
                # Symmetry: how far the envelope reaches above the unshifted number vs
                # below it. Equal arms mean delta=(0,0) is an average case, not a best case.
                "gain_above_baseline": best50 - base50,
                "loss_below_baseline": base50 - worst50,
            })
        if len(rungs) < 2:
            out["heads"][head] = {"rungs": rungs, "verdict": "not enough rungs to read a shape"}
            continue

        d50 = [r["dAP50"] for r in rungs]
        ms = [r["M"] for r in rungs]
        monotone = all(b >= a for a, b in zip(d50, d50[1:]))
        # Sub-linear: the envelope per unit of shift must fall as M grows. Comparing
        # dAP50/M across rungs is the scale-free version of "grows slower than M".
        per_m = [d / m for d, m in zip(d50, ms)]
        sublinear = all(b <= a for a, b in zip(per_m, per_m[1:]))
        # Their Sec. 4.1 caveat: dAP need not be monotone, because the search optimises AP50.
        dap = [r["dAP"] for r in rungs]
        dap_monotone = all(b >= a for a, b in zip(dap, dap[1:]))
        asym = [
            abs(r["gain_above_baseline"] - r["loss_below_baseline"])
            / max(1e-9, 0.5 * (r["gain_above_baseline"] + r["loss_below_baseline"]))
            for r in rungs
        ]
        out["heads"][head] = {
            "rungs": rungs,
            "dAP50_monotone_in_M": bool(monotone),
            "dAP50_sublinear_in_M": bool(sublinear),
            "dAP50_per_M": per_m,
            "dAP_monotone_in_M": bool(dap_monotone),
            "mean_envelope_asymmetry": sum(asym) / len(asym),
            "max_envelope_asymmetry": max(asym),
            "verdict": (
                "shape matches Manfredi & Wang" if monotone and sublinear
                else "monotone but NOT sub-linear" if monotone
                else "NOT monotone -- their central shape claim does not hold here"
            ),
        }
    return out


def figure(lad: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = {h: v for h, v in lad["heads"].items() if len(v.get("rungs", [])) >= 2}
    if not heads:
        LOGGER.warning("no ladder rungs to plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for head, v in heads.items():
        ms = [r["M"] for r in v["rungs"]]
        axes[0].plot(ms, [r["dAP50"] for r in v["rungs"]], "o-", label=f"{head}  ΔAP50")
        axes[0].plot(ms, [r["dAP"] for r in v["rungs"]], "s--", alpha=.6, label=f"{head}  ΔAP")
        axes[1].plot(ms, [r["gain_above_baseline"] for r in v["rungs"]], "^-",
                     label=f"{head}  best − baseline")
        axes[1].plot(ms, [r["loss_below_baseline"] for r in v["rungs"]], "v-",
                     label=f"{head}  baseline − worst")
    axes[0].set(xlabel="max shift M (px)", ylabel="envelope (AP points)",
                title=f"Envelope growth — {lad['model']}, N={lad['n_images']}")
    axes[1].set(xlabel="max shift M (px)", ylabel="AP50 points",
                title="Is δ=(0,0) an average case or a best case?")
    for a in axes:
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    LOGGER.info(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="both", choices=["both", "small", "ladder"])
    ap.add_argument("--model", default="yolo26n", help="ladder model")
    ap.add_argument("--tag", default="small", help="artifact tag the small oracle was run with")
    ap.add_argument("--out", default="results/final_report.json")
    ap.add_argument("--figure", default="results/figures/ladder.png")
    args = ap.parse_args()
    setup_logging()

    report: dict = {}

    if args.what in ("both", "small"):
        s = small_oracle(args.tag)
        report["small_oracle"] = s
        LOGGER.info("=== 12.1 small-object oracle ===")
        LOGGER.info(f"  prediction: {s['prediction']}")
        for model, row in s["checkpoints"].items():
            LOGGER.info(
                f"  {model:9s} small: e2e {row['small']['dAP_e2e']:.4f} vs nms "
                f"{row['small']['dAP_nms']:.4f}  ({row['small']['dAP_ratio_pct']:+.1f}%)   "
                f"| all-area was {row['all']['dAP_ratio_pct']:+.1f}%"
            )
            LOGGER.info(
                f"  {'':9s} dAP50 ratio (secondary): small {row['small']['dAP50_ratio_pct']:+.1f}% "
                f"| all-area {row['all']['dAP50_ratio_pct']:+.1f}%"
            )
            LOGGER.info(f"  {'':9s} -> {row['verdict']}")
        if not s["checkpoints"]:
            LOGGER.warning("  no complete checkpoint -- nothing to state")

    if args.what in ("both", "ladder"):
        lad = ladder(args.model)
        report["ladder"] = lad
        LOGGER.info("=== 12.2 max-shift ladder ===")
        for head, v in lad["heads"].items():
            for r in v.get("rungs", []):
                LOGGER.info(
                    f"  {lad['model']}/{head}  M={r['M']:2d}  {r['n_shifts']:3d} shifts   "
                    f"dAP={r['dAP']:6.3f}  dAP50={r['dAP50']:6.3f}   "
                    f"best−base={r['gain_above_baseline']:+6.3f}  "
                    f"base−worst={r['loss_below_baseline']:+6.3f}"
                )
            if "verdict" in v:
                LOGGER.info(f"  {lad['model']}/{head}  -> {v['verdict']}")
                if "mean_envelope_asymmetry" in v:
                    LOGGER.info(
                        f"  {lad['model']}/{head}  dAP monotone too: {v['dAP_monotone_in_M']} "
                        f"(their Sec. 4.1 says it need not be)  |  mean envelope asymmetry "
                        f"{v['mean_envelope_asymmetry']:.2f}"
                    )
        if any(len(v.get("rungs", [])) >= 2 for v in lad["heads"].values()):
            figure(lad, Path(args.figure))
            report["figure"] = args.figure

    write_json(args.out, report)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
