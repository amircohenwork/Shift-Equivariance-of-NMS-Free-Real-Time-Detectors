"""Every headline number this project quotes, checked against the file it came from.

A table is transcribed by hand exactly once and then read hundreds of times, so a wrong digit
survives every proofread that is not looking for it -- one did, in the shift-TTA table, where
the row labelled "best gain" quoted the second-best merge. These tests are the transcription
step run backwards.

Values are stated here as they appear in RESULTS.md and in the course report, to the
precision those quote them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ARMS = [("yolo26n", "e2e"), ("yolo26n", "nms"), ("yolov10n", "e2e"), ("yolov10n", "nms")]

pytestmark = pytest.mark.skipif(not (RES / "oracle.csv").exists(),
                                reason="results/ not present in this checkout")


def _sweep(model: str, head: str) -> dict:
    return json.loads((RES / "sweep" / f"{model}_{head}_m31s2_n1000_pad114_summary.json").read_text())


def _oracle_row(model: str, head: str, area: str) -> dict:
    with open(RES / "oracle.csv") as fh:
        for r in csv.DictReader(fh):
            if (r["model"], r["head"], r["M"], r["N"], r["area"]) == (model, head, "1", "5000", area) \
                    and "K2" not in r["file"]:
                return r
    raise AssertionError(f"no oracle row for {model}/{head}/{area}")


# ------------------------------------------------------------------ Table 1: core ablation
@pytest.mark.parametrize(
    "model, head, base_ap, dap, dap50, flicker, dropout, drop_small, cflip, jitter, si, si_conf",
    [
        ("yolo26n", "e2e", 40.25, 3.76, 6.96, 39.2, 46.5, 80.2, 7.65, 1.75, 0.576, 0.605),
        ("yolo26n", "nms", 40.90, 3.25, 6.08, 30.2, 28.8, 57.7, 8.15, 1.85, 0.616, 0.649),
        ("yolov10n", "e2e", 38.45, 3.67, 6.57, 36.8, 45.4, 80.9, 7.23, 1.69, 0.582, 0.610),
        ("yolov10n", "nms", 39.52, 2.89, 5.49, 28.4, 27.6, 53.2, 7.69, 1.76, 0.631, 0.665),
    ],
)
def test_table_1(model, head, base_ap, dap, dap50, flicker, dropout, drop_small,
                 cflip, jitter, si, si_conf):
    o, s = _oracle_row(model, head, "all"), _sweep(model, head)
    st, sx = s["stability"], s["stability_index"]
    assert float(o["baseline_AP"]) == pytest.approx(base_ap, abs=0.005)
    assert float(o["dAP"]) == pytest.approx(dap, abs=0.005)
    assert float(o["dAP50"]) == pytest.approx(dap50, abs=0.005)
    assert st["flicker_rate"] * 100 == pytest.approx(flicker, abs=0.05)
    assert st["dropout_rate"] * 100 == pytest.approx(dropout, abs=0.05)
    assert st["small_dropout_rate"] * 100 == pytest.approx(drop_small, abs=0.05)
    assert st["class_flip_rate"] * 100 == pytest.approx(cflip, abs=0.005)
    assert st["jitter_px"] == pytest.approx(jitter, abs=0.005)
    assert sx["SI"] == pytest.approx(si, abs=0.0005)
    assert sx["SI_conf"] == pytest.approx(si_conf, abs=0.0005)


# --------------------------------------------------------------------- Table 2: shift-TTA
@pytest.mark.parametrize(
    "model, head, base, nms, vote, topk, gain, recovered, suppression",
    [
        ("yolo26n", "e2e", 40.25, 41.11, 41.08, 14.09, 0.86, 51, 27.0),
        ("yolo26n", "nms", 40.90, 41.23, 41.22, 14.05, 0.33, 21, 27.2),
        ("yolov10n", "e2e", 38.45, 39.38, 39.36, 13.32, 0.93, 52, 26.1),
        ("yolov10n", "nms", 39.52, 39.81, 39.82, 13.34, 0.30, 20, 26.5),
    ],
)
def test_table_2(model, head, base, nms, vote, topk, gain, recovered, suppression):
    t = json.loads((RES / "tta" / f"{model}_{head}_m1s1_full_pad114.json").read_text())
    r = t["results"]
    assert t["baseline_delta0"]["AP"] * 100 == pytest.approx(base, abs=0.005)
    assert r["nms"]["AP"] * 100 == pytest.approx(nms, abs=0.005)
    assert r["vote"]["AP"] * 100 == pytest.approx(vote, abs=0.005)
    assert r["topk"]["AP"] * 100 == pytest.approx(topk, abs=0.005)
    # "best" must mean best: the nms merge is not always the stronger of the two.
    best = max(r["nms"]["dAP_vs_baseline"], r["vote"]["dAP_vs_baseline"]) * 100
    best_env = max(r["nms"]["envelope_recovered"], r["vote"]["envelope_recovered"]) * 100
    assert best == pytest.approx(gain, abs=0.005)
    assert best_env == pytest.approx(recovered, abs=0.5)
    assert t["nms_minus_topk_AP"] * 100 == pytest.approx(suppression, abs=0.05)


# ------------------------------------------------------- Table A1: reproduction gates
@pytest.mark.parametrize(
    "gate_file, head, ours, published",
    [
        ("gateA_reproduce.json", "e2e", 40.044, 40.1),
        ("gateA_reproduce.json", "nms", 40.813, 40.9),
        ("gateA_yolov10n.json", "e2e", 38.477, 38.5),
        ("gateA_yolov10n.json", "nms", 39.464, None),
        ("gateA_yolov8n.json", "nms", 37.391, 37.3),
        ("gateA_yolo11n.json", "nms", 39.421, 39.5),
        ("gateA_rtdetr-l.json", "e2e", 52.775, 53.0),
    ],
)
def test_table_a1(gate_file, head, ours, published):
    d = json.loads((RES / "phase0" / gate_file).read_text())["heads"][head]
    assert d["mAP50-95"] == pytest.approx(ours, abs=0.0005)
    assert d.get("published") == published


# ------------------------------------------------- Table A2: small-object oracle
@pytest.mark.parametrize(
    "model, head, dap_s, base_s, ratio_pct",
    [
        ("yolo26n", "e2e", 6.08, 19.89, 30.6),
        ("yolo26n", "nms", 5.30, 20.93, 25.3),
        ("yolov10n", "e2e", 5.32, 19.11, 27.9),
        ("yolov10n", "nms", 4.74, 20.01, 23.7),
    ],
)
def test_table_a2(model, head, dap_s, base_s, ratio_pct):
    o = _oracle_row(model, head, "small")
    d, b = float(o["dAP"]), float(o["baseline_AP"])
    assert d == pytest.approx(dap_s, abs=0.005)
    assert b == pytest.approx(base_s, abs=0.005)
    assert d / b * 100 == pytest.approx(ratio_pct, abs=0.05)


# --------------------------------------------------------- claims made in running text
def test_grid_locking_counts_in_the_text():
    gl = json.loads((RES / "head_gap_full.json").read_text())["grid_locked"]
    power = [v["stride_power"] * 100 for v in gl.values()]
    assert round(min(power)) == 41 and round(max(power)) == 89, "text says 41-89%"
    assert sum(v["grid_locked"] for v in gl.values()) == 20, "text says 20 of 20"
    even = sum(1 for v in gl.values() if v["parity_p"] < 0.05 and v["parity_even_minus_odd"] > 0)
    odd = sum(1 for v in gl.values() if v["parity_p"] < 0.05 and v["parity_even_minus_odd"] < 0)
    assert (even, odd) == (8, 0), "text says even beats odd on 8 curves and none the other way"

    cross = json.loads((RES / "crosscheck.json").read_text())["gridlock"]
    assert sum(v["n_curves"] for v in cross.values()) == 70
    assert sum(v["n_locked"] for v in cross.values()) == 69, "text says 69 of 70"


def test_padding_ablation_numbers_in_a5():
    p = json.loads((RES / "pad_ablation.json").read_text())
    g = p["group_summary"]
    assert g["constant_border_mean_stride_power"] == pytest.approx(0.691, abs=0.0005)
    assert g["content_border_mean_stride_power"] == pytest.approx(0.574, abs=0.0005)
    assert (p["n_curves_grid_locked"], p["n_curves"]) == (74, 80)


def test_ladder_shape_claims_in_a4():
    lad = json.loads((RES / "final_report.json").read_text())["ladder"]["heads"]
    per_m = lad["e2e"]["dAP50_per_M"]
    assert per_m[0] == pytest.approx(6.0, abs=0.05) and per_m[-1] == pytest.approx(1.7, abs=0.05)
    assert per_m == sorted(per_m, reverse=True), "sub-linear means dAP50/M is non-increasing"
    assert lad["e2e"]["mean_envelope_asymmetry"] == pytest.approx(0.08, abs=0.005)
    assert lad["nms"]["mean_envelope_asymmetry"] == pytest.approx(0.12, abs=0.005)


def test_envelope_is_about_three_times_the_uniform_sweep():
    """The two metric families are quoted as 3.76 against 1.23."""
    assert _sweep("yolo26n", "e2e")["curve"]["AP_range"] * 100 == pytest.approx(1.23, abs=0.005)
    assert float(_oracle_row("yolo26n", "e2e", "all")["dAP"]) == pytest.approx(3.76, abs=0.005)


def test_k2_control_quoted_in_a7():
    rows = {}
    with open(RES / "oracle.csv") as fh:
        for r in csv.DictReader(fh):
            if "K2" in r["file"]:
                rows[r["head"]] = float(r["dAP"])
    assert rows["e2e"] == pytest.approx(3.832, abs=0.0005)
    assert rows["nms"] == pytest.approx(3.337, abs=0.0005)
    assert (rows["e2e"] / rows["nms"] - 1) * 100 == pytest.approx(14.8, abs=0.05)


def test_validator_offset_quoted_in_a7():
    d = json.loads((RES / "phase0" / "fusion_delta.json").read_text())["heads"]
    for head in ("e2e", "nms"):
        assert abs(d[head]["unfused_minus_ultralytics"]) == pytest.approx(0.03, abs=0.006)
