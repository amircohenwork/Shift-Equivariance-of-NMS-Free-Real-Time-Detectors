"""The §12 verdicts must be able to come out *against* us.

A reporting script that only ever prints "confirmed" is not evidence, and a test that only
feeds it confirming data cannot tell the difference. Every case here plants numbers whose
correct verdict is known in advance, including the ones that refute the prediction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("final_report", ROOT / "scripts" / "14_final_report.py")
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

from shifteq.utils import write_json  # noqa: E402


def _oracle(dap: float, dap50: float, base_ap: float = 40.0, base_ap50: float = 55.0,
            best50: float | None = None, worst50: float | None = None,
            n_shifts: int = 4) -> dict:
    """A minimal oracle report. Fractions on disk, percents in the reader."""
    best50 = base_ap50 + dap50 / 2 if best50 is None else best50
    worst50 = best50 - dap50 if worst50 is None else worst50
    return {
        "baseline": {"AP": base_ap / 100, "AP50": base_ap50 / 100},
        "shifts": [[0, 0]] * n_shifts,
        "delta": {"dAP": dap / 100, "dAP50": dap50 / 100,
                  "AP_best": (base_ap + dap / 2) / 100, "AP_worst": (base_ap - dap / 2) / 100,
                  "AP50_best": best50 / 100, "AP50_worst": worst50 / 100},
    }


def _plant_small(d: Path, model: str, *, small_e2e: float, small_nms: float,
                 all_e2e: float, all_nms: float) -> None:
    for head, sm, al in (("e2e", small_e2e, all_e2e), ("nms", small_nms, all_nms)):
        rid = f"{model}_{head}_m1s1_full_pad114"
        write_json(d / f"{rid}_small_AP50_small.json", _oracle(sm, sm * 1.8))
        write_json(d / f"{rid}_AP50_all.json", _oracle(al, al * 1.8))


# ---------------------------------------------------------------------------- 12.1
@pytest.mark.parametrize(
    "small_e2e, small_nms, expect",
    [
        # gap widens: small ratio +50% beats all-area ratio +15.4%  -> the prediction
        (6.0, 4.0, "confirmed"),
        # e2e still worse, but by *less* than on all objects -> partial, not a pass
        (4.2, 4.0, "partial"),
        # e2e's small-object envelope is NARROWER -> the prediction is refuted
        (3.5, 4.0, "REFUTED"),
    ],
)
def test_small_oracle_verdict_tracks_the_data(tmp_path, monkeypatch, small_e2e, small_nms, expect):
    monkeypatch.setattr(fr, "ORACLE_DIR", tmp_path)
    # all-area gap fixed at 3.7555 vs 3.2537 = +15.4%, the published yolo26n figure
    _plant_small(tmp_path, "yolo26n", small_e2e=small_e2e, small_nms=small_nms,
                 all_e2e=3.7555, all_nms=3.2537)
    row = fr.small_oracle()["checkpoints"]["yolo26n"]
    assert row["verdict"].startswith(expect), row["verdict"]


def test_small_oracle_reports_both_quantities_not_the_flattering_one(tmp_path, monkeypatch):
    """dAP50 is secondary but must still be printed, per §12.1."""
    monkeypatch.setattr(fr, "ORACLE_DIR", tmp_path)
    _plant_small(tmp_path, "yolo26n", small_e2e=6.0, small_nms=4.0, all_e2e=3.7555, all_nms=3.2537)
    row = fr.small_oracle()["checkpoints"]["yolo26n"]
    for area in ("small", "all"):
        assert "dAP_ratio_pct" in row[area] and "dAP50_ratio_pct" in row[area]
    # the primary quantity is the one the verdict is computed from
    assert row["gap_widened_vs_all_area"] == (
        row["small"]["dAP_ratio_pct"] > row["all"]["dAP_ratio_pct"]
    )


def test_small_oracle_skips_a_checkpoint_missing_one_head(tmp_path, monkeypatch):
    """One head alone cannot state a gap; it must be dropped, not half-reported."""
    monkeypatch.setattr(fr, "ORACLE_DIR", tmp_path)
    rid = "yolo26n_e2e_m1s1_full_pad114"
    write_json(tmp_path / f"{rid}_small_AP50_small.json", _oracle(6.0, 10.0))
    write_json(tmp_path / f"{rid}_AP50_all.json", _oracle(3.7555, 6.9))
    out = fr.small_oracle()
    assert "yolo26n:e2e" in out["arms"]
    assert out["checkpoints"] == {}


# ---------------------------------------------------------------------------- 12.2
def _plant_ladder(d: Path, head: str, d50_by_m: dict[int, float],
                  gain_frac: float = 0.5) -> None:
    for M, d50 in d50_by_m.items():
        base50 = 55.0
        best50 = base50 + d50 * gain_frac
        write_json(
            d / f"yolo26n_{head}_m{M}s1_n100_pad114_AP50_all.json",
            _oracle(d50 / 2, d50, base_ap50=base50, best50=best50, worst50=best50 - d50,
                    n_shifts=(M + 1) ** 2),
        )


@pytest.mark.parametrize(
    "d50_by_m, expect",
    [
        # sub-linear: dAP50/M falls 6.0 -> 3.0 -> 1.71 -> 1.07
        ({1: 6.0, 3: 9.0, 7: 12.0, 15: 16.0}, "shape matches"),
        # monotone but super-linear (dAP50/M rises) -- their central claim would not hold
        ({1: 2.0, 3: 8.0, 7: 25.0, 15: 60.0}, "monotone but NOT sub-linear"),
        # dAP50 falls at M=15 -- not monotone at all
        ({1: 6.0, 3: 9.0, 7: 12.0, 15: 8.0}, "NOT monotone"),
    ],
)
def test_ladder_verdict_tracks_the_shape(tmp_path, monkeypatch, d50_by_m, expect):
    monkeypatch.setattr(fr, "ORACLE_DIR", tmp_path)
    _plant_ladder(tmp_path, "e2e", d50_by_m)
    v = fr.ladder()["heads"]["e2e"]
    assert v["verdict"].startswith(expect), v["verdict"]


def test_ladder_symmetry_is_measured_not_assumed(tmp_path, monkeypatch):
    """A perfectly symmetric envelope reads 0; a one-sided one must not."""
    monkeypatch.setattr(fr, "ORACLE_DIR", tmp_path)
    _plant_ladder(tmp_path, "e2e", {1: 6.0, 3: 9.0, 7: 12.0, 15: 16.0}, gain_frac=0.5)
    sym = fr.ladder()["heads"]["e2e"]["mean_envelope_asymmetry"]
    assert sym == pytest.approx(0.0, abs=1e-9)

    _plant_ladder(tmp_path, "nms", {1: 6.0, 3: 9.0, 7: 12.0, 15: 16.0}, gain_frac=0.05)
    asym = fr.ladder()["heads"]["nms"]["mean_envelope_asymmetry"]
    assert asym > 1.5, "an envelope almost entirely below the baseline must read as asymmetric"


def test_ladder_needs_two_rungs_before_claiming_a_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(fr, "ORACLE_DIR", tmp_path)
    _plant_ladder(tmp_path, "e2e", {1: 6.0})
    assert "not enough rungs" in fr.ladder()["heads"]["e2e"]["verdict"]


# ------------------------------------------------------- the comparability column
def test_a_small_area_run_is_not_comparable_to_manfredi_wang(tmp_path):
    """Their Table 1 is max_shift=1 over ALL areas.

    A small-object envelope at M=1 satisfies the max_shift half of that and nothing else,
    and the flag as originally written recorded it as comparable. Quoting ΔAP_small against
    their all-area table would be a category error, so the report derives the column rather
    than trusting what an older run wrote into its JSON.
    """
    import importlib.util as iu

    spec = iu.spec_from_file_location("report", ROOT / "scripts" / "06_report.py")
    rep = iu.module_from_spec(spec)
    spec.loader.exec_module(rep)

    base = {"model": "yolo26n", "head": "e2e", "max_shift": 1, "n_images": 5000,
            "objective": "AP50", "baseline": {"AP": 0.4, "AP50": 0.55},
            "delta": {"dAP": 0.06, "dAP50": 0.12, "AP_best": 0.23, "AP_worst": 0.17,
                      "AP50_best": 0.40, "AP50_worst": 0.28}}
    # Both claim comparable_to_manfredi_wang=True, as a pre-fix run would have.
    write_json(tmp_path / "small.json", {**base, "area": "small",
                                         "comparable_to_manfredi_wang": True})
    write_json(tmp_path / "all.json", {**base, "area": "all",
                                       "comparable_to_manfredi_wang": True})
    rows = {r["area"]: r["comparable"] for r in rep.collect_oracle(tmp_path)}
    assert rows["all"] is True
    assert rows["small"] is False, "a small-area envelope is not their Table 1"
