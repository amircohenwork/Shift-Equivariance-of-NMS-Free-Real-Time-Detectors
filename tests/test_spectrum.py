"""The spectral test has to find a period when one exists and stay quiet when it doesn't.

A periodicity claim is easy to manufacture accidentally -- detrending artefacts, leakage,
or a permutation null that does not actually destroy the effect. These tests plant known
signals and check the machinery recovers them, then plant pure noise and check it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shifteq.spectrum import (analyse, bootstrap_range_ci, dispersion,  # noqa: E402
                              paired_dispersion_test, stride_report)

N = 64
D = np.arange(N)


@pytest.mark.parametrize("period", [8, 16, 32])
def test_recovers_a_planted_stride_period(period):
    v = 40 + 0.5 * np.sin(2 * np.pi * D / period)
    s = analyse(D, v, n_permutations=2000)
    pk = s.peak()
    assert pk["period_px"] == pytest.approx(period, abs=1e-6), pk
    assert pk["p_value"] < 0.01, pk
    assert stride_report(s)["interpretation"] == "grid-locked"


def test_pure_noise_is_not_called_grid_locked():
    rng = np.random.default_rng(0)
    calls = 0
    for seed in range(20):
        v = 40 + rng.normal(0, 0.2, N)
        s = analyse(D, v, n_permutations=2000, seed=seed)
        if stride_report(s)["interpretation"] == "grid-locked":
            calls += 1
    # a handful of false positives across 20 draws is expected at alpha=0.05; a majority
    # would mean the test is broken
    assert calls <= 4, f"{calls}/20 noise draws called grid-locked"


def test_linear_drift_alone_is_not_a_period():
    """The canvas grows toward the bottom-right, so a monotone drift with delta is expected
    from border effects. It must not masquerade as a long period."""
    v = 40 + 0.01 * D
    s = analyse(D, v, n_permutations=2000)
    assert s.power[1:].max() < 0.9 or s.peak()["p_value"] > 0.01, (
        "detrending failed: a pure ramp produced a significant spectral peak"
    )


def test_signal_plus_noise_still_found():
    rng = np.random.default_rng(1)
    v = 40 + 0.4 * np.sin(2 * np.pi * D / 16) + rng.normal(0, 0.15, N)
    s = analyse(D, v, n_permutations=4000)
    assert s.peak()["period_px"] == pytest.approx(16, abs=1e-6)
    assert s.at_period(16)["p_value"] < 0.05


def test_power_is_normalised_and_dc_excluded_from_peak():
    rng = np.random.default_rng(2)
    v = 40 + rng.normal(0, 0.3, N)
    s = analyse(D, v, n_permutations=500)
    assert s.power[1:].sum() == pytest.approx(1.0, rel=1e-9)
    assert s.peak()["bin"] != 0


def test_rejects_non_contiguous_delta():
    with pytest.raises(ValueError):
        analyse(np.array([0, 2, 4, 6]), np.zeros(4), n_permutations=10)


def test_p_values_never_exactly_zero():
    """Finite permutations cannot support p = 0; add-one smoothing must be applied."""
    v = 40 + np.sin(2 * np.pi * D / 8)
    s = analyse(D, v, n_permutations=100)
    assert s.p_values.min() > 0


def test_stride_power_is_a_fraction_even_on_a_short_sweep():
    """`at_period` snaps to the nearest bin, so on a short sweep several strides land on
    one bin. Summing per stride would then report a "power fraction" above 1 -- which a
    4-point smoke run did. The total must stay a fraction at any length."""
    rng = np.random.default_rng(11)
    for n in (4, 8, 16, 32, 64):
        d = np.arange(n)
        s = analyse(d, 40 + rng.normal(0, 0.3, n), n_permutations=100)
        r = stride_report(s)
        assert 0.0 <= r["stride_power_total"] <= 1.0 + 1e-9, (n, r["stride_power_total"])
        assert r["stride_bins"] <= r["stride_bins_requested"]


def test_protocol_length_separates_every_stride():
    """At the protocol's 64 points, periods 2/4/8/16/32 must land on distinct bins --
    otherwise the headline table is summing one bin several times."""
    s = analyse(D, 40 + np.sin(2 * np.pi * D / 16), n_permutations=100)
    r = stride_report(s)
    assert r["stride_bins"] == 5 == r["stride_bins_requested"], r
    assert "note" not in r
    bins = sorted(r["strides"][str(k)]["bin"] for k in (2, 4, 8, 16, 32))
    assert bins == [2, 4, 8, 16, 32], bins


def test_paired_test_is_calibrated_under_the_null():
    """Two curves drawn from the same process must not be declared different more often
    than alpha. A dispersion test that fires on identically-distributed inputs would
    manufacture a head gap out of nothing, which is the failure mode that matters here."""
    rng = np.random.default_rng(7)
    fired = 0
    for _ in range(200):
        a = 40 + rng.normal(0, 0.3, N)
        b = 40 + rng.normal(0, 0.3, N)
        if paired_dispersion_test(a, b, n_perm=500)["p_value"] < 0.05:
            fired += 1
    assert fired <= 20, f"{fired}/200 false positives at alpha=0.05"


def test_paired_test_detects_a_genuinely_wider_curve():
    rng = np.random.default_rng(8)
    base = rng.normal(0, 0.3, N)
    a = 40 + base * 3.0          # same shape, three times the amplitude
    b = 40 + base
    t = paired_dispersion_test(a, b, n_perm=5000)
    assert t["difference"] > 0 and t["p_value"] < 0.01, t


def test_paired_test_is_antisymmetric():
    rng = np.random.default_rng(9)
    a = 40 + rng.normal(0, 0.5, N)
    b = 40 + rng.normal(0, 0.2, N)
    ab = paired_dispersion_test(a, b, n_perm=2000)
    ba = paired_dispersion_test(b, a, n_perm=2000)
    assert ab["difference"] == pytest.approx(-ba["difference"], abs=1e-12)
    assert ab["p_value"] == pytest.approx(ba["p_value"], abs=1e-12)


def test_relative_mode_removes_the_ap_level():
    """Two curves with the same *fractional* wobble at different AP levels must come out
    equal in relative mode and unequal in absolute mode -- that is the whole point of
    quoting a CV in the cross-model table."""
    rng = np.random.default_rng(10)
    shape = rng.normal(0, 0.01, N)
    hi = 53.0 * (1 + shape)
    lo = 37.0 * (1 + shape)
    assert paired_dispersion_test(hi, lo, n_perm=2000)["difference"] > 0
    rel = paired_dispersion_test(hi, lo, n_perm=2000, relative=True)
    assert rel["difference"] == pytest.approx(0.0, abs=1e-9), rel


def test_dispersion_reports_consistent_summaries():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    d = dispersion(v)
    assert d["mean"] == pytest.approx(2.5)
    assert d["range"] == pytest.approx(3.0)
    assert d["mean_abs_dev"] == pytest.approx(1.0)
    assert d["cv_percent"] == pytest.approx(40.0)


def test_bootstrap_range_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    per_image = 40 + rng.normal(0, 5, (16, 400)) + np.sin(2 * np.pi * np.arange(16) / 8)[:, None]
    curve = per_image.mean(axis=1)
    point = curve.max() - curve.min()
    ci = bootstrap_range_ci(per_image, n_boot=400)
    assert ci["lo"] <= point <= ci["hi"], (ci, point)
    assert ci["lo"] >= 0
