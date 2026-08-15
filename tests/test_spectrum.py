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

from shifteq.spectrum import analyse, bootstrap_range_ci, stride_report  # noqa: E402

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


def test_bootstrap_range_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    per_image = 40 + rng.normal(0, 5, (16, 400)) + np.sin(2 * np.pi * np.arange(16) / 8)[:, None]
    curve = per_image.mean(axis=1)
    point = curve.max() - curve.min()
    ci = bootstrap_range_ci(per_image, n_boot=400)
    assert ci["lo"] <= point <= ci["hi"], (ci, point)
    assert ci["lo"] >= 0
