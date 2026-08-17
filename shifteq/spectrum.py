"""Is AP(delta) grid-locked, or is it noise?

The question this answers
-------------------------
If shift-variance comes from strided downsampling violating the sampling theorem (Zhang,
ICML'19), the detector's response should be *periodic in the stride*: translating by exactly
one stride returns every feature-map cell to the same sub-cell phase, so AP should repeat
with period 8, 16 or 32 px. If instead the response is image-specific chaos, AP(delta) is
broadband and no period stands out.

Those two stories imply completely different mitigations -- anti-aliasing helps the first
and not the second -- so this is worth deciding with a test rather than by looking at a
heatmap. An earlier eyeball reading of the step-2 2-D grid in this project called the
response "close to spatially uncorrelated"; that claim was made off a coarse grid and is
exactly what this module exists to check properly.

Method
------
Sample AP at every integer delta along one axis over two full strides (delta in [0, 63], 64
points), remove the mean and any linear drift, and take the power spectrum. With 64 samples
the bins land exactly on the periods of interest: period 32 px is bin 2, 16 px is bin 4,
8 px is bin 8. No interpolation, no leakage at those bins.

Significance comes from a permutation null: shuffle the AP values across delta many times
and recompute the spectrum. Under the null there is no ordering in delta, so any real
periodicity vanishes while the marginal distribution of AP values is preserved exactly.
That controls for the finite-image-sample noise floor, which is the thing most likely to
manufacture a spurious peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Spectrum:
    """Power spectrum of AP(delta) along one axis."""

    delta: np.ndarray        # [N] integer shifts, must be contiguous 0..N-1
    values: np.ndarray       # [N] AP at each shift
    freqs: np.ndarray        # [N//2+1] cycles per pixel
    periods: np.ndarray      # [N//2+1] pixels per cycle (inf at DC)
    power: np.ndarray        # [N//2+1] normalised so sum(power[1:]) == 1
    p_values: np.ndarray     # [N//2+1] per-bin permutation p-value (valid only for a bin
                             #          chosen A PRIORI, e.g. a network stride)
    p_peak_fwer: float       # family-wise p for "any periodicity at all", via the
                             #          max-over-bins null -- the correct test for a peak
                             #          that was selected by being the maximum
    p_stride_fwer: float     # family-wise p over the stride bins only
    n_permutations: int

    def peak(self, exclude_dc: bool = True) -> dict:
        """The strongest bin.

        `p_value` here is the FAMILY-WISE corrected one. Reporting the per-bin p-value for a
        bin selected by being the largest of 32 would overstate significance by roughly that
        factor, which is exactly the mistake this field of plot invites.
        """
        p = self.power.copy()
        if exclude_dc:
            p[0] = -np.inf
        k = int(np.argmax(p))
        return {"bin": k, "period_px": float(self.periods[k]),
                "power_frac": float(self.power[k]),
                "p_value": float(self.p_peak_fwer),
                "p_value_uncorrected": float(self.p_values[k])}

    def at_period(self, period: float) -> dict:
        """Power in the bin closest to a given period, e.g. a network stride."""
        with np.errstate(divide="ignore"):
            k = int(np.argmin(np.abs(self.periods - period)))
        return {"bin": k, "period_px": float(self.periods[k]),
                "power_frac": float(self.power[k]), "p_value": float(self.p_values[k])}


#: Every downsampling factor in the network, not just the detection strides.
#: YOLO26's stem halves resolution, the next stage halves it again, and only then do the
#: P3/P4/P5 heads sit at 8/16/32. Aliasing is introduced at *every* one of those steps, so
#: a periodicity test that only looks at 8/16/32 will call a stem-driven period-2 response
#: "broadband" and get the headline exactly backwards. It did, in the first run of this
#: project, which is why the full cascade is the default.
NETWORK_STRIDES = (2, 4, 8, 16, 32)


def _detrend(v: np.ndarray) -> np.ndarray:
    """Remove mean and linear drift.

    The linear term matters here: the canvas grows toward the bottom-right, so a slow drift
    with delta is expected from border effects alone and would otherwise smear power into
    the low-frequency bins and mimic a long period.
    """
    n = len(v)
    t = np.arange(n, dtype=np.float64)
    A = np.stack([np.ones(n), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def power_spectrum(values: np.ndarray, detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    if detrend:
        v = _detrend(v)
    n = len(v)
    amp = np.abs(np.fft.rfft(v)) ** 2
    tot = amp[1:].sum()
    return amp / tot if tot > 0 else amp, np.fft.rfftfreq(n, d=1.0)


def analyse(delta: np.ndarray, values: np.ndarray, n_permutations: int = 20000,
            seed: int = 0, detrend: bool = True) -> Spectrum:
    delta = np.asarray(delta)
    values = np.asarray(values, dtype=np.float64)
    if not np.array_equal(delta, np.arange(len(delta))):
        raise ValueError("delta must be contiguous integers 0..N-1 (step-1 sampling)")

    power, freqs = power_spectrum(values, detrend)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)

    stride_bins = sorted({int(np.argmin(np.abs(periods - s))) for s in NETWORK_STRIDES})
    obs_peak = power[1:].max()
    obs_stride = power[stride_bins].max()

    rng = np.random.default_rng(seed)
    ge = np.zeros_like(power)
    ge_peak = 0
    ge_stride = 0
    for _ in range(n_permutations):
        p, _ = power_spectrum(rng.permutation(values), detrend)
        ge += (p >= power)
        # max-over-bins null: how often does ANY bin of a shuffled curve beat our best bin?
        ge_peak += p[1:].max() >= obs_peak
        ge_stride += p[stride_bins].max() >= obs_stride
    # add-one smoothing: a p-value of exactly 0 is not supported by finite permutations
    denom = n_permutations + 1.0
    return Spectrum(delta=delta, values=values, freqs=freqs, periods=periods,
                    power=power, p_values=(ge + 1.0) / denom,
                    p_peak_fwer=(ge_peak + 1.0) / denom,
                    p_stride_fwer=(ge_stride + 1.0) / denom,
                    n_permutations=n_permutations)


def parity_effect(values: np.ndarray, n_permutations: int = 20000, seed: int = 0) -> dict:
    """Mean AP at even delta minus mean AP at odd delta.

    This is the interpretable form of a period-2 peak: a stride-2 stem makes even and odd
    translations land on different sub-sampling phases, so "shifting by an odd number of
    pixels costs X AP" is a claim a reader can act on, where "power at the Nyquist bin" is
    not.
    """
    v = np.asarray(values, dtype=np.float64)
    idx = np.arange(len(v))
    obs = float(v[idx % 2 == 0].mean() - v[idx % 2 == 1].mean())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_permutations):
        p = rng.permutation(v)
        if abs(float(p[idx % 2 == 0].mean() - p[idx % 2 == 1].mean())) >= abs(obs):
            ge += 1
    return {"even_minus_odd_AP": obs,
            "even_mean": float(v[idx % 2 == 0].mean()),
            "odd_mean": float(v[idx % 2 == 1].mean()),
            "p_value": (ge + 1.0) / (n_permutations + 1.0)}


def stride_report(spec: Spectrum, strides=NETWORK_STRIDES) -> dict:
    """Power and significance at each network stride -- the headline table.

    Bins are deduplicated before the total is taken. `at_period` snaps to the *closest*
    bin, and on a short sweep several strides snap to the same one -- at 4 sample points,
    periods 4, 8, 16 and 32 are all one bin -- so summing per stride would count that bin
    four times and report a "power fraction" above 1. The protocol sweep uses 64 points,
    where periods 2/4/8/16/32 land on distinct bins 32/16/8/4/2 and dedup is a no-op, so no
    published number changes; this matters for anyone re-running at another `max_shift`.
    `expected_by_chance` counts the same unique bins, so enrichment stays a like-for-like
    ratio.
    """
    out = {"peak": spec.peak(), "strides": {}}
    for s in strides:
        out["strides"][str(s)] = spec.at_period(s)
    unique_bins = {out["strides"][str(s)]["bin"]: out["strides"][str(s)]["power_frac"]
                   for s in strides}
    total = float(sum(unique_bins.values()))
    out["stride_power_total"] = total
    out["stride_bins"] = len(unique_bins)
    out["stride_bins_requested"] = len(strides)
    if len(unique_bins) < len(strides):
        out["note"] = (f"{len(strides)} strides collapsed onto {len(unique_bins)} FFT bins "
                       f"at N={len(spec.values)}; the sweep is too short to separate them")
    # Fraction of spectrum these bins would hold if power were spread uniformly. With 64
    # samples there are 32 non-DC bins, so 5 stride bins is ~16% by chance alone; the
    # threshold has to beat that, not zero.
    n_bins = max(len(spec.power) - 1, 1)
    out["expected_by_chance"] = len(unique_bins) / n_bins
    out["enrichment"] = total / out["expected_by_chance"] if out["expected_by_chance"] else 0.0
    out["min_stride_p_uncorrected"] = min(out["strides"][str(s)]["p_value"] for s in strides)
    out["p_stride_fwer"] = float(spec.p_stride_fwer)
    out["p_peak_fwer"] = float(spec.p_peak_fwer)
    out["interpretation"] = (
        "grid-locked" if out["enrichment"] >= 2.0 and spec.p_stride_fwer < 0.05
        else "broadband / not clearly stride-periodic"
    )
    return out


def dispersion(values: np.ndarray) -> dict:
    """How much a curve wobbles -- in AP points, and relative to its own level.

    The relative form exists for the cross-model table. RT-DETR sits at 53 AP and YOLOv8-N
    at 37, so "RT-DETR's curve moves by more AP points" would be partly a statement about
    where the models are on the AP scale, not about how shift-stable they are. Quoting the
    coefficient of variation alongside the absolute number keeps that distinction visible.
    """
    v = np.asarray(values, dtype=np.float64)
    mean = float(v.mean())
    mad = float(np.abs(v - mean).mean())
    return {"mean": mean, "mean_abs_dev": mad, "std": float(v.std(ddof=1)),
            "range": float(v.max() - v.min()), "min": float(v.min()), "max": float(v.max()),
            "cv_percent": 100.0 * mad / mean if mean else float("nan")}


def paired_dispersion_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 0,
                           relative: bool = False) -> dict:
    """Is curve `a` more dispersed than curve `b`, paired by shift position?

    Statistic: mean over deltas of (|a - mean(a)| - |b - mean(b)|). Null: the sign of each
    per-delta difference is exchangeable, which is what "the two curves are equally stable"
    implies once each is centred on its own mean. Pairing matters -- both curves were
    measured on the same images at the same deltas, so some deltas are simply hard for
    everything, and an unpaired test would count that shared difficulty as signal.

    `relative=True` divides each curve's deviations by its own mean first, which is the
    right comparison when the two curves sit at different AP levels (cross-model), and the
    wrong one when they do not (two heads of one checkpoint, where the AP levels are within
    a point of each other and the absolute cost is what a practitioner pays).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    da = np.abs(a - a.mean())
    db = np.abs(b - b.mean())
    if relative:
        da = da / a.mean() * 100.0
        db = db / b.mean() * 100.0
    d = da - db
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1.0) / (n_perm + 1.0)
    return {"mean_abs_dev_a": float(da.mean()), "mean_abs_dev_b": float(db.mean()),
            "difference": obs, "p_value": float(p), "n_perm": n_perm,
            "relative": bool(relative)}


def bootstrap_range_ci(per_image_ap: np.ndarray, n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    """CI for max-min of a curve, resampling IMAGES rather than shifts.

    `per_image_ap` is [n_shifts, n_images] of per-image AP contributions. Resampling images
    is the right unit: the shifts are the fixed design, the images are the random sample,
    and the question "is the e2e range really wider than the nms range" is a question about
    generalising to other images.
    """
    rng = np.random.default_rng(seed)
    n_shift, n_img = per_image_ap.shape
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_img, n_img)
        curve = per_image_ap[:, idx].mean(axis=1)
        stats[b] = curve.max() - curve.min()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(stats.mean()), "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "alpha": alpha}
