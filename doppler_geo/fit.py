"""Fit the straight-line, retarded-time Doppler model to one tone ridge.

Parameters, and the feature of the curve that determines each:
  f0    rest frequency (the propeller's blade-passage frequency)
  v     aircraft speed            <- how far the pitch drops   (f_hi - f_lo)/(f_hi + f_lo) = v/c
  d     closest-approach distance <- how steep the drop is     slope at centre = -f0 v^2 / (c d)
  t_ca  closest-approach time     <- where the drop is centred (heard at t_ca + d/c)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .physics import flyby_frequency


@dataclass
class Flyover:
    """One detected flyover: the observed ridge (fundamental-scaled) plus its Doppler fit."""

    t: np.ndarray
    f: np.ndarray
    w: np.ndarray
    f0: float
    v: float
    d: float
    t_ca: float              # emission time of closest approach (recording-relative s)
    c: float
    rms_hz: float
    sigma: dict
    harmonics: list = field(default_factory=list)
    ridges: list = field(default_factory=list)
    quality: str = "ok"
    name: str = ""           # F1, F2, ... for usable flyovers, in time order

    @property
    def t_arr(self) -> float:
        """Arrival time of the closest-approach sound (the inflection point of the observed curve)."""
        return self.t_ca + self.d / self.c

    def model(self, t):
        return flyby_frequency(t, self.f0, self.v, self.d, self.t_ca, self.c)

    def summary(self) -> str:
        return (f"{self.name or '--':3s} t={self.t_arr:7.1f}s  f0={self.f0:6.2f} Hz  v={self.v:5.1f} m/s  "
                f"d={self.d:6.0f} m  rms={self.rms_hz:.2f} Hz  [{self.quality}]")


def _initial_guesses(t, f, c):
    fs = np.convolve(f, np.ones(5) / 5, mode="same") if len(f) > 10 else f
    n = len(fs)
    k = max(2, n // 10)
    f_hi, f_lo = np.median(fs[:k]), np.median(fs[-k:])
    f_mid = 0.5 * (f_hi + f_lo)
    cross = np.flatnonzero((fs[:-1] - f_mid) * (fs[1:] - f_mid) <= 0)
    t_mid = t[cross[len(cross) // 2]] if len(cross) else t[n // 2]
    near = np.abs(t - t_mid) < 6
    slope = np.polyfit(t[near], f[near], 1)[0] if near.sum() >= 4 else -0.3
    v0 = float(np.clip(c * (f_hi - f_lo) / max(f_hi + f_lo, 1e-9), 30, 150))
    starts = []
    for v in sorted({v0, 45.0, 65.0, 90.0, 125.0}):
        d0 = float(np.clip(f_mid * v * v / (c * max(abs(slope), 1e-3)), 150, 8000))
        for d in (d0 * 0.5, d0, d0 * 2):
            d = float(np.clip(d, 100, 12000))
            starts.append([f_mid, v, d, t_mid - d / c])
    return starts


def fit_flyover(t, f, w=None, c: float = 340.0, v_bounds=(15.0, 220.0), d_bounds=(50.0, 15000.0),
                loss_scale_hz: float = 0.4, oversample: float = 4.0) -> Flyover:
    """Robust least-squares fit with multi-start. `oversample` (= window/hop) inflates the
    parameter sigmas because neighbouring spectrogram frames are not independent."""
    t, f = np.asarray(t, float), np.asarray(f, float)
    w = np.ones_like(f) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)

    def resid(p):
        return sw * (flyby_frequency(t, p[0], p[1], p[2], p[3], c) - f)

    lo = [0.5 * f.min(), v_bounds[0], d_bounds[0], t.min() - 600]
    hi = [2.0 * f.max(), v_bounds[1], d_bounds[1], t.max() + 600]
    best = None
    for x0 in _initial_guesses(t, f, c):
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=loss_scale_hz,
                          x_scale=[1.0, 10.0, 300.0, 5.0], max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    res = (flyby_frequency(t, *p, c) - f)
    rms = float(np.sqrt(np.sum(w * res**2) / np.sum(w)))
    try:
        steps = np.array([1e-3, 1e-2, 1.0, 1e-2])
        J = np.stack([(resid(p + np.eye(4)[i] * steps[i]) - resid(p - np.eye(4)[i] * steps[i])) / (2 * steps[i])
                      for i in range(4)], axis=1)
        dof = max(len(t) - 4, 1)
        cov = np.linalg.inv(J.T @ J) * (np.sum(w * res**2) / dof) * oversample
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(4, np.nan)
    fly = Flyover(t, f, w, float(p[0]), float(p[1]), float(p[2]), float(p[3]), c, rms,
                  dict(zip(("f0", "v", "d", "t_ca"), map(float, sig))))
    fly.quality = assess(fly, v_bounds, d_bounds)
    return fly


def assess(fly: Flyover, v_bounds, d_bounds) -> str:
    """'ok' only if the ridge brackets closest approach and the fit is sane."""
    if not (fly.t.min() + 8 < fly.t_arr < fly.t.max() - 8):
        return "partial"     # only the approach or only the departure was heard
    if fly.rms_hz > 1.5:
        return "poor-fit"
    if not (v_bounds[0] * 1.05 < fly.v < v_bounds[1] * 0.95 and d_bounds[0] * 1.05 < fly.d < d_bounds[1] * 0.95):
        return "at-bound"
    if fly.sigma.get("d", 0) > 0.5 * fly.d:
        return "unconstrained"
    if fly.v < MIN_FLYING_SPEED:
        return "too-slow"    # below a light plane's flying speed: almost always a ridge fragment
    if fly.d > MAX_USEFUL_RANGE:
        return "too-far"
    return "ok"


MIN_FLYING_SPEED = 33.0     # m/s (~64 kt)
MAX_USEFUL_RANGE = 8000.0   # m, matches SearchConfig.max_range_m


def _fit_members(members, c, **kw):
    t = np.concatenate([r.t for r, _ in members])
    f = np.concatenate([r.f / h for r, h in members])
    w = np.concatenate([np.clip(r.snr, 0, 30) for r, _ in members])
    o = np.argsort(t, kind="stable")
    fly = fit_flyover(t[o], f[o], w[o] / w.mean(), c=c, **kw)
    fly.harmonics, fly.ridges = [h for _, h in members], [r for r, _ in members]
    return fly


def fit_groups(groups, c: float, good_rms_hz: float = 0.8, **kw) -> list[Flyover]:
    """Fit each harmonic group. A badly fitting group is pruned: repeatedly drop the member whose
    removal improves the Doppler fit most (it belonged to another plane); leftovers are fitted alone."""
    out = []
    queue = [list(zip(g.ridges, g.harmonics)) for g in groups]
    while queue:
        members = queue.pop(0)
        fly = _fit_members(members, c, **kw)
        while fly.rms_hz > good_rms_hz and len(members) > 1:
            trials = [(_fit_members(members[:i] + members[i + 1:], c, **kw), i) for i in range(len(members))]
            best, i = min(trials, key=lambda x: x[0].rms_hz)
            if best.rms_hz > 0.7 * fly.rms_hz:
                break
            r, _ = members.pop(i)
            queue.append([(r, 1)])
            fly = best
        out.append(fly)
    return dedupe(out)


def _is_harmonic_of(a: Flyover, b: Flyover, tol: float = 0.02, max_dt: float = 30.0) -> bool:
    """a's ridge overlaps b's in time at a steady integer frequency ratio (or its inverse)."""
    if abs(a.t_arr - b.t_arr) > max_dt:
        return False
    lo, hi = max(a.t.min(), b.t.min()), min(a.t.max(), b.t.max())
    if hi - lo < 0.5 * min(np.ptp(a.t), np.ptp(b.t)):
        return False
    tt = np.linspace(lo, hi, 40)
    ratio = np.interp(tt, a.t, a.f) / np.interp(tt, b.t, b.f)
    r = float(np.median(ratio))
    if r < 1:
        r, ratio = 1 / r, 1 / ratio
    return np.std(ratio) / r < 0.03 and abs(r - round(r)) / r < tol


def dedupe(flyovers, dt: float = 10.0):
    """Drop second detections of one flyover (ungrouped harmonics or fragments): when two fits
    overlap in time at an integer pitch ratio, or agree on closest-approach time, keep the
    better-supported one."""
    keep = []
    for fl in sorted(flyovers, key=lambda x: (x.quality != "ok", -np.sum(x.w))):
        dup = any(k.quality == "ok" and (abs(fl.t_arr - k.t_arr) <= dt or _is_harmonic_of(fl, k)) for k in keep)
        if fl.quality == "ok" and dup:
            fl.quality = "duplicate"
        keep.append(fl)
    keep.sort(key=lambda x: x.t_arr)
    for i, fl in enumerate(f for f in keep if f.quality == "ok"):
        fl.name = f"F{i + 1}"
    return keep
