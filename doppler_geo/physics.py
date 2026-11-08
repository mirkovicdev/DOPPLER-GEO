"""Acoustic Doppler physics for a propeller tone heard by a fixed microphone.

Notation
--------
tau  emission time: when the propeller blade made the sound
t    arrival time:  when the microphone heard it,  t = tau + R(tau) / c
R    slant range aircraft -> microphone at the emission time
v_r  radial velocity dR/dtau, positive while the aircraft recedes

Phase is conserved along the ray, so the received frequency is the emitted frequency
divided by how fast arrival time runs relative to emission time:

    f_obs = f0 / (dt/dtau) = f0 / (1 + v_r/c) = f0 * c / (c + v_r)

with v_r evaluated at the *emission* time (the "retarded-time" correction). Ignoring that
correction makes the fitted curve symmetric when the real one is not, which biases the
recovered distance and time of closest approach.
"""
from __future__ import annotations

import numpy as np

MAX_DELAY_S = 45.0  # longest sound travel time considered (~15 km)


def speed_of_sound(temp_c: float = 15.0) -> float:
    """Speed of sound in dry air (m/s): 331.3 * sqrt(1 + T/273.15)."""
    return float(331.3 * np.sqrt(1.0 + np.asarray(temp_c) / 273.15))


# --------------------------------------------------------------------------------------
# Straight, constant-speed flyby: closed form, used to fit a single observed curve.
# The aircraft passes at minimum slant distance d at emission time t_ca with speed v.
# --------------------------------------------------------------------------------------

def flyby_emission_time(t, t_ca, v, d, c):
    """Solve t = tau + R(tau)/c for tau, R(tau) = sqrt(d^2 + v^2 (tau - t_ca)^2).

    Squaring gives a quadratic in u = tau - t_ca; the root with tau < t is
        u = (c^2 T - sqrt(c^2 v^2 T^2 + (c^2 - v^2) d^2)) / (c^2 - v^2),   T = t - t_ca.
    """
    T = np.asarray(t, float) - t_ca
    c2, v2 = c * c, v * v
    return t_ca + (c2 * T - np.sqrt(c2 * v2 * T * T + (c2 - v2) * d * d)) / (c2 - v2)


def flyby_frequency(t, f0, v, d, t_ca, c, retarded: bool = True):
    """Received tone frequency at arrival times t for a straight flyby.

    retarded=False gives the naive model that evaluates the geometry at the arrival time
    (ignores the sound's travel time); it is kept only to show why that is wrong.
    """
    t = np.asarray(t, float)
    u = (flyby_emission_time(t, t_ca, v, d, c) if retarded else t) - t_ca
    r = np.sqrt(d * d + v * v * u * u)
    return f0 * c / (c + v * v * u / r)


def flyby_arrival_ca(t_ca, d, c):
    """Arrival time of the sound emitted at closest approach (the curve's inflection point)."""
    return t_ca + d / c


def flyby_center_slope(f0, v, d, c):
    """df/dt at the closest-approach arrival: -f0 v^2 / (c d). Steeper curve = closer plane."""
    return -f0 * v * v / (c * d)


# --------------------------------------------------------------------------------------
# General trajectory (turns, climbs) with optional uniform wind: exact numerical model.
# --------------------------------------------------------------------------------------

def propagation_delay(src, obs, c, wind=None):
    """Sound travel time from src to obs (arrays (...,3), metres).

    In a uniform wind w the wavefront is carried with the air, so the travel time T solves
    |D - w T| = c T with D = obs - src:
        T = (-(D.w) + sqrt((D.w)^2 + (c^2 - |w|^2) |D|^2)) / (c^2 - |w|^2)
    """
    D = np.asarray(obs, float) - np.asarray(src, float)
    d2 = np.einsum("...i,...i->...", D, D)
    if wind is None:
        return np.sqrt(d2) / c
    w = np.asarray(wind, float)
    dw = D @ w
    w2 = float(w @ w)
    return (-dw + np.sqrt(dw * dw + (c * c - w2) * d2)) / (c * c - w2)


def emission_to_arrival(tau, src_pos, src_vel, obs, c, wind=None):
    """Map emission samples (tau, position, velocity) to (arrival time, f_obs/f0, slant range).

    dt/dtau = 1 + dT/dtau. Differentiating |D - wT| = cT with dD/dtau = -v gives
        dt/dtau = 1 - (k . v) / (c^2 T + k . w),      k = D - w T  (air-frame ray, |k| = cT)
    which reduces to 1 + v_r/c in still air.
    """
    tau = np.asarray(tau, float)
    src_pos = np.asarray(src_pos, float)
    src_vel = np.asarray(src_vel, float)
    D = np.asarray(obs, float) - src_pos
    T = propagation_delay(src_pos, obs, c, wind)
    if wind is None:
        k = D
        kw = 0.0
    else:
        w = np.asarray(wind, float)
        k = D - T[..., None] * w
        kw = k @ w
    kv = np.einsum("...i,...i->...", k, src_vel)
    dt_dtau = 1.0 - kv / (c * c * T + kw)
    return tau + T, 1.0 / dt_dtau, np.linalg.norm(D, axis=-1)


def doppler_curve(track, obs, t_lo, t_hi, c, wind=None, step: float = 0.5):
    """Doppler factor g(t) = f_obs/f0 heard at `obs` for arrival times spanning [t_lo, t_hi].

    Returns (t_arrival, g, slant_range) sampled on an emission-time grid, or None if the
    track has no data there. Interpolate g at any arrival time with np.interp.
    """
    tau = np.arange(t_lo - MAX_DELAY_S, t_hi + step, step)
    pos, vel, ok = track.state(tau)
    if ok.sum() < 4:
        return None
    return emission_to_arrival(tau[ok], pos[ok], vel[ok], obs, c, wind)


def wind_vector(speed_ms: float, from_deg: float):
    """Meteorological wind (blowing FROM `from_deg`, clockwise from north) as an ENU vector."""
    a = np.radians(from_deg)
    return np.array([-speed_ms * np.sin(a), -speed_ms * np.cos(a), 0.0])
