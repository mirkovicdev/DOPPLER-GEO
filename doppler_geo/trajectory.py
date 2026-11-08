"""Aircraft tracks: raw ADS-B reports -> smoothed, uniformly sampled trajectories in the local frame.

ADS-B gives GNSS positions (tens of metres of noise) and GNSS velocity (sub-m/s). The Doppler
curve depends on velocity and on the direction to the microphone, so both are fused with a
Kalman/RTS smoother per axis, then resampled to a uniform 1 s grid with a cubic Hermite spline.
Short dropouts (< max_gap, 60 s) are bridged by the smoother; longer gaps split the track and
nothing is extrapolated across them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from ._kernels import rts_cv
from .geo import LocalFrame


@dataclass
class Flight:
    """Raw reports for one aircraft. Units: s, deg, m (ellipsoidal), m/s."""

    icao24: str
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    gs: np.ndarray | None = None       # ground speed, m/s
    track: np.ndarray | None = None    # degrees true
    vrate: np.ndarray | None = None    # m/s, positive up
    callsign: str = ""
    type_code: str = ""


@dataclass
class FlightTrack:
    """Smoothed trajectory on a uniform time grid in the search frame."""

    icao24: str
    callsign: str
    type_code: str
    t0: float
    dt: float
    pos: np.ndarray  # (N, 3) ENU metres
    vel: np.ndarray  # (N, 3) ENU m/s

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def t_end(self) -> float:
        return self.t0 + (self.n - 1) * self.dt

    @property
    def times(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.n)

    @property
    def label(self) -> str:
        return (self.callsign or self.icao24).strip()

    def state(self, tau):
        """Linearly interpolated (pos, vel, ok) at times tau; ok=False outside the track."""
        x = (np.asarray(tau, float) - self.t0) / self.dt
        ok = (x >= 0) & (x <= self.n - 1)
        i = np.clip(np.floor(x).astype(np.int64), 0, max(self.n - 2, 0))
        a = np.clip(x - i, 0.0, 1.0)[..., None]
        j = np.minimum(i + 1, self.n - 1)
        pos = self.pos[i] * (1 - a) + self.pos[j] * a
        vel = self.vel[i] * (1 - a) + self.vel[j] * a
        return pos, vel, ok

    def window(self, t_lo: float, t_hi: float) -> "FlightTrack | None":
        i0 = max(0, int(np.floor((t_lo - self.t0) / self.dt)))
        i1 = min(self.n, int(np.ceil((t_hi - self.t0) / self.dt)) + 1)
        if i1 - i0 < 3:
            return None
        return FlightTrack(self.icao24, self.callsign, self.type_code, self.t0 + i0 * self.dt,
                           self.dt, self.pos[i0:i1], self.vel[i0:i1])


def smooth_flight(flight: Flight, frame: LocalFrame, dt: float = 1.0, max_gap: float = 60.0,
                  sigma_pos: float = 15.0, sigma_alt: float = 8.0, sigma_vel: float = 0.7,
                  q_horiz: float = 1.0, q_vert: float = 0.3, min_points: int = 8) -> list[FlightTrack]:
    t = np.asarray(flight.t, float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([np.diff(t) > 1e-3, [True]])  # drop duplicate timestamps (keep last)
    idx = order[keep]
    t = t[keep]
    lat, lon, alt = (np.asarray(a, float)[idx] for a in (flight.lat, flight.lon, flight.alt))
    enu = frame.to_enu(lat, lon, alt)

    if flight.gs is not None and flight.track is not None:
        gs = np.asarray(flight.gs, float)[idx]
        trk = np.radians(np.asarray(flight.track, float)[idx])
        vr = np.asarray(flight.vrate, float)[idx] if flight.vrate is not None else np.zeros_like(gs)
        vr = np.where(np.isfinite(vr), vr, 0.0)
        vloc = np.stack([gs * np.sin(trk), gs * np.cos(trk), vr], -1)
        vel_meas = frame.vector_to_enu(lat, lon, vloc)
        has_v = np.isfinite(vel_meas).all(-1)
        vel_meas = np.where(has_v[:, None], vel_meas, 0.0)
    else:
        vel_meas = np.zeros_like(enu)
        has_v = np.zeros(len(t), bool)

    good = np.isfinite(enu).all(-1)
    t, enu, vel_meas, has_v = t[good], enu[good], vel_meas[good], has_v[good]
    if len(t) < min_points:
        return []

    out = []
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if e - s < min_points or t[e - 1] - t[s] < 10 * dt:
            continue
        ts = t[s:e]
        pos_s = np.empty((e - s, 3))
        vel_s = np.empty((e - s, 3))
        for ax in range(3):
            rp = (sigma_alt if ax == 2 else sigma_pos) ** 2
            q = q_vert if ax == 2 else q_horiz
            xs = rts_cv(ts, np.ascontiguousarray(enu[s:e, ax]), np.ascontiguousarray(vel_meas[s:e, ax]),
                        has_v[s:e], rp, sigma_vel**2, q)
            pos_s[:, ax], vel_s[:, ax] = xs[:, 0], xs[:, 1]
        spl = CubicHermiteSpline(ts, pos_s, vel_s, axis=0)
        tg = np.arange(np.ceil(ts[0] / dt) * dt, ts[-1], dt)
        if len(tg) < 3:
            continue
        out.append(FlightTrack(flight.icao24, flight.callsign, flight.type_code, float(tg[0]), dt,
                               spl(tg), spl(tg, 1)))
    return out


def smooth_flights(flights, frame: LocalFrame, **kw) -> list[FlightTrack]:
    tracks = []
    for f in flights:
        tracks.extend(smooth_flight(f, frame, **kw))
    return tracks
