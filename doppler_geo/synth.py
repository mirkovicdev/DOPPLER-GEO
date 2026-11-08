"""Synthetic scenario: prop + jet traffic, hilly terrain, a receiver, its audio recording and ADS-B.

Everything is generated from one set of true 3D trajectories, so the audio, the ADS-B feed and
the ground truth are physically consistent:
  * audio is rendered sample-by-sample with exact retarded-time propagation in a windy atmosphere
    (phase of the received tone = phase emitted at tau(t)), 1/r spreading, absorption,
    propeller directivity, RPM wobble, harmonics, pink/brown noise, mains hum, wind gusts and
    broadband jet noise;
  * ADS-B is sampled irregularly with dropouts, GNSS noise and the real quantisation steps
    (25 ft altitude, 0.1 kt speed, 64 ft/min vertical rate).
The estimator is deliberately given an imperfect atmosphere (wrong temperature, no wind) and a
recording clock that is off by minutes.

All callsigns start with "SIM" to mark the traffic as synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch

from .adsb import FPM, FT, KT
from .geo import GridTerrain, LocalFrame
from .physics import emission_to_arrival, speed_of_sound, wind_vector


@dataclass
class AircraftType:
    code: str
    engine: str
    blades: int
    rpm: tuple
    speed: tuple    # m/s
    level_db: float  # source level relative to a C172


FLEET = [
    AircraftType("C172", "P", 2, (2350, 2550), (50, 60), 0.0),
    AircraftType("PA28", "P", 2, (2350, 2550), (52, 62), 0.0),
    AircraftType("C182", "P", 2, (2300, 2450), (60, 70), 1.0),
    AircraftType("SR22", "P", 3, (2500, 2700), (75, 88), 1.5),
    AircraftType("C208", "T", 3, (1750, 1900), (78, 90), 2.0),
    AircraftType("PC12", "T", 4, (1550, 1700), (105, 130), 2.0),
]
JET_TYPES = ["B738", "A320", "CRJ9", "E75L", "C56X"]


@dataclass
class TruthTrack:
    icao24: str
    callsign: str
    type_code: str
    engine: str
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    bpf: float = 0.0
    level_db: float = 0.0
    harm_amp: tuple = (1.0, 0.6, 0.35, 0.2)
    role: str = "background"

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])

    def state(self, tau):
        x = (np.asarray(tau, float) - self.t[0]) / self.dt
        n = len(self.t)
        ok = (x >= 0) & (x <= n - 1)
        i = np.clip(np.floor(x).astype(np.int64), 0, n - 2)
        a = np.clip(x - i, 0, 1)[..., None]
        return self.pos[i] * (1 - a) + self.pos[i + 1] * a, self.vel[i] * (1 - a) + self.vel[i + 1] * a, ok

    @property
    def label(self):
        return self.callsign


@dataclass
class World:
    frame: LocalFrame
    terrain: GridTerrain
    tracks: list
    t_begin: float
    t_end: float
    observer: np.ndarray
    obs_lat: float
    obs_lon: float
    obs_h: float
    rec_start: float
    rec_duration: float
    temp_c: float
    wind: np.ndarray
    c: float
    audio: np.ndarray | None = None
    fs: float = 4000.0
    adsb: pd.DataFrame | None = None
    passes: list = field(default_factory=list)


def _smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    pad = np.r_[np.full(n, x[0]), x, np.full(n, x[-1])]
    return np.convolve(pad, k, mode="same")[n:-n]


def _piecewise(rng, n, dt, seg_s, p_zero, draw):
    out = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.uniform(*seg_s) / dt)
        out[i:i + L] = 0.0 if rng.random() < p_zero else draw()
        i += L
    return out


def _fly(rng, frame, t, ia, anchor_xy, heading, speed, alt, quiet_s=0.0, turny=True, climby=True):
    """Integrate a gently manoeuvring flight that is at anchor_xy (heading, alt) at t[ia]."""
    n, dt = len(t), float(t[1] - t[0])
    hr = _piecewise(rng, n, dt, (60, 240), 0.55, lambda: rng.choice([-1, 1]) * rng.uniform(0.3, 1.5)) if turny else np.zeros(n)
    vs = _piecewise(rng, n, dt, (120, 400), 0.7, lambda: rng.choice([-1, 1]) * rng.uniform(1, 3)) if climby else np.zeros(n)
    if quiet_s > 0:
        q = np.abs(t - t[ia]) < quiet_s
        hr[q] = 0.0
        vs[q] = 0.0
    hr = _smooth(hr, int(10 / dt))
    vs = _smooth(vs, int(10 / dt))
    hdg = heading + np.cumsum(hr) * dt
    hdg -= hdg[ia] - heading
    m = int(60 / dt)
    spd = speed + 1.5 * _smooth(rng.normal(0, 1, n), m) * np.sqrt(m)
    spd -= spd[ia] - speed
    alt_t = alt + np.cumsum(vs) * dt
    alt_t -= alt_t[ia] - alt
    alt_t = _smooth(np.clip(alt_t, alt - 600, alt + 900), int(10 / dt))
    vx, vy = spd * np.sin(np.radians(hdg)), spd * np.cos(np.radians(hdg))
    x = np.cumsum(vx) * dt
    y = np.cumsum(vy) * dt
    x += anchor_xy[0] - x[ia]
    y += anchor_xy[1] - y[ia]
    up, _, _ = frame.up_for_height(x, y, lambda la, lo: alt_t)
    pos = np.stack([x, y, up], -1)
    return pos, np.gradient(pos, dt, axis=0)


def _circuits(frame, t, airport_xy, airport_h, heading, speed=45.0):
    """Touch-and-go traffic pattern (left-hand circuits) around an airfield."""
    # leg lengths chosen so the circuit closes: downwind = upwind + final, crosswind = base
    legs = [("up", 45, 0), ("turn", 30, -3), ("x", 30, 0), ("turn", 30, -3), ("down", 100, 0),
            ("turn", 30, -3), ("base", 30, 0), ("turn", 30, -3), ("final", 55, 0)]
    alt_pts = {"up": (20, 250), "x": (250, 300), "down": (300, 300), "base": (300, 160), "final": (160, 15)}
    dt = float(t[1] - t[0])
    hr = np.zeros(len(t))
    agl = np.zeros(len(t))
    i, prev = 0, 20.0
    while i < len(t):
        for name, dur, rate in legs:
            L = int(dur / dt)
            hr[i:i + L] = rate
            a0, a1 = alt_pts.get(name, (prev, prev))
            agl[i:i + L] = np.linspace(a0, a1, L)[: max(0, min(L, len(t) - i))]
            prev = a1
            i += L
            if i >= len(t):
                break
    hdg = heading + np.cumsum(hr) * dt
    x = airport_xy[0] + np.cumsum(speed * np.sin(np.radians(hdg))) * dt
    y = airport_xy[1] + np.cumsum(speed * np.cos(np.radians(hdg))) * dt
    up, _, _ = frame.up_for_height(x, y, lambda la, lo: airport_h + _smooth(agl, int(8 / dt)))
    pos = np.stack([x, y, up], -1)
    return pos, np.gradient(pos, dt, axis=0)


def make_terrain(frame, half_km=75.0, base=420.0, seed=0):
    rng = np.random.default_rng(seed)
    lat_span = half_km / 111.0
    lon_span = half_km / (111.0 * np.cos(np.radians(frame.lat0)))
    lats = np.linspace(frame.lat0 - lat_span, frame.lat0 + lat_span, 301)
    lons = np.linspace(frame.lon0 - lon_span, frame.lon0 + lon_span, 301)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    xy = frame.to_enu(LA, LO, np.zeros_like(LA))
    x, y = xy[..., 0] / 1000, xy[..., 1] / 1000
    h = np.full(x.shape, base)
    for amp, lx, ly in ((60, 23, 31), (35, 11, 9), (18, 5, 7), (9, 2.3, 3.1)):
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        h += amp * np.sin(2 * np.pi * x / lx + p1) * np.cos(2 * np.pi * y / ly + p2)
    return GridTerrain(lats, lons, h)


def make_world(seed: int = 7, lat0: float = 38.45, lon0: float = -98.20, n_background: int = 70,
               n_jets: int = 14, n_targets: int = 2, rec_duration: float = 1200.0,
               temp_c: float = 6.0, wind_ms: float = 6.0, wind_from: float = 240.0) -> World:
    rng = np.random.default_rng(seed)
    frame = LocalFrame(lat0, lon0)
    terrain = make_terrain(frame, seed=seed)
    c = speed_of_sound(temp_c)
    wind = wind_vector(wind_ms, wind_from)
    t_begin = 1_762_000_000.0 + rng.uniform(0, 86400 * 30)
    t_begin = float(np.floor(t_begin / 60) * 60)
    t_end = t_begin + 3 * 3600.0
    dt = 0.2
    T = np.arange(t_begin, t_end, dt)

    ox, oy = rng.uniform(-12000, 12000, 2)
    oup, olat, olon = frame.ground_up(np.array([ox]), np.array([oy]), terrain)
    obs_h = float(terrain.elevation(olat, olon)[0]) + 1.5
    oup, olat, olon = frame.up_for_height(np.array([ox]), np.array([oy]), lambda la, lo: np.array([obs_h]))
    observer = np.array([ox, oy, oup[0]])
    ground = float(obs_h)
    rec_start = float(np.floor((t_begin + rng.uniform(4500, 6500)) / 1) * 1)

    tracks = []
    used = set()

    def ident(prefix="SIM"):
        while True:
            n = int(rng.integers(100, 999))
            if n not in used:
                used.add(n)
                return f"a{rng.integers(0, 16**5):05x}", f"{prefix}{n}"

    def bpf_of(at):
        return at.blades * rng.uniform(*at.rpm) / 60.0

    # Engineered flyovers during the recording (close enough to be audible)
    t_arrs = [rec_start + rng.uniform(150, 330)]
    for _ in range(n_targets - 1):
        t_arrs.append(t_arrs[-1] + rng.uniform(380, 700))
    headings = []
    for k, ta in enumerate(t_arrs):
        at = FLEET[rng.integers(0, len(FLEET))]
        while True:
            hdg = rng.uniform(0, 360)
            if all(abs(((hdg - h) + 180) % 360 - 180) > 35 for h in headings):
                break
        headings.append(hdg)
        off = rng.uniform(350, 1500) * rng.choice([-1, 1])
        agl = rng.uniform(350, 1300)
        spd = rng.uniform(*at.speed)
        perp = np.radians(hdg + 90)
        anchor = (ox + off * np.sin(perp), oy + off * np.cos(perp))
        tau_ca = ta - np.hypot(off, agl) / c
        span = (tau_ca - rng.uniform(1500, 2500), tau_ca + rng.uniform(1500, 2500))
        sel = (T >= span[0]) & (T <= span[1])
        t = T[sel]
        ia = int(np.argmin(np.abs(t - tau_ca)))
        pos, vel = _fly(rng, frame, t, ia, anchor, hdg, spd, ground + agl, quiet_s=240)
        icao, cs = ident()
        tracks.append(TruthTrack(icao, cs, at.code, at.engine, t, pos, vel, bpf_of(at), at.level_db, role="target"))

    # Background prop traffic
    for _ in range(n_background):
        at = FLEET[rng.integers(0, len(FLEET))]
        tm = rng.uniform(t_begin, t_end)
        span = (max(t_begin, tm - rng.uniform(1200, 3000)), min(t_end, tm + rng.uniform(1200, 3000)))
        t = T[(T >= span[0]) & (T <= span[1])]
        if len(t) < 600:
            continue
        ia = int(np.argmin(np.abs(t - tm)))
        pos, vel = _fly(rng, frame, t, ia, rng.uniform(-55000, 55000, 2), rng.uniform(0, 360),
                        rng.uniform(*at.speed), ground + rng.uniform(400, 2800))
        icao, cs = ident()
        tracks.append(TruthTrack(icao, cs, at.code, at.engine, t, pos, vel, bpf_of(at), at.level_db))

    # Pattern traffic at an airfield ~9-15 km from the microphone
    ang = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(9000, 15000)
    ap = (ox + dist * np.cos(ang), oy + dist * np.sin(ang))
    _, aplat, aplon = frame.ground_up(np.array([ap[0]]), np.array([ap[1]]), terrain)
    ap_h = float(terrain.elevation(aplat, aplon)[0])
    rwy = rng.uniform(0, 360)
    for k in range(2):
        at = FLEET[k]
        t = T[T >= t_begin + 300 * k]
        pos, vel = _circuits(frame, t, ap, ap_h, rwy)
        icao, cs = ident()
        tracks.append(TruthTrack(icao, cs, at.code, at.engine, t, pos, vel, bpf_of(at), at.level_db, role="pattern"))

    # Jets at cruise altitude (broadband only; filtered out of the tonal search by type)
    for _ in range(n_jets):
        tm = rng.uniform(t_begin, t_end)
        t = T[(T >= max(t_begin, tm - 600)) & (T <= min(t_end, tm + 600))]
        if len(t) < 100:
            continue
        ia = len(t) // 2
        pos, vel = _fly(rng, frame, t, ia, rng.uniform(-50000, 50000, 2), rng.uniform(0, 360),
                        rng.uniform(210, 245), rng.uniform(9000, 11500), turny=False, climby=False)
        icao, cs = ident()
        tracks.append(TruthTrack(icao, cs, JET_TYPES[rng.integers(0, len(JET_TYPES))], "J", t, pos, vel))

    w = World(frame, terrain, tracks, t_begin, t_end, observer, float(olat[0]), float(olon[0]), obs_h,
              rec_start, rec_duration, temp_c, wind, c)
    w.passes = true_passes(w)
    return w


def true_passes(w: World, rmax: float = 8000.0):
    """Closest approaches of every prop aircraft to the microphone that are heard during the recording."""
    out = []
    for tr in w.tracks:
        if tr.engine not in "PT":
            continue
        tau = np.arange(w.rec_start - 60, w.rec_start + w.rec_duration, 0.5)
        pos, vel, ok = tr.state(tau)
        if ok.sum() < 3:
            continue
        t_arr, g, r = emission_to_arrival(tau[ok], pos[ok], vel[ok], w.observer, w.c, w.wind)
        for i in np.flatnonzero((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:])) + 1:
            if r[i] < rmax and w.rec_start <= t_arr[i] <= w.rec_start + w.rec_duration:
                out.append(dict(callsign=tr.callsign, type=tr.type_code, role=tr.role,
                                t_rel=float(t_arr[i] - w.rec_start), slant_m=float(r[i]),
                                speed_ms=float(np.linalg.norm(vel[ok][i])), bpf_hz=tr.bpf))
    return sorted(out, key=lambda d: d["t_rel"])


def _colored_noise(rng, n, fs, alpha=1.0, fmin=2.0):
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec *= 1.0 / np.maximum(f, fmin) ** (alpha / 2)
    x = np.fft.irfft(spec, n)
    return x / x.std()


def render_audio(w: World, fs: float = 4000.0, snr_1km_db: float = 24.0, seed: int = 0,
                 hum: bool = True, jets: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    n = int(w.rec_duration * fs)
    t = w.rec_start + np.arange(n) / fs

    noise = 0.8 * _colored_noise(rng, n, fs, 1.0) + 0.6 * _colored_noise(rng, n, fs, 2.0)
    gust_env = 1 + 2.5 * np.clip(_colored_noise(rng, n, fs, 2.0, 0.02), 0, None)
    sos = butter(4, 25, "low", fs=fs, output="sos")
    noise += 0.8 * sosfilt(sos, rng.normal(size=n)) * gust_env
    fw, pw = welch(noise[: int(min(n, 240 * fs))], fs, nperseg=int(2 * fs))
    psd100 = float(np.interp(100.0, fw, pw))
    enbw = 1.5 / 2.0  # Hann, 2 s window
    a1km = np.sqrt(2 * psd100 * enbw * 10 ** (snr_1km_db / 10))

    sig = noise.copy()
    if hum:
        for h, a in ((60, 6.0), (120, 3.0), (180, 1.5)):
            sig += a * a1km * np.sin(2 * np.pi * h * (t - w.rec_start) + rng.uniform(0, 6.28))

    alpha_db_per_m = 0.0004
    for tr in w.tracks:
        tau = np.arange(w.rec_start - 70, w.rec_start + w.rec_duration + 1, 0.05)
        pos, vel, ok = tr.state(tau)
        if ok.sum() < 10:
            continue
        tau, pos, vel = tau[ok], pos[ok], vel[ok]
        t_arr, g, r = emission_to_arrival(tau, pos, vel, w.observer, w.c, w.wind)
        if r.min() > (25000 if tr.engine == "J" else 15000):
            continue
        tt = np.interp(t, t_arr, tau, left=np.nan, right=np.nan)
        live = np.isfinite(tt)
        if not live.any():
            continue
        spread = (1000.0 / r) * 10 ** (-alpha_db_per_m * r / 20)
        if tr.engine == "J":
            if not jets:
                continue
            env = np.interp(tt[live], tau, spread) * 6 * a1km
            sosj = butter(2, [50, 1200], "band", fs=fs, output="sos")
            sig[live] += env * sosfilt(sosj, rng.normal(size=live.sum())) * 3
            continue
        ray = w.observer - pos
        cosang = np.einsum("ij,ij->i", ray, vel) / (np.linalg.norm(ray, axis=1) * np.linalg.norm(vel, axis=1) + 1e-9)
        direct = 0.5 + 0.5 * (1 - cosang**2)
        wobble = 1 + 0.003 * _smooth(rng.normal(0, 1, len(tau)), 400) * np.sqrt(400)  # ~0.3 % RPM wander, 20 s
        phase = 2 * np.pi * np.cumsum(tr.bpf * wobble) * 0.05
        amp = a1km * 10 ** (tr.level_db / 20) * spread * direct
        ph = np.interp(tt[live], tau, phase)
        am = np.interp(tt[live], tau, amp)
        for h, ha in enumerate(tr.harm_amp, start=1):
            sig[live] += am * ha * np.sin(h * ph + rng.uniform(0, 6.28))
    return (0.8 * sig / np.abs(sig).max()).astype(np.float32)


def synth_adsb(w: World, seed: int = 0, region_m: float = 65000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    frames = []
    for tr in w.tracks:
        times, t = [], tr.t[0] + rng.uniform(0, 1)
        while t < tr.t[-1]:
            times.append(t)
            t += rng.uniform(0.4, 1.6)
            if rng.random() < 0.003:
                t += rng.uniform(5, 40)
        times = np.array(times)
        pos, vel, ok = tr.state(times)
        keep = ok & (np.abs(pos[:, 0]) < region_m) & (np.abs(pos[:, 1]) < region_m)
        if keep.sum() < 10:
            continue
        times, pos, vel = times[keep], pos[keep], vel[keep]
        noisy = pos + np.c_[rng.normal(0, 6, (len(pos), 2)), np.zeros(len(pos))]
        lat, lon, h = w.frame.to_geodetic(noisy)
        h = np.round((h + rng.normal(0, 3, len(h))) / (25 * FT)) * (25 * FT)
        vl = w.frame.vector_from_enu(lat, lon, vel)
        gs = np.round((np.hypot(vl[:, 0], vl[:, 1]) + rng.normal(0, 0.2, len(h))) / KT, 1) * KT
        trk = (np.degrees(np.arctan2(vl[:, 0], vl[:, 1])) + rng.normal(0, 0.2, len(h))) % 360
        vr = np.round(vl[:, 2] / (64 * FPM)) * (64 * FPM)
        frames.append(pd.DataFrame(dict(time=times, icao24=tr.icao24, lat=lat, lon=lon, alt_m=h, gs_ms=gs,
                                        track_deg=trk, vrate_ms=vr, callsign=tr.callsign, type_code=tr.type_code)))
    return pd.concat(frames, ignore_index=True).sort_values("time", ignore_index=True)
