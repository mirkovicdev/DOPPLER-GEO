"""Geolocation search: which spot on the ground hears exactly these flyovers, at these times?

Stages (each prunes candidate locations; counts are reported as a funnel):
 1. Grid      - regular east/north grid on the terrain surface (default 200 m over 60 x 60 km).
 2. Passes    - for every prop track, every closest approach to every cell: time it would be heard
                (emission time + travel time), 3D slant distance, ground speed.
 3. Match     - flyover #1 must match a pass (distance/speed within tolerance, time inside the
                recording-clock uncertainty). That fixes a clock offset t0. Every further flyover
                must match another pass at the same cell at t0 + its recording time (+- tol).
 4. Silence   - an aircraft that came closer than the farthest matched flyover during the recording
                would have been audible. If no tone was detected at that time, the cell is rejected.
 5. Doppler   - predict the full curve each surviving hypothesis would produce (real ADS-B track,
                retarded time, terrain) and score it against the observed ridge (RMS in Hz).
 6. Refine    - continuous Nelder-Mead over (east, north, clock offset) from the best few.
"""
from __future__ import annotations

import itertools
import time as _time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._kernels import count_violations, find_passes
from .geo import LocalFrame
from .physics import doppler_curve


@dataclass
class Recording:
    start_guess: float      # unix time (UTC) of the first audio sample, best guess
    start_unc: float        # +- seconds
    duration: float         # seconds


@dataclass
class SearchConfig:
    half_size_m: float = 30_000.0
    grid_res_m: float = 200.0
    center_en: tuple = (0.0, 0.0)
    max_range_m: float = 8_000.0       # passes farther than this are never considered audible
    dist_ratio: tuple = (0.6, 1.6)     # pass slant range / fitted distance
    speed_ratio: tuple = (0.8, 1.25)   # pass ground speed / fitted speed
    time_tol_s: float = 15.0
    silence: str = "window"            # "window" (whole recording), "between" (first to last flyover), "off"
    silence_factor: float = 1.0        # threshold = factor x farthest matched slant range
    silence_edge_s: float = 30.0
    max_violations: int = 0
    max_scored: int = 20_000
    rescore_top: int = 400
    refine_top: int = 6
    surface_half_m: float = 1_000.0
    surface_res_m: float = 25.0
    miss_hz: float = 3.0
    oversample: float = 4.0            # spectrogram window / hop (correlated residuals)
    min_events: int = 1
    fit_atmosphere: bool = True        # refine also solves for wind (E, N) and speed of sound
    wind_prior_ms: float = 10.0
    c_prior_ms: float = 6.0


@dataclass
class Grid:
    e: np.ndarray
    n: np.ndarray
    up: np.ndarray    # (ny, nx)
    lat: np.ndarray
    lon: np.ndarray
    res: float

    @property
    def shape(self):
        return self.up.shape

    @property
    def size(self):
        return self.up.size

    def cell_enu(self, cell):
        cell = np.asarray(cell)
        iy, ix = np.divmod(cell, len(self.e))
        return np.stack([self.e[ix], self.n[iy], self.up[iy, ix]], -1)

    def mask(self, cells):
        m = np.zeros(self.size, bool)
        m[np.asarray(cells, np.int64)] = True
        return m.reshape(self.shape)


def build_grid(frame: LocalFrame, terrain, cfg: SearchConfig) -> Grid:
    h, r = cfg.half_size_m, cfg.grid_res_m
    e = cfg.center_en[0] + np.arange(-h, h + r / 2, r)
    n = cfg.center_en[1] + np.arange(-h, h + r / 2, r)
    E, N = np.meshgrid(e, n)
    up, lat, lon = frame.ground_up(E, N, terrain)
    return Grid(e, n, up, lat, lon, r)


@dataclass
class Passes:
    cell: np.ndarray
    track: np.ndarray
    tau: np.ndarray
    t_arr: np.ndarray
    r: np.ndarray
    speed: np.ndarray
    cell_start: np.ndarray

    def __len__(self):
        return len(self.cell)


def find_all_passes(grid: Grid, tracks, c: float, rmax: float) -> Passes:
    cap = 2_000_000
    while True:
        bufs = [np.empty(cap, np.int64), np.empty(cap, np.int64)] + [np.empty(cap) for _ in range(4)]
        count = 0
        up = np.ascontiguousarray(grid.up)
        for i, tr in enumerate(tracks):
            p, v = tr.pos, tr.vel
            count = find_passes(grid.e[0], grid.n[0], grid.res, len(grid.e), len(grid.n), up, tr.times,
                                *(np.ascontiguousarray(p[:, k]) for k in range(3)),
                                *(np.ascontiguousarray(v[:, k]) for k in range(3)),
                                rmax, c, i, *bufs, count)
        if count <= cap:
            break
        cap = int(count * 1.2)
    cell, track, tau, t_arr, r, speed = (b[:count] for b in bufs)
    o = np.lexsort((t_arr, cell))
    cell, track, tau, t_arr, r, speed = (a[o] for a in (cell, track, tau, t_arr, r, speed))
    cell_start = np.searchsorted(cell, np.arange(grid.size + 1)).astype(np.int64)
    return Passes(cell, track, tau, t_arr, r, speed, cell_start)


@dataclass
class Hypotheses:
    cell: np.ndarray
    t0: np.ndarray
    ids: np.ndarray   # (H, m) pass indices, one per event

    def __len__(self):
        return len(self.cell)

    def take(self, sel):
        return Hypotheses(self.cell[sel], self.t0[sel], self.ids[sel])


def _event_mask(P: Passes, ev, rec: Recording, cfg: SearchConfig):
    lo = rec.start_guess - rec.start_unc + ev.t_arr - cfg.time_tol_s
    hi = rec.start_guess + rec.start_unc + ev.t_arr + cfg.time_tol_s
    sd = np.nan_to_num(ev.sigma.get("d", 0.0)) if hasattr(ev, "sigma") else 0.0
    sv = np.nan_to_num(ev.sigma.get("v", 0.0)) if hasattr(ev, "sigma") else 0.0
    return ((P.r >= cfg.dist_ratio[0] * ev.d - 2 * sd) & (P.r <= cfg.dist_ratio[1] * ev.d + 2 * sd)
            & (P.speed >= cfg.speed_ratio[0] * ev.v - 2 * sv) & (P.speed <= cfg.speed_ratio[1] * ev.v + 2 * sv)
            & (P.t_arr >= lo) & (P.t_arr <= hi))


def match_events(P: Passes, events, rec: Recording, cfg: SearchConfig, stages: list) -> Hypotheses:
    """Chain-join passes across events at the same cell with a consistent clock offset t0."""
    idx = np.flatnonzero(_event_mask(P, events[0], rec, cfg))
    hyp = Hypotheses(P.cell[idx], P.t_arr[idx] - events[0].t_arr, idx[:, None])
    stages.append((f"{_ev_label(events[0])}: range, speed and epoch admissible", _ev_name(events[0]), hyp))
    for j in range(1, len(events)):
        ev = events[j]
        idxj = np.flatnonzero(_event_mask(P, ev, rec, cfg))
        cj = P.cell[idxj]
        lo = np.searchsorted(cj, hyp.cell, "left")
        cnt = np.searchsorted(cj, hyp.cell, "right") - lo
        tot = int(cnt.sum())
        if tot == 0:
            hyp = hyp.take(np.zeros(0, np.int64))
        else:
            hrep = np.repeat(np.arange(len(hyp)), cnt)
            within = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            q = idxj[np.repeat(lo, cnt) + within]
            ok = np.abs(P.t_arr[q] - (hyp.t0[hrep] + ev.t_arr)) <= cfg.time_tol_s
            ok &= np.all(hyp.ids[hrep] != q[:, None], axis=1)
            hrep, q = hrep[ok], q[ok]
            ids = np.hstack([hyp.ids[hrep], q[:, None]])
            t_rel = np.array([e.t_arr for e in events[:j + 1]])
            t0 = np.mean(P.t_arr[ids] - t_rel[None, :], axis=1)
            hyp = Hypotheses(hyp.cell[hrep], t0, ids)
        stages.append((f"+ {_ev_label(ev)}: consistent with a common clock epoch", f"+{_ev_name(ev)}", hyp))
    return hyp


def _ev_name(ev):
    return getattr(ev, "name", "") or f"t={ev.t_arr:.0f}s"


def _ev_label(ev):
    return f"{_ev_name(ev)} (d = {ev.d / 1000:.2f} km, v = {ev.v:.0f} m/s, t = {ev.t_arr:.0f} s)"


def silence_violations(P: Passes, hyp: Hypotheses, events, rec: Recording, cfg: SearchConfig, heard):
    if cfg.silence == "off" or len(hyp) == 0:
        return np.zeros(len(hyp), np.int64)
    rthr = cfg.silence_factor * P.r[hyp.ids].max(axis=1)
    if cfg.silence == "between":
        lo = hyp.t0 + min(e.t_arr for e in events)
        hi = hyp.t0 + max(e.t_arr for e in events)
    else:
        lo = hyp.t0 + cfg.silence_edge_s
        hi = hyp.t0 + rec.duration - cfg.silence_edge_s
    heard = np.asarray(heard, float).reshape(-1, 2)
    return count_violations(hyp.cell.astype(np.int64), lo.astype(float), hi.astype(float), hyp.t0.astype(float),
                            rthr.astype(float), np.ascontiguousarray(hyp.ids, np.int64),
                            np.ascontiguousarray(P.track[hyp.ids], np.int64), np.ascontiguousarray(P.t_arr[hyp.ids]),
                            P.cell_start, P.track, P.t_arr, P.r,
                            np.ascontiguousarray(heard[:, 0]), np.ascontiguousarray(heard[:, 1]), 90.0)


@dataclass
class _Ev:
    t: np.ndarray
    f: np.ndarray
    w: np.ndarray
    logf: np.ndarray


class DopplerScorer:
    """Weighted RMS (Hz) between observed ridges and the exact predicted Doppler curves.

    Each event's rest frequency f0 is solved in closed form (log-domain weighted mean), so only
    location and clock offset are searched. Residuals are truncated at `miss_hz`."""

    def __init__(self, flyovers, tracks, c, wind=None, miss_hz=3.0, max_points=None):
        self.tracks, self.c, self.wind, self.miss2 = tracks, c, wind, miss_hz**2
        self.events = []
        for fl in flyovers:
            sel = np.arange(len(fl.t))
            if max_points and len(sel) > max_points:
                sel = np.unique(np.linspace(0, len(sel) - 1, max_points).astype(int))
            self.events.append(_Ev(fl.t[sel], fl.f[sel], fl.w[sel], np.log(fl.f[sel])))

    def sse(self, obs, tids, t0, offsets, c=None, wind=None):
        offsets = np.atleast_1d(np.asarray(offsets, float))
        c = self.c if c is None else c
        wind = self.wind if wind is None else wind
        total = np.zeros(len(offsets))
        wsum = 0.0
        f0s = []
        for ev, ti in zip(self.events, tids):
            wsum += ev.w.sum()
            cur = doppler_curve(self.tracks[ti], obs, t0 + offsets.min() + ev.t[0],
                                t0 + offsets.max() + ev.t[-1], c, wind)
            if cur is None:
                total += ev.w.sum() * self.miss2
                f0s.append(np.full(len(offsets), np.nan))
                continue
            t_arr, g, _ = cur
            tq = (t0 + offsets)[:, None] + ev.t[None, :]
            gq = np.interp(tq.ravel(), t_arr, g, left=np.nan, right=np.nan).reshape(tq.shape)
            valid = np.isfinite(gq)
            gv = np.where(valid, gq, 1.0)
            wv = ev.w[None, :] * valid
            logf0 = np.sum(wv * (ev.logf[None, :] - np.log(gv)), 1) / np.maximum(wv.sum(1), 1e-9)
            res = ev.f[None, :] - np.exp(logf0)[:, None] * gv
            r2 = np.where(valid, np.minimum(res * res, self.miss2), self.miss2)
            total += np.sum(ev.w[None, :] * r2, 1)
            f0s.append(np.exp(logf0))
        return total, wsum, f0s

    def best_offset(self, obs, tids, t0, tol, n=21, c=None, wind=None):
        offs = np.linspace(-tol, tol, n)
        s, w, _ = self.sse(obs, tids, t0, offs, c, wind)
        i = int(np.argmin(s))
        step = offs[1] - offs[0] if n > 1 else 0.0
        if step > 0:
            offs2 = offs[i] + np.linspace(-step, step, 9)
            s2, w, _ = self.sse(obs, tids, t0, offs2, c, wind)
            j = int(np.argmin(s2))
            return float(np.sqrt(s2[j] / w)), t0 + offs2[j], float(s2[j]), w
        return float(np.sqrt(s[i] / w)), t0 + offs[i], float(s[i]), w


@dataclass
class LocateResult:
    best: dict
    candidates: list
    funnel: list
    stage_cells: dict
    cell_score: np.ndarray
    surface: dict
    grid: Grid
    events: list
    tracks: list
    n_passes: int
    timings: dict = field(default_factory=dict)


def _candidate(frame, terrain, e, n, t0, rms, tids, tracks, f0s, viol=0):
    up, lat, lon = frame.ground_up(np.array([e]), np.array([n]), terrain)
    return dict(e=float(e), n=float(n), up=float(up[0]), lat=float(lat[0]), lon=float(lon[0]), t0=float(t0),
                rms_hz=float(rms), track_ids=[int(t) for t in tids], aircraft=[tracks[t].label for t in tids],
                types=[tracks[t].type_code for t in tids],
                f0_hz=[float(np.ravel(f)[0]) if f is not None else float("nan") for f in f0s],
                violations=int(viol))


def locate(flyovers, tracks, frame: LocalFrame, terrain, rec: Recording, cfg: SearchConfig | None = None,
           c: float = 340.0, wind=None, heard=(), log=print) -> LocateResult:
    """Run the full search. `flyovers` are fitted Flyover objects (quality 'ok' recommended);
    `heard` are (start, end) recording-relative spans of every tone ridge detected (any quality)."""
    cfg = cfg or SearchConfig()
    tm = {}
    t_start = _time.perf_counter()
    events_all = sorted(flyovers, key=lambda f: f.t_arr)
    if not events_all:
        raise ValueError("no flyovers to search with")

    grid = build_grid(frame, terrain, cfg)
    t_lo = rec.start_guess - rec.start_unc - 120
    t_hi = rec.start_guess + rec.start_unc + rec.duration + 60
    wtracks = [w for w in (tr.window(t_lo, t_hi) for tr in tracks) if w is not None]
    log(f"[grid]    {grid.shape[1]} x {grid.shape[0]} = {grid.size:,} cells at {cfg.grid_res_m:.0f} m; "
        f"{len(wtracks)} prop-aircraft track segments in the time window")
    P = find_all_passes(grid, wtracks, c, cfg.max_range_m)
    tm["passes"] = _time.perf_counter() - t_start
    log(f"[passes]  {len(P):,} closest approaches within {cfg.max_range_m / 1000:.0f} km "
        f"({tm['passes']:.1f} s)")

    heard = list(heard) + [(f.t.min() - 20, f.t.max() + 20) for f in events_all]

    # Try all events together; if nothing survives, drop events (unequipped aircraft, bad fits).
    chosen = None
    m = len(events_all)
    for size in range(m, max(cfg.min_events, 1) - 1, -1):
        trials = []
        for subset in itertools.combinations(range(m), size):
            # closest (most constraining) flyover first: the funnel narrows fastest that way
            evs = sorted((events_all[i] for i in subset), key=lambda e: e.d)
            stages = []
            hyp = match_events(P, evs, rec, cfg, stages)
            viol = silence_violations(P, hyp, evs, rec, cfg, heard)
            hyp_s = hyp.take(viol <= cfg.max_violations)
            stages.append(("silence constraint: no unheard aircraft within the matched range", "silence", hyp_s))
            if len(hyp_s):
                trials.append((subset, evs, hyp_s, stages))
        if trials:
            chosen = trials
            break
        log(f"[match]   no location explains all {size} flyover(s) together; dropping one")
    if chosen is None:
        raise RuntimeError("no candidate location survived matching + silence; widen tolerances or the clock window")

    best_overall = None
    for subset, evs, hyp, stages in chosen:
        res = _score_subset(P, hyp, evs, wtracks, grid, frame, terrain, cfg, c, wind, log)
        res["events"], res["stages"] = evs, stages
        if best_overall is None or res["candidates"][0]["rms_hz"] < best_overall["candidates"][0]["rms_hz"]:
            best_overall = res
    r = best_overall
    evs = r["events"]  # same order as every candidate's track_ids

    funnel = [("all grid cells", "grid", grid.size, None)]
    stage_cells = {}
    for label, short, h in r["stages"]:
        cells = np.unique(h.cell)
        funnel.append((label, short, len(cells), len(h)))
        stage_cells[short] = cells
    funnel.append(("Doppler likelihood maximum", "Doppler", 1, 1))
    for label, _, cells, nh in funnel:
        log(f"[funnel]  {cells:>9,} cells  {'' if nh is None else f'{nh:>9,} hypotheses'}  {label}")

    best = r["candidates"][0]
    rivals = [cd for cd in r["candidates"][1:]
              if cd["rms_hz"] < 1.1 * best["rms_hz"] and np.hypot(cd["e"] - best["e"], cd["n"] - best["n"]) > 500]
    best["ambiguous"] = bool(rivals)
    best["low_confidence"] = bool(best["rms_hz"] > 0.8)
    if rivals:
        d = np.hypot(rivals[0]["e"] - best["e"], rivals[0]["n"] - best["n"])
        log(f"[warning] AMBIGUOUS: a position {d / 1000:.1f} km away fits almost as well "
            f"({rivals[0]['rms_hz']:.3f} vs {best['rms_hz']:.3f} Hz). With an unknown clock offset, two flyovers "
            f"can admit a second solution; another flyover, a reliable clock or a smaller search area resolves it.")
    if best["low_confidence"]:
        log(f"[warning] LOW CONFIDENCE: best misfit {best['rms_hz']:.2f} Hz is high; a flyover may be mis-detected "
            f"(check fig2_spectrogram) or the matching aircraft may be missing from the ADS-B data.")
    t_s = _time.perf_counter()
    surface = error_surface(best, evs, wtracks, frame, terrain, cfg, c, wind)
    tm["surface"] = _time.perf_counter() - t_s
    tm["total"] = _time.perf_counter() - t_start
    return LocateResult(best, r["candidates"], funnel, stage_cells, r["cell_score"], surface, grid, evs,
                        wtracks, len(P), tm)


def _score_subset(P, hyp, evs, tracks, grid, frame, terrain, cfg, c, wind, log):
    if len(hyp) > cfg.max_scored:
        # quick pre-score from distance / speed / timing consistency
        pre = np.zeros(len(hyp))
        for k, ev in enumerate(evs):
            q = hyp.ids[:, k]
            pre += (np.log(P.r[q] / ev.d) / 0.3) ** 2 + (np.log(P.speed[q] / ev.v) / 0.1) ** 2
            pre += ((P.t_arr[q] - hyp.t0 - ev.t_arr) / (0.5 * cfg.time_tol_s)) ** 2
        hyp = hyp.take(np.argsort(pre)[:cfg.max_scored])

    coarse = DopplerScorer(evs, tracks, c, wind, cfg.miss_hz, max_points=100)
    obs = grid.cell_enu(hyp.cell)
    tids_all = P.track[hyp.ids]
    rms = np.empty(len(hyp))
    t0b = np.empty(len(hyp))
    t_s = _time.perf_counter()
    for h in range(len(hyp)):
        rms[h], t0b[h], _, _ = coarse.best_offset(obs[h], tids_all[h], hyp.t0[h], cfg.time_tol_s, n=21)
    log(f"[doppler] scored {len(hyp):,} hypotheses against the observed curves ({_time.perf_counter() - t_s:.1f} s)")

    cell_score = np.full(grid.size, np.nan)
    np.fmin.at(cell_score, hyp.cell, rms)
    cell_score = cell_score.reshape(grid.shape)

    full = DopplerScorer(evs, tracks, c, wind, cfg.miss_hz)
    order = np.argsort(rms)[:cfg.rescore_top]
    rescored = []
    for h in order:
        r2, t02, _, _ = full.best_offset(obs[h], tids_all[h], t0b[h], 2.0, n=9)
        rescored.append((r2, h, t02))
    rescored.sort()

    picked = []
    for r2, h, t02 in rescored:
        key = tuple(tids_all[h])
        if any(key == tuple(tids_all[p[1]]) and np.hypot(*(obs[h][:2] - obs[p[1]][:2])) < 1500 for p in picked):
            continue
        picked.append((r2, h, t02))
        if len(picked) >= cfg.refine_top:
            break

    w0 = np.zeros(3) if wind is None else np.asarray(wind, float)
    cands = []
    for r2, h, t02 in picked:
        tids = tids_all[h]

        def unpack(x):
            if not cfg.fit_atmosphere:
                return x[0], x[1], x[2], c, wind, 0.0
            dw = np.array([x[3], x[4], 0.0])
            ws2 = float(np.sum((w0[:2] + dw[:2]) ** 2))
            prior = 0.01 * (ws2 / cfg.wind_prior_ms**2 + (x[5] / cfg.c_prior_ms) ** 2)
            # hard walls: surface wind beyond ~25 m/s or c off by > 15 m/s (~25 C) is not plausible
            prior += 10.0 * (max(0.0, np.sqrt(ws2) - 25.0) ** 2 + max(0.0, abs(x[5]) - 15.0) ** 2)
            return x[0], x[1], x[2], c + x[5], w0 + dw, prior

        def obj(x, tids=tids):
            e, n, t0, cc, ww, prior = unpack(x)
            up, _, _ = frame.ground_up(np.array([e]), np.array([n]), terrain)
            s, w, _ = full.sse(np.array([e, n, up[0]]), tids, t0, 0.0, cc, ww)
            return float(np.sqrt(s[0] / w + prior))

        e0, n0 = obs[h][0], obs[h][1]
        x0 = [e0, n0, t02] + ([0.0, 0.0, 0.0] if cfg.fit_atmosphere else [])
        steps = [150.0, 150.0, 1.5] + ([4.0, 4.0, 3.0] if cfg.fit_atmosphere else [])
        simplex = np.vstack([x0] + [np.array(x0) + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
        opt = minimize(obj, simplex[0], method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, xatol=0.3, fatol=1e-7, maxiter=400 * len(x0)))
        e, n, t0, cc, ww, _ = unpack(opt.x)
        up, _, _ = frame.ground_up(np.array([e]), np.array([n]), terrain)
        s, w, f0s = full.sse(np.array([e, n, up[0]]), tids, t0, 0.0, cc, ww)
        cand = _candidate(frame, terrain, e, n, t0, float(np.sqrt(s[0] / w)), tids, tracks, f0s)
        cand["c"] = float(cc)
        cand["wind_enu"] = None if ww is None else [float(v) for v in ww]
        cands.append(cand)
    cands.sort(key=lambda d: d["rms_hz"])
    log(f"[refine]  best {cands[0]['rms_hz']:.3f} Hz RMS; runner-up "
        + (f"{cands[1]['rms_hz']:.3f} Hz ({cands[1]['rms_hz'] / max(cands[0]['rms_hz'], 1e-9):.1f}x worse)"
           if len(cands) > 1 else "none"))
    return dict(candidates=cands, cell_score=cell_score)


def error_surface(best, evs, tracks, frame, terrain, cfg, c, wind):
    """RMS misfit on a fine grid around the best fit, plus approximate 68 / 95 % regions."""
    c = best.get("c", c)
    wind = np.asarray(best["wind_enu"]) if best.get("wind_enu") is not None else wind
    scorer = DopplerScorer(evs, tracks, c, wind, cfg.miss_hz, max_points=200)

    def evaluate(half, res):
        e = best["e"] + np.arange(-half, half + res / 2, res)
        n = best["n"] + np.arange(-half, half + res / 2, res)
        E, N = np.meshgrid(e, n)
        up, lat, lon = frame.ground_up(E, N, terrain)
        sse = np.empty(E.shape)
        wsum = 1.0
        for idx in np.ndindex(E.shape):
            _, _, sse[idx], wsum = scorer.best_offset(np.array([E[idx], N[idx], up[idx]]), best["track_ids"],
                                                      best["t0"], 3.0, n=7)
        return dict(e=e, n=n, lat=lat, lon=lon, sse=sse, res=res), wsum

    coarse, wsum = evaluate(cfg.surface_half_m, cfg.surface_res_m)
    k = 2 + 1 + len(evs)

    def finish(s, smin):
        # residuals of neighbouring spectrogram frames are correlated: count window/hop frames as one sample
        sigma2 = smin / max(wsum - k * cfg.oversample, 1.0)
        s["dchi2"] = (s["sse"] - smin) / sigma2 / cfg.oversample
        s["rms"] = np.sqrt(s["sse"] / wsum)
        return s

    smin = float(coarse["sse"].min())
    finish(coarse, smin)
    r95 = np.sqrt((coarse["dchi2"] <= 6.18).sum() * cfg.surface_res_m**2 / np.pi)
    half = float(max(150.0, 6 * r95))
    fine, _ = evaluate(half, half / 40)
    smin = min(smin, float(fine["sse"].min()))
    finish(coarse, smin)
    finish(fine, smin)
    a68 = float((fine["dchi2"] <= 2.30).sum() * fine["res"] ** 2)
    a95 = float((fine["dchi2"] <= 6.18).sum() * fine["res"] ** 2)
    return dict(coarse, fine=fine, n_eff=max(wsum / cfg.oversample, k + 1),
                radius68_m=float(np.sqrt(a68 / np.pi)), radius95_m=float(np.sqrt(a95 / np.pi)))
