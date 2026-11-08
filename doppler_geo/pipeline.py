"""End-to-end orchestration: audio analysis, ADS-B preparation, search, outputs, and the synthetic demo."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adsb import filter_tonal, flights_from_table
from .fit import fit_groups
from .geo import LocalFrame, haversine_m
from .physics import speed_of_sound
from .search import Recording, SearchConfig, locate
from .spectral import compute_spectrogram, group_harmonics, resample, track_ridges
from .trajectory import smooth_flights


@dataclass
class AnalysisConfig:
    fs: float = 1000.0
    win_s: float = 2.0
    hop_s: float = 0.5
    fmin: float = 30.0
    fmax: float = 400.0
    seed_db: float = 11.0       # a ridge must contain at least one pixel this far above the noise
    keep_db: float = 6.5        # ...and is followed until its running contrast falls below this
    min_dur_s: float = 20.0
    min_median_db: float = 7.0


@dataclass
class Analysis:
    spec: object
    ridges: list
    groups: list
    flyovers: list
    c: float

    @property
    def good(self):
        return [f for f in self.flyovers if f.quality == "ok"]

    @property
    def heard(self):
        """Recording-relative spans of every detected tone (used to excuse planes in the silence check)."""
        return [(g.t0 - 20, g.t1 + 20) for g in self.groups]


def analyze_audio(x, fs, c: float, cfg: AnalysisConfig | None = None, log=print) -> Analysis:
    cfg = cfg or AnalysisConfig()
    x = resample(x, fs, cfg.fs)
    spec = compute_spectrogram(x, cfg.fs, cfg.win_s, cfg.hop_s, cfg.fmin, cfg.fmax)
    ridges = track_ridges(spec, seed_db=cfg.seed_db, keep_db=cfg.keep_db, min_dur_s=cfg.min_dur_s,
                          min_median_db=cfg.min_median_db)
    groups = group_harmonics(ridges)
    flyovers = fit_groups(groups, c, oversample=cfg.win_s / cfg.hop_s)
    log(f"[audio]   {len(x) / cfg.fs:.0f} s of audio -> {len(ridges)} tone ridges -> {len(groups)} sources "
        f"-> {sum(f.quality == 'ok' for f in flyovers)} complete flyovers")
    for f in flyovers:
        log(f"[flyover] {f.summary()}  harmonics {f.harmonics}")
    return Analysis(spec, ridges, groups, flyovers, c)


def prepare_tracks(adsb_df, frame: LocalFrame, log=print):
    df = filter_tonal(adsb_df)
    flights = flights_from_table(df)
    tracks = smooth_flights(flights, frame)
    log(f"[adsb]    {adsb_df['icao24'].nunique()} aircraft in the feed, {len(flights)} propeller/unknown types "
        f"-> {len(tracks)} smoothed track segments")
    return tracks


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def atmosphere_of(cand: dict) -> dict:
    """Wind (speed, direction it blows FROM) and air temperature implied by a candidate's fitted c."""
    out = {}
    if cand.get("c"):
        out["speed_of_sound_ms"] = cand["c"]
        out["temp_c"] = 273.15 * ((cand["c"] / 331.3) ** 2 - 1)
    if cand.get("wind_enu") is not None:
        we, wn = cand["wind_enu"][:2]
        out["wind_ms"] = float(np.hypot(we, wn))
        out["wind_from_deg"] = float(np.degrees(np.arctan2(-we, -wn)) % 360)
    return out


def summarize(result, analysis: Analysis, rec: Recording, truth: dict | None = None) -> dict:
    best = result.best
    out = dict(
        estimate=dict(lat=best["lat"], lon=best["lon"], rms_hz=best["rms_hz"], atmosphere=atmosphere_of(best),
                      ambiguous=best.get("ambiguous", False), low_confidence=best.get("low_confidence", False),
                      clock_offset_s=best["t0"] - rec.start_guess,
                      recording_start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(best["t0"])),
                      matched_aircraft=best["aircraft"], types=best["types"], rest_freq_hz=best["f0_hz"],
                      radius68_m=result.surface["radius68_m"], radius95_m=result.surface["radius95_m"]),
        runner_ups=[dict(lat=c["lat"], lon=c["lon"], rms_hz=c["rms_hz"], aircraft=c["aircraft"])
                    for c in result.candidates[1:]],
        flyovers=[dict(t_arrival_s=f.t_arr, speed_ms=f.v, distance_m=f.d, rest_freq_hz=f.f0, rms_hz=f.rms_hz,
                       quality=f.quality, sigma=f.sigma) for f in analysis.flyovers],
        funnel=[dict(stage=s, short=k, cells=c, hypotheses=h) for s, k, c, h in result.funnel],
        n_passes=result.n_passes, timings_s=result.timings)
    if truth:
        err = float(haversine_m(best["lat"], best["lon"], truth["lat"], truth["lon"]))
        out["truth"] = truth
        out["error_m"] = err
    return _jsonable(out)


def run_demo(out_dir="out/demo", seed: int = 7, clock_error_s: float | None = None, clock_unc_s: float = 900.0,
             analyst_temp_c: float = 10.0, use_true_wind: bool = False, figures: bool = True,
             cfg: SearchConfig | None = None, log=print) -> dict:
    """Synthetic end-to-end run: generate a scenario, write its recording, ADS-B and ground truth, then
    locate the receiver using only the recording, the ADS-B and an approximate start time."""
    import soundfile as sf

    from .synth import make_world, render_audio, synth_adsb

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 99)
    t_all = time.perf_counter()

    log(f"[world]   generating synthetic scenario (seed {seed}) ...")
    w = make_world(seed)
    audio = render_audio(w, seed=seed)
    adsb = synth_adsb(w, seed=seed)
    sf.write(out / "recording.wav", audio, int(w.fs))
    adsb.to_parquet(out / "adsb.parquet", index=False)
    w.terrain.save_npz(out / "terrain.npz")
    truth = dict(lat=w.obs_lat, lon=w.obs_lon, height_m=w.obs_h, recording_start=w.rec_start,
                 temp_c=w.temp_c, wind_enu=w.wind.tolist(), true_passes=w.passes)
    (out / "truth.json").write_text(json.dumps(_jsonable(truth), indent=2))
    log(f"[world]   true receiver position {w.obs_lat:.5f}, {w.obs_lon:.5f}; "
        f"{len(w.passes)} prop flyovers during the {w.rec_duration / 60:.0f}-min recording")
    for p in w.passes:
        log(f"[truth]   {p['callsign']} ({p['type']}) at t={p['t_rel']:.0f}s, slant {p['slant_m']:.0f} m, "
            f"{p['speed_ms']:.0f} m/s, BPF {p['bpf_hz']:.1f} Hz")

    # ---- from here on, only the inputs available in practice ----
    c = speed_of_sound(analyst_temp_c)
    clock_error = rng.uniform(-0.6, 0.6) * clock_unc_s if clock_error_s is None else clock_error_s
    rec = Recording(w.rec_start + clock_error, clock_unc_s, w.rec_duration)
    log(f"[analyst] assumed {analyst_temp_c:.0f} C (c = {c:.1f} m/s; truth {w.c:.1f}), "
        f"{'true' if use_true_wind else 'no'} wind, clock off by {clock_error:+.0f} s (searching +-{clock_unc_s:.0f} s)")
    analysis = analyze_audio(audio, w.fs, c, log=log)
    tracks = prepare_tracks(adsb, w.frame, log=log)
    result = locate(analysis.good, tracks, w.frame, w.terrain, rec, cfg, c,
                    wind=w.wind if use_true_wind else None, heard=analysis.heard, log=log)
    summary = summarize(result, analysis, rec, dict(lat=w.obs_lat, lon=w.obs_lon))
    summary["clock_error_true_s"] = clock_error
    summary["clock_offset_found_s"] = result.best["t0"] - w.rec_start
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    log(f"[result]  estimate {result.best['lat']:.5f}, {result.best['lon']:.5f}  ->  "
        f"{summary['error_m']:.0f} m from the true position "
        f"(clock recovered to {summary['clock_offset_found_s']:+.1f} s)")
    atm = summary["estimate"]["atmosphere"]
    if "wind_ms" in atm:
        tw = np.hypot(*w.wind[:2])
        tf = np.degrees(np.arctan2(-w.wind[0], -w.wind[1])) % 360
        log(f"[result]  fitted atmosphere: wind {atm['wind_ms']:.1f} m/s from {atm['wind_from_deg']:.0f} deg, "
            f"{atm['temp_c']:.1f} C   (truth: {tw:.1f} m/s from {tf:.0f} deg, {w.temp_c:.1f} C)")

    if figures:
        from . import viz

        viz.make_all(out, analysis, result, w.frame, c,
                     truth=dict(lat=w.obs_lat, lon=w.obs_lon, e=w.observer[0], n=w.observer[1]), log=log)
    log(f"[done]    {time.perf_counter() - t_all:.0f} s total; outputs in {out.resolve()}")
    return summary


def run_locate(audio_path, adsb_path, center_lat, center_lon, start_utc: float, start_unc: float,
               out_dir="out/run", temp_c: float = 15.0, wind=None, terrain=None, cfg: SearchConfig | None = None,
               analysis_cfg: AnalysisConfig | None = None, figures: bool = True, log=print) -> dict:
    """Real-data run: an audio file, an ADS-B table, a search centre and an approximate start time."""
    from .adsb import load_table
    from .geo import FlatTerrain
    from .spectral import load_audio

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = LocalFrame(center_lat, center_lon)
    terrain = terrain or FlatTerrain(0.0)
    c = speed_of_sound(temp_c)
    x, fs = load_audio(audio_path, (analysis_cfg or AnalysisConfig()).fs)
    analysis = analyze_audio(x, fs, c, analysis_cfg, log=log)
    if not analysis.good:
        raise RuntimeError("no complete flyover found in the audio (see the spectrogram figure)")
    tracks = prepare_tracks(load_table(adsb_path), frame, log=log)
    rec = Recording(start_utc, start_unc, len(x) / fs)
    result = locate(analysis.good, tracks, frame, terrain, rec, cfg, c, wind=wind, heard=analysis.heard, log=log)
    summary = summarize(result, analysis, rec)
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    log(f"[result]  estimate {result.best['lat']:.5f}, {result.best['lon']:.5f}  "
        f"(+-{result.surface['radius68_m']:.0f} m statistical, 68 %)")
    if figures:
        from . import viz

        viz.make_all(out, analysis, result, frame, c, wind=wind, log=log)
    return summary
