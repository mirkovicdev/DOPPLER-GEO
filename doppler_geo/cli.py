"""Command line: python -m doppler_geo {demo, analyze, locate, convert-readsb} ..."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _utc(s: str) -> float:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _wind(s: str | None):
    if not s:
        return None
    from .physics import wind_vector

    speed, frm = s.split("@")
    return wind_vector(float(speed), float(frm))


def cmd_demo(a):
    from .pipeline import run_demo
    from .search import SearchConfig

    run_demo(a.out, seed=a.seed, clock_unc_s=a.clock_unc, analyst_temp_c=a.temp, use_true_wind=a.true_wind,
             figures=not a.no_figures, cfg=SearchConfig(silence=a.silence))


def cmd_analyze(a):
    from . import viz
    from .physics import speed_of_sound
    from .pipeline import AnalysisConfig, analyze_audio
    from .spectral import load_audio

    cfg = AnalysisConfig(fmin=a.fmin, fmax=a.fmax)
    x, fs = load_audio(a.audio, cfg.fs)
    an = analyze_audio(x, fs, speed_of_sound(a.temp), cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    viz.save(viz.fig_spectrogram(an.spec, an.flyovers), out / "fig2_spectrogram")
    if an.good:
        viz.save(viz.fig_flyover_fits(an.flyovers), out / "fig3_flyover_fits")
    print(f"[figure]  {out / 'fig2_spectrogram.pdf'} (+ .png)")
    rows = [dict(t_arrival_s=f.t_arr, rest_freq_hz=f.f0, speed_ms=f.v, speed_kt=f.v * 1.944, distance_m=f.d,
                 rms_hz=f.rms_hz, quality=f.quality, sigma=f.sigma, harmonics=f.harmonics) for f in an.flyovers]
    (out / "flyovers.json").write_text(json.dumps(rows, indent=2))
    print(f"[done]    {out / 'flyovers.json'}")


def cmd_locate(a):
    from .geo import load_terrain
    from .pipeline import AnalysisConfig, run_locate
    from .search import SearchConfig

    lat, lon = (float(v) for v in a.center.split(","))
    cfg = SearchConfig(half_size_m=a.radius_km * 1000, grid_res_m=a.grid_m, silence=a.silence,
                       time_tol_s=a.time_tol, max_range_m=a.max_range_km * 1000)
    run_locate(a.audio, a.adsb, lat, lon, _utc(a.start), a.start_unc, out_dir=a.out, temp_c=a.temp,
               wind=_wind(a.wind), terrain=load_terrain(a.terrain, a.geoid_offset), cfg=cfg,
               analysis_cfg=AnalysisConfig(fmin=a.fmin, fmax=a.fmax), figures=not a.no_figures)


def cmd_convert(a):
    from .adsb import convert_readsb

    bbox = tuple(float(v) for v in a.bbox.split(","))
    df = convert_readsb(a.sources, bbox, _utc(a.start), _utc(a.end), tonal_only=not a.all_types,
                        baro_offset_m=a.baro_offset)
    df.to_parquet(a.out, index=False)
    print(f"[done]    {len(df):,} reports from {df['icao24'].nunique() if len(df) else 0} aircraft -> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="doppler_geo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic end-to-end demo with known ground truth")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", default="out/demo")
    d.add_argument("--clock-unc", type=float, default=900.0, help="+- seconds of recording-clock uncertainty")
    d.add_argument("--temp", type=float, default=10.0, help="air temperature assumed by the estimator (C)")
    d.add_argument("--true-wind", action="store_true", help="give the estimator the true wind")
    d.add_argument("--silence", default="window", choices=["window", "between", "off"])
    d.add_argument("--no-figures", action="store_true")
    d.set_defaults(fn=cmd_demo)

    an = sub.add_parser("analyze", help="spectrogram + flyover detection + Doppler fits for one audio file")
    an.add_argument("audio")
    an.add_argument("--temp", type=float, default=15.0)
    an.add_argument("--fmin", type=float, default=30.0)
    an.add_argument("--fmax", type=float, default=400.0)
    an.add_argument("--out", default="out/analyze")
    an.set_defaults(fn=cmd_analyze)

    lo = sub.add_parser("locate", help="full geolocation with a real ADS-B table")
    lo.add_argument("audio")
    lo.add_argument("--adsb", required=True, help="parquet/csv (see `convert-readsb`)")
    lo.add_argument("--center", required=True, help="search centre 'lat,lon'")
    lo.add_argument("--radius-km", type=float, default=30.0, help="half-width of the square search area")
    lo.add_argument("--grid-m", type=float, default=200.0)
    lo.add_argument("--start", required=True, help="approximate UTC time of the first audio sample (ISO 8601)")
    lo.add_argument("--start-unc", type=float, default=600.0, help="+- seconds")
    lo.add_argument("--temp", type=float, default=15.0, help="air temperature (C)")
    lo.add_argument("--wind", default=None, help="'speed_ms@from_deg', e.g. 6@240")
    lo.add_argument("--terrain", default="none", help="'none', a ground height in m, or a DEM (.npz/.tif)")
    lo.add_argument("--geoid-offset", type=float, default=0.0, help="geoid undulation added to DEM heights (m)")
    lo.add_argument("--silence", default="window", choices=["window", "between", "off"])
    lo.add_argument("--time-tol", type=float, default=15.0)
    lo.add_argument("--max-range-km", type=float, default=8.0)
    lo.add_argument("--fmin", type=float, default=30.0)
    lo.add_argument("--fmax", type=float, default=400.0)
    lo.add_argument("--out", default="out/run")
    lo.add_argument("--no-figures", action="store_true")
    lo.set_defaults(fn=cmd_locate)

    cv = sub.add_parser("convert-readsb", help="readsb trace files (adsb.lol globe_history) -> parquet")
    cv.add_argument("sources", nargs="+", help="directories, trace files, or (split) .tar archives")
    cv.add_argument("--bbox", required=True, help="lat_min,lat_max,lon_min,lon_max")
    cv.add_argument("--start", required=True)
    cv.add_argument("--end", required=True)
    cv.add_argument("--out", default="flights.parquet")
    cv.add_argument("--all-types", action="store_true", help="keep jets and helicopters too")
    cv.add_argument("--baro-offset", type=float, default=0.0, help="added to barometric altitude when no GNSS altitude")
    cv.set_defaults(fn=cmd_convert)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
