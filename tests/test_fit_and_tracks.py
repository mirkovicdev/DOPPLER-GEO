import gzip
import json

import numpy as np

from doppler_geo.adsb import classify_type, parse_readsb_trace
from doppler_geo.fit import fit_flyover
from doppler_geo.geo import LocalFrame
from doppler_geo.physics import flyby_frequency
from doppler_geo.trajectory import Flight, smooth_flight


def test_fit_recovers_speed_distance_and_time():
    rng = np.random.default_rng(0)
    c = 338.0
    t = np.arange(0, 240, 0.5)
    f = flyby_frequency(t, 84.0, 55.0, 1400.0, 115.0, c) + rng.normal(0, 0.15, t.size)
    fl = fit_flyover(t, f, c=c)
    assert fl.quality == "ok"
    assert abs(fl.v - 55.0) / 55.0 < 0.03
    assert abs(fl.d - 1400.0) / 1400.0 < 0.05
    assert abs(fl.t_arr - (115.0 + 1400.0 / c)) < 1.0
    assert abs(fl.f0 - 84.0) < 0.2


def test_fit_flags_partial_flyover():
    c = 338.0
    t = np.arange(0, 100, 0.5)  # ends before the closest approach (heard at ~154 s)
    fl = fit_flyover(t, flyby_frequency(t, 84.0, 55.0, 1400.0, 150.0, c), c=c)
    assert fl.quality != "ok"


def test_smoother_fuses_noisy_positions_with_velocity():
    rng = np.random.default_rng(1)
    fr = LocalFrame(38.0, -98.0)
    t = np.cumsum(rng.uniform(0.5, 1.5, 600))
    e, n = 60.0 * t * np.sin(np.radians(30)), 60.0 * t * np.cos(np.radians(30))
    lat, lon, h = fr.to_geodetic(np.stack([e + rng.normal(0, 15, t.size), n + rng.normal(0, 15, t.size),
                                           np.full(t.size, 1500.0)], -1))
    fl = Flight("abc123", t, lat, lon, h, gs=np.full(t.size, 60.0) + rng.normal(0, 0.3, t.size),
                track=np.full(t.size, 30.0), vrate=np.zeros(t.size))
    (tr,) = smooth_flight(fl, fr)
    pos, vel, ok = tr.state(t[50:-50])
    truth = np.stack([e[50:-50], n[50:-50]], -1)
    assert np.sqrt(np.mean(np.sum((pos[:, :2] - truth) ** 2, 1))) < 8.0
    assert np.max(np.abs(np.linalg.norm(vel[:, :2], axis=1) - 60.0)) < 0.6


def test_readsb_trace_parsing_and_types():
    trace = {"icao": "a1b2c3", "t": "C172", "r": "N12345", "timestamp": 1_700_000_000.0,
             "trace": [[0.0, 38.1, -98.1, 3000, 100.0, 90.0, 0, 0, {"flight": "N12345  "}, "adsb_icao", 3100, 0],
                       [1.0, 38.1, -98.099, 3000, 100.0, 90.0, 0, 0, None, "adsb_icao", 3100, 0],
                       [2.0, 38.1, -98.098, "ground", 0, 90.0, 0, 0, None, "adsb_icao", None, 0]]}
    df = parse_readsb_trace(gzip.compress(json.dumps(trace).encode()))
    assert len(df) == 2  # the on-ground report is dropped
    assert df["callsign"].iloc[0] == "N12345"
    assert abs(df["alt_m"].iloc[0] - 3100 * 0.3048) < 1e-6  # geometric altitude preferred
    assert abs(df["gs_ms"].iloc[0] - 51.4444) < 1e-3
    assert classify_type("C172") == "P" and classify_type("PC12") == "T" and classify_type("B738") == "J"
    assert classify_type("ZZZZ", "L2T") == "T" and classify_type("XXXX") == "?"
