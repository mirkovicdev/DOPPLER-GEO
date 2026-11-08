import numpy as np
from scipy.signal import hilbert

from doppler_geo.geo import FlatTerrain, LocalFrame
from doppler_geo.physics import (emission_to_arrival, flyby_emission_time, flyby_frequency, propagation_delay,
                                 speed_of_sound, wind_vector)


class StraightTrack:
    def __init__(self, v, d, h, t_ca):
        self.v, self.d, self.h, self.t_ca = v, d, h, t_ca

    def state(self, tau):
        tau = np.asarray(tau, float)
        u = tau - self.t_ca
        y_off = np.sqrt(self.d**2 - self.h**2)
        pos = np.stack([self.v * u, np.full_like(u, y_off), np.full_like(u, self.h)], -1)
        vel = np.tile([self.v, 0.0, 0.0], (len(u), 1))
        return pos, vel, np.ones(len(u), bool)


def test_retarded_time_identity():
    c, v, d, t_ca = 340.0, 60.0, 900.0, 100.0
    t = np.linspace(-200, 400, 101)
    tau = flyby_emission_time(t, t_ca, v, d, c)
    r = np.sqrt(d**2 + (v * (tau - t_ca)) ** 2)
    assert np.allclose(tau + r / c, t, atol=1e-9)
    assert np.all(tau < t)


def test_closed_form_matches_general_model():
    c = 338.0
    tr = StraightTrack(v=70.0, d=1200.0, h=800.0, t_ca=50.0)
    tau = np.linspace(-150, 250, 2001)
    pos, vel, _ = tr.state(tau)
    t_arr, g, _ = emission_to_arrival(tau, pos, vel, np.zeros(3), c)
    f_closed = flyby_frequency(t_arr, 100.0, 70.0, 1200.0, 50.0, c)
    assert np.allclose(100.0 * g, f_closed, atol=1e-9)


def test_wind_delay_and_still_air_limit():
    src = np.array([[1000.0, 500.0, 300.0]])
    obs = np.zeros(3)
    c = 340.0
    assert np.allclose(propagation_delay(src, obs, c, np.zeros(3)), propagation_delay(src, obs, c))
    w = wind_vector(10.0, 45.0)  # from the NE, blowing toward SW
    T = propagation_delay(src, obs, c, w)
    # the wavefront leaves src and is carried by the air: |D - w T| = c T
    D = obs - src
    assert np.allclose(np.linalg.norm(D - T[:, None] * w, axis=1), c * T)


def test_rendered_tone_has_predicted_instantaneous_frequency():
    fs, c, f0 = 2000.0, speed_of_sound(10), 90.0
    tr = StraightTrack(v=60.0, d=700.0, h=500.0, t_ca=30.0)
    tau = np.arange(-60, 120, 0.01)
    pos, vel, _ = tr.state(tau)
    t_arr, g, _ = emission_to_arrival(tau, pos, vel, np.zeros(3), c)
    t = np.arange(0, 60, 1 / fs)
    x = np.sin(2 * np.pi * f0 * np.interp(t, t_arr, tau))   # received phase = emitted phase at tau(t)
    inst = np.gradient(np.unwrap(np.angle(hilbert(x))), t) / (2 * np.pi)
    mid = slice(int(5 * fs), int(55 * fs))
    assert np.max(np.abs(inst[mid] - f0 * np.interp(t[mid], t_arr, g))) < 0.05


def test_frame_roundtrip_and_ground_height():
    fr = LocalFrame(38.0, -98.0)
    lat, lon, h = 38.3, -97.6, 512.0
    enu = fr.to_enu(lat, lon, h)
    la2, lo2, h2 = fr.to_geodetic(enu)
    assert abs(la2 - lat) < 1e-9 and abs(lo2 - lon) < 1e-9 and abs(h2 - h) < 1e-4
    up, la3, lo3 = fr.ground_up(np.array([30000.0]), np.array([-20000.0]), FlatTerrain(400.0))
    _, _, h3 = fr.to_geodetic(np.array([30000.0, -20000.0, up[0]]))
    assert abs(h3 - 400.0) < 1e-3
    assert up[0] < 400.0 - 50  # Earth curvature: 36 km away the ground drops ~100 m below the tangent plane
