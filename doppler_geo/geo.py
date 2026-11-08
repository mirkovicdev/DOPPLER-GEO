"""Geodesy: WGS84 geodetic <-> ECEF <-> one local East-North-Up (ENU) frame, plus terrain.

Every distance in the pipeline is computed in a single ENU frame centred on the search
area. ENU is a rotation + translation of ECEF, so 3D distances are exact and Earth
curvature appears naturally (a ground point 50 km away sits ~200 m "below" the frame).

Heights are ellipsoidal (WGS84, "HAE"), which is what ADS-B geometric altitude reports.
DEMs are usually geoid/MSL heights: add the local geoid undulation (about -30 m over most
of the US) via ``geoid_offset_m``.
"""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(lat)
    lon = np.radians(lon)
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * s
    return x, y, z


def ecef_to_geodetic(x, y, z):
    """Bowring's closed form; sub-millimetre for points near the Earth's surface."""
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    th = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * np.sin(th) ** 3,
                     p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def enu_basis(lat, lon):
    """Rows are the unit East, North, Up vectors (expressed in ECEF) at lat/lon. Shape (..., 3, 3)."""
    la, lo = np.radians(lat), np.radians(lon)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    z = np.zeros_like(sl)
    e = np.stack([-so, co, z], -1)
    n = np.stack([-sl * co, -sl * so, cl], -1)
    u = np.stack([cl * co, cl * so, sl], -1)
    return np.stack([e, n, u], -2)


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class LocalFrame:
    """Local tangent-plane ENU frame anchored at (lat0, lon0, h0)."""

    def __init__(self, lat0: float, lon0: float, h0: float = 0.0):
        self.lat0, self.lon0, self.h0 = float(lat0), float(lon0), float(h0)
        self.origin = np.array(geodetic_to_ecef(self.lat0, self.lon0, self.h0))
        self.rot = enu_basis(self.lat0, self.lon0)  # rows: e, n, u

    def to_enu(self, lat, lon, h):
        x, y, z = geodetic_to_ecef(np.asarray(lat, float), np.asarray(lon, float), np.asarray(h, float))
        return (np.stack([x, y, z], -1) - self.origin) @ self.rot.T

    def to_geodetic(self, enu):
        ecef = np.asarray(enu, float) @ self.rot + self.origin
        return ecef_to_geodetic(ecef[..., 0], ecef[..., 1], ecef[..., 2])

    def vector_to_enu(self, lat, lon, v_local):
        """Rotate vectors given in the local ENU at (lat, lon) into this frame."""
        v_ecef = np.einsum("...ij,...i->...j", enu_basis(lat, lon), np.asarray(v_local, float))
        return v_ecef @ self.rot.T

    def vector_from_enu(self, lat, lon, v):
        """Inverse of vector_to_enu."""
        v_ecef = np.asarray(v, float) @ self.rot
        return np.einsum("...ij,...j->...i", enu_basis(lat, lon), v_ecef)

    def up_for_height(self, east, north, height_fn, iters: int = 3):
        """Find the ENU 'up' that puts tangent-plane point (east, north) at ellipsoidal height
        height_fn(lat, lon). Returns (up, lat, lon)."""
        east, north = np.broadcast_arrays(np.asarray(east, float), np.asarray(north, float))
        up = np.zeros(east.shape)
        for _ in range(iters):
            lat, lon, h = self.to_geodetic(np.stack([east, north, up], -1))
            up = up + (height_fn(lat, lon) - h)
        lat, lon, _ = self.to_geodetic(np.stack([east, north, up], -1))
        return up, lat, lon

    def ground_up(self, east, north, terrain, iters: int = 3):
        return self.up_for_height(east, north, terrain.elevation, iters)


class FlatTerrain:
    """Constant ground height (ellipsoidal metres)."""

    def __init__(self, height_m: float = 0.0):
        self.height_m = float(height_m)

    def elevation(self, lat, lon):
        return np.full(np.shape(lat), self.height_m)


class GridTerrain:
    """Regular lat/lon elevation grid (e.g. SRTM / Copernicus DEM), bilinear interpolation."""

    def __init__(self, lats, lons, heights, geoid_offset_m: float = 0.0):
        from scipy.interpolate import RegularGridInterpolator

        lats, lons, heights = np.asarray(lats, float), np.asarray(lons, float), np.asarray(heights, float)
        if lats[0] > lats[-1]:
            lats, heights = lats[::-1], heights[::-1]
        if lons[0] > lons[-1]:
            lons, heights = lons[::-1], heights[:, ::-1]
        self.lats, self.lons, self.heights = lats, lons, heights
        self.geoid_offset_m = float(geoid_offset_m)
        self._interp = RegularGridInterpolator((lats, lons), heights, bounds_error=False, fill_value=None)

    def elevation(self, lat, lon):
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        pts = np.stack([lat.ravel(), lon.ravel()], -1)
        return self._interp(pts).reshape(lat.shape) + self.geoid_offset_m

    def save_npz(self, path):
        np.savez_compressed(path, lats=self.lats, lons=self.lons, heights=self.heights,
                            geoid_offset_m=self.geoid_offset_m)

    @classmethod
    def from_npz(cls, path, geoid_offset_m: float | None = None):
        d = np.load(path)
        off = float(d["geoid_offset_m"]) if geoid_offset_m is None and "geoid_offset_m" in d else (geoid_offset_m or 0.0)
        return cls(d["lats"], d["lons"], d["heights"], off)

    @classmethod
    def from_geotiff(cls, path, geoid_offset_m: float = 0.0):
        """Load a north-up lat/lon GeoTIFF DEM (needs `rasterio`)."""
        import rasterio  # optional dependency

        with rasterio.open(path) as ds:
            h = ds.read(1).astype(float)
            t = ds.transform
            lons = t.c + t.a * (np.arange(ds.width) + 0.5)
            lats = t.f + t.e * (np.arange(ds.height) + 0.5)
        return cls(lats, lons, h, geoid_offset_m)


def load_terrain(spec: str | float | None, geoid_offset_m: float = 0.0):
    """'none'/None -> flat 0 m, a number -> flat at that height, *.npz / *.tif -> DEM."""
    if spec is None or str(spec).lower() == "none":
        return FlatTerrain(0.0)
    try:
        return FlatTerrain(float(spec) + geoid_offset_m)
    except ValueError:
        pass
    s = str(spec)
    if s.endswith(".npz"):
        return GridTerrain.from_npz(s, geoid_offset_m or None)
    return GridTerrain.from_geotiff(s, geoid_offset_m)
