"""ADS-B input: tabular files, readsb trace files (adsb.lol / ADS-B Exchange globe_history), type filtering.

Canonical table columns (one row per position report):
    time (unix s, UTC) | icao24 | lat | lon | alt_m (ellipsoidal) | gs_ms | track_deg | vrate_ms
    | callsign | type_code

Altitude: prefer geometric (GNSS) altitude, which readsb reports relative to the WGS84
ellipsoid. Barometric altitude is pressure altitude: off by the QNH correction (up to
hundreds of feet) and by the geoid. Use `baro_offset_m` when only barometric altitude is available.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .trajectory import Flight

FT = 0.3048
KT = 0.514444
FPM = 0.00508

# ICAO type designators -> engine class: P piston prop, T turboprop, J jet, H helicopter.
# Not exhaustive; unknown types are kept by default.
_PROPS_P = """C150 C152 C162 C170 C172 C175 C177 C77R C180 C182 C82R C82S C185 C195 C205 C206 C207 C210 P210 T210
C303 C310 C320 C335 C337 C340 C402 C414 C421 PA11 PA12 PA18 PA20 PA22 PA23 PA24 PA25 PA27 PA28 P28A P28B P28R P28T
PA30 PA31 PA32 P32R P32T PA34 PA36 PA38 PA44 PA46 PA60 SR20 SR22 S22T BE19 BE23 BE24 BE33 BE35 BE36 BE50 BE55 BE58
BE60 BE76 BE77 BE95 M20P M20T M20J DA20 DA40 DA42 DA62 AA1 AA5 GA7 RV4 RV6 RV7 RV8 RV9 RV10 RV12 RV14 J3 T6 T34
CH7A CH7B BL8 AN2 GLAS LNC2 LNC4 COL3 COL4 EVOT C72R TOBA TB20 TB21 P68 M5 M7 G164 AT3T AT5T""".split()
_PROPS_T = """C208 C425 C441 BE9L BE9T BE10 BE20 BE30 B350 B190 BE99 TBM7 TBM8 TBM9 PC12 PC6T PC21 PC7 PC9 DHC2 DHC3
DHC6 DHC7 DH8A DH8B DH8C DH8D AT43 AT45 AT72 AT73 AT75 AT76 SF34 SB20 E110 E120 JS31 JS32 JS41 D228 D328 SW3 SW4
MU2 AC90 AC95 AC6L AEST AN24 AN26 AN28 C130 C30J L188 P180 PAY1 PAY2 PAY3 PAY4 KODI EPIC M600 P46T AT6T AT8T AT8
M18 T6 BN2T CN35 C295 L410 SH36 F27 F50 Y12""".split()
_JETS = """A318 A319 A320 A321 A19N A20N A21N A332 A333 A338 A339 A359 A35K A388 B712 B733 B734 B735 B736 B737
B738 B739 B37M B38M B39M B3XM B752 B753 B762 B763 B764 B772 B773 B77W B77L B778 B779 B788 B789 B78X B744 B748
CRJ1 CRJ2 CRJ7 CRJ9 CRJX E135 E145 E170 E75L E75S E190 E195 E290 E295 BCS1 BCS3 C25A C25B C25C C25M C500 C501
C510 C525 C550 C560 C56X C650 C680 C68A C700 C750 CL30 CL35 CL60 GLEX GL5T GL7T GLF4 GLF5 GLF6 G280 G150 F2TH F900
FA50 FA7X FA8X LJ31 LJ35 LJ40 LJ45 LJ55 LJ60 LJ75 E50P E55P E545 E550 HDJT PRM1 BE40 SF50 H25B MD11 MD82 MD83
MD87 MD88 MD90 DC10 C17 K35R E3TF F16 F18 F15 T38""".split()
_HELIS = """R22 R44 R66 EC20 EC25 EC30 EC35 EC45 EC55 EC75 AS32 AS50 AS55 AS65 B06 B407 B412 B429 B47G B505 A109 A119
A139 A169 A189 S76 S92 H60 H500 MD52 MD60 BK17 UH1 CH47""".split()

ENGINE_CLASS: dict[str, str] = {}
for _codes, _cls in ((_PROPS_P, "P"), (_PROPS_T, "T"), (_JETS, "J"), (_HELIS, "H")):
    for _c in _codes:
        ENGINE_CLASS.setdefault(_c, _cls)


def classify_type(type_code: str | None, short_type: str | None = None) -> str:
    """Engine class of an aircraft: 'P', 'T', 'J', 'H' or '?' (unknown).

    `short_type` is the ICAO description like 'L1P' / 'L2T' / 'H1T' (tar1090-db, ADSBx basic db).
    """
    if short_type and len(short_type) == 3:
        if short_type[0] in "HG":
            return "H"
        if short_type[2] in "PTJ":
            return short_type[2]
    code = (type_code or "").strip().upper()
    return ENGINE_CLASS.get(code, "?")


def filter_tonal(df: pd.DataFrame, classes=("P", "T"), keep_unknown: bool = True) -> pd.DataFrame:
    """Keep aircraft that produce a propeller tone (piston + turboprop by default)."""
    if "type_code" not in df.columns:
        return df
    short = df["short_type"] if "short_type" in df.columns else pd.Series([None] * len(df), index=df.index)
    cls = pd.Series([classify_type(t, s) for t, s in zip(df["type_code"], short)], index=df.index)
    ok = cls.isin(classes) | (keep_unknown & (cls == "?"))
    return df[ok]


_ALIASES = {
    "timestamp": "time", "t": "time", "hex": "icao24", "icao": "icao24", "latitude": "lat",
    "longitude": "lon", "geoaltitude": "alt_m", "alt_geom_m": "alt_m", "velocity": "gs_ms",
    "heading": "track_deg", "track": "track_deg", "vertrate": "vrate_ms", "flight": "callsign",
    "typecode": "type_code", "type": "type_code", "icaotype": "type_code",
}


def normalize_table(df: pd.DataFrame, baro_offset_m: float = 0.0) -> pd.DataFrame:
    """Rename common column spellings (OpenSky state vectors, readsb exports) to the canonical names."""
    df = df.rename(columns={k: v for k, v in _ALIASES.items() if k in df.columns and v not in df.columns})
    if "alt_m" not in df.columns:
        if "baroaltitude" in df.columns:
            df["alt_m"] = df["baroaltitude"] + baro_offset_m
        else:
            raise ValueError("no altitude column (alt_m / geoaltitude / baroaltitude)")
    need = ["time", "icao24", "lat", "lon", "alt_m"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"ADS-B table is missing columns: {missing}")
    for c in ("gs_ms", "track_deg", "vrate_ms"):
        if c not in df.columns:
            df[c] = np.nan
    for c in ("callsign", "type_code"):
        if c not in df.columns:
            df[c] = ""
    df["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    df["type_code"] = df["type_code"].fillna("").astype(str).str.strip().str.upper()
    df["icao24"] = df["icao24"].astype(str).str.lower()
    if np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = df["time"].astype("int64") / 1e9
    return df.dropna(subset=["time", "lat", "lon", "alt_m"])


def load_table(path, baro_offset_m: float = 0.0) -> pd.DataFrame:
    p = str(path)
    df = pd.read_parquet(p) if p.endswith((".parquet", ".pq")) else pd.read_csv(p)
    return normalize_table(df, baro_offset_m)


def flights_from_table(df: pd.DataFrame, min_points: int = 8) -> list[Flight]:
    out = []
    for icao, g in df.sort_values("time").groupby("icao24", sort=False):
        if len(g) < min_points:
            continue
        cs = next((c for c in g["callsign"] if c), "")
        tc = next((c for c in g["type_code"] if c), "")
        has_v = g["gs_ms"].notna().any() and g["track_deg"].notna().any()
        out.append(Flight(
            icao24=icao, t=g["time"].to_numpy(float), lat=g["lat"].to_numpy(float),
            lon=g["lon"].to_numpy(float), alt=g["alt_m"].to_numpy(float),
            gs=g["gs_ms"].to_numpy(float) if has_v else None,
            track=g["track_deg"].to_numpy(float) if has_v else None,
            vrate=g["vrate_ms"].to_numpy(float) if has_v else None,
            callsign=cs, type_code=tc))
    return out


# --------------------------------------------------------------------------------------
# readsb trace files (adsb.lol globe_history releases, ADS-B Exchange samples)
# trace row: [dt, lat, lon, alt_baro_ft|"ground"|null, gs_kt, track_deg, flags, vrate_fpm,
#             details|null, source, alt_geom_ft, geom_rate_fpm, ias_kt, roll_deg]
# --------------------------------------------------------------------------------------

def _maybe_gunzip(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def parse_readsb_trace(raw: bytes | dict, bbox=None, t_lo=None, t_hi=None,
                       baro_offset_m: float = 0.0) -> pd.DataFrame | None:
    """Parse one trace_full_<icao>.json (gzipped or not). bbox = (lat_min, lat_max, lon_min, lon_max)."""
    obj = raw if isinstance(raw, dict) else json.loads(_maybe_gunzip(raw))
    base = float(obj.get("timestamp", 0.0))
    icao = str(obj.get("icao", "")).lower().lstrip("~")
    rows = obj.get("trace") or []
    if not rows:
        return None
    recs = []
    callsign = ""
    for r in rows:
        t = base + float(r[0])
        if (t_lo is not None and t < t_lo) or (t_hi is not None and t > t_hi):
            continue
        lat, lon = r[1], r[2]
        if lat is None or lon is None:
            continue
        if bbox is not None and not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
            continue
        if len(r) > 8 and isinstance(r[8], dict) and r[8].get("flight"):
            callsign = str(r[8]["flight"]).strip()
        alt_baro = r[3]
        if alt_baro == "ground":
            continue
        alt_geom = r[10] if len(r) > 10 else None
        if alt_geom is not None:
            alt_m = float(alt_geom) * FT
        elif alt_baro is not None:
            alt_m = float(alt_baro) * FT + baro_offset_m
        else:
            continue
        gs = r[4] if len(r) > 4 else None
        trk = r[5] if len(r) > 5 else None
        vr = r[7] if len(r) > 7 else None
        recs.append((t, lat, lon, alt_m,
                     np.nan if gs is None else gs * KT,
                     np.nan if trk is None else float(trk),
                     np.nan if vr is None else vr * FPM, callsign))
    if not recs:
        return None
    df = pd.DataFrame(recs, columns=["time", "lat", "lon", "alt_m", "gs_ms", "track_deg", "vrate_ms", "callsign"])
    df["icao24"] = icao
    df["type_code"] = str(obj.get("t") or "").upper()
    df["registration"] = str(obj.get("r") or "")
    return df


class _ConcatReader(io.RawIOBase):
    """Read several files back-to-back (adsb.lol splits each day's tar into .tar.aa, .tar.ab, ...)."""

    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]
        self._i = 0

    def readable(self):
        return True

    def readinto(self, b):
        while self._i < len(self._files):
            n = self._files[self._i].readinto(b)
            if n:
                return n
            self._files[self._i].close()
            self._i += 1
        return 0


def _iter_trace_blobs(sources):
    """Yield raw bytes of every trace file in directories, single files, or (split) tar archives."""
    sources = [Path(s) for s in sources]
    tar_parts = sorted(s for s in sources if s.is_file() and ".tar" in s.name)
    if tar_parts:
        with tarfile.open(fileobj=io.BufferedReader(_ConcatReader(tar_parts)), mode="r|*") as tf:
            for m in tf:
                if m.isfile() and "trace_full" in m.name:
                    yield tf.extractfile(m).read()
    for s in sources:
        if s.is_dir():
            for root, _, files in os.walk(s):
                for fn in files:
                    if "trace_full" in fn:
                        yield (Path(root) / fn).read_bytes()
        elif s.is_file() and ".tar" not in s.name:
            yield s.read_bytes()


def convert_readsb(sources, bbox, t_lo, t_hi, tonal_only: bool = True,
                   baro_offset_m: float = 0.0, progress=print) -> pd.DataFrame:
    """Extract the reports inside bbox/time window from readsb traces into the canonical table."""
    frames = []
    for i, blob in enumerate(_iter_trace_blobs(sources)):
        try:
            df = parse_readsb_trace(blob, bbox, t_lo, t_hi, baro_offset_m)
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if df is not None and len(df) >= 2:
            frames.append(df)
        if progress and i and i % 20000 == 0:
            progress(f"  scanned {i:,} traces, kept {len(frames):,} aircraft")
    if not frames:
        return pd.DataFrame(columns=["time", "icao24", "lat", "lon", "alt_m", "gs_ms", "track_deg",
                                     "vrate_ms", "callsign", "type_code"])
    df = pd.concat(frames, ignore_index=True)
    return filter_tonal(df) if tonal_only else df
