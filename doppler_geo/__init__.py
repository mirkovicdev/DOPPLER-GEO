"""doppler_geo - locate an audio recording from the Doppler curves of passing propeller aircraft.

Pipeline:  audio -> spectrogram -> tone ridges -> Doppler curve fits (speed, miss distance,
closest-approach time) -> match against ADS-B tracks on a ground grid -> silence constraint
-> full retarded-time Doppler scoring -> continuous refinement.

Usage (see notebooks/doppler_geolocation.ipynb for a walk-through):

    import doppler_geo as dg
    c = dg.speed_of_sound(12.0)
    frame = dg.LocalFrame(lat0, lon0)
    an = dg.analyze_audio(x, fs, c)
    tracks = dg.prepare_tracks(dg.load_table("flights.parquet"), frame)
    result = dg.locate(an.good, tracks, frame, terrain, dg.Recording(t_start, 600, len(x) / fs), c=c,
                       heard=an.heard)
"""

from .adsb import load_table
from .geo import FlatTerrain, GridTerrain, LocalFrame, load_terrain
from .physics import emission_to_arrival, flyby_frequency, speed_of_sound, wind_vector
from .pipeline import Analysis, AnalysisConfig, analyze_audio, prepare_tracks, run_demo, run_locate
from .search import Recording, SearchConfig, locate

__version__ = "0.2.0"
__all__ = [
    "Analysis", "AnalysisConfig", "FlatTerrain", "GridTerrain", "LocalFrame", "Recording", "SearchConfig",
    "analyze_audio", "emission_to_arrival", "flyby_frequency", "load_table", "load_terrain", "locate",
    "prepare_tracks", "run_demo", "run_locate", "speed_of_sound", "wind_vector",
]
