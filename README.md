# doppler_geo

Estimate where an audio recording was made from the Doppler-shifted tones of propeller aircraft that
passed nearby, matched against ADS-B flight tracks.

A propeller radiates a narrow-band tone at its blade-passage frequency. As an aircraft passes a
microphone, the received frequency traces a Doppler curve. That curve determines three things: the
aircraft's speed, its closest-approach distance, and the time of closest approach. Most aircraft
broadcast their trajectories through ADS-B. A few such curves therefore constrain where the microphone
was. `doppler_geo` implements the full chain:

1. tone extraction;
2. retarded-time Doppler fitting;
3. a grid search over candidate positions with timing and silence constraints;
4. maximum-likelihood refinement.

It includes a synthetic scenario generator, so the method can be tested end to end against known
ground truth.

> **Responsible use.** Apply this to your own recordings or to the synthetic scenarios. Do not use it to
> locate people without their consent.

---

## Contents

| | |
|---|---|
| `doppler_geo/` | Python package and command-line tool |
| [`notebooks/doppler_geolocation.ipynb`](notebooks/doppler_geolocation.ipynb) | Executed notebook: signal model, method, results and a 12-scenario Monte Carlo. Every number and figure is computed by the package |
| [`paper/doppler_geolocation.pdf`](paper/doppler_geolocation.pdf) | Eight-page write-up of the same material (LaTeX source alongside) |
| `tests/` | Physics identities, fitting, track smoothing, trace parsing, end-to-end run |

## Installation

```bash
pip install -e ".[fast,test]"
```

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `fast` | numba | compiling the hot loops; strongly recommended |
| `dem` | rasterio | reading GeoTIFF elevation models |
| `notebook` | Jupyter | running the notebook |

## Quick start

```bash
python -m doppler_geo demo --seed 7 --out out/demo
```

This generates a synthetic scenario, writes its recording and ADS-B feed, and locates the receiver
from those alone. It takes about 30–45 s. The output directory contains:

| File | Content |
|---|---|
| `recording.wav` | 20-min synthetic recording (4 kHz mono) |
| `adsb.parquet`, `terrain.npz` | ADS-B feed and elevation model given to the estimator |
| `truth.json` | Ground truth (not used by the estimator) |
| `result.json` | Estimate, uncertainty, matched aircraft, candidate counts per stage |
| `result.kml` | Estimate, true position and matched flight paths for Google Earth |
| `figures/fig1`–`fig7` | Figures as PDF and 300-dpi PNG, with `captions.md` |

All aircraft in the synthetic scenarios are fictitious; their callsigns start with `SIM`.

### Python API

```python
import doppler_geo as dg

c = dg.speed_of_sound(12.0)                      # air temperature in °C
frame = dg.LocalFrame(38.45, -98.20)             # search centre
an = dg.analyze_audio(x, fs, c)                  # x: mono samples, fs: sample rate
tracks = dg.prepare_tracks(dg.load_table("flights.parquet"), frame)
res = dg.locate(an.good, tracks, frame, dg.FlatTerrain(400.0),
                dg.Recording(start_guess=t_utc, start_unc=600, duration=len(x) / fs),
                c=c, heard=an.heard)
print(res.best["lat"], res.best["lon"], res.best["ambiguous"])
```

---

## Method

### Signal model (`physics.py`)

Sound emitted at time τ arrives at time t:

```
t = τ + R(τ)/c                     (R = slant range, c = speed of sound)
f(t) = f0 / (dt/dτ) = f0 · c / (c + v_r(τ))
```

- **Radial velocity.** `v_r = dR/dτ` is positive when the source recedes. It is evaluated at the
  emission time; evaluating the geometry at the arrival time gives a symmetric, biased curve.
- **Straight flight.** For a straight, constant-speed pass the emission time has a closed form
  (`flyby_emission_time`):

  ```
  τ − τ_ca = (c²T − √(c²v²T² + (c²−v²)d²)) / (c² − v²),   T = t − τ_ca
  ```

- **Wind.** In a uniform wind `w`, the travel time solves `|D − wT| = cT`, and
  `dt/dτ = 1 − (k·v)/(c²T + k·w)` with `k = D − wT` (`emission_to_arrival`).
- **Real trajectories.** Turning and climbing tracks are evaluated numerically with the same equations.

### Tone extraction (`spectral.py`)

- 2-s Hann STFT, 0.5-s hop, 30–400 Hz.
- Each frequency bin's median over time is subtracted, which suppresses stationary lines such as mains
  hum. Each frame's median over frequency is subtracted, which suppresses broadband transients.
- Ridges are traced from strong seeds in both time directions inside a slope-adaptive window. The local
  maximum closest to a linear prediction is preferred over the loudest one, which keeps the trace on its
  own line where two lines cross.
- Ridges with a steady integer frequency ratio are grouped as harmonics of one propeller. If a group fits
  poorly, the ridge whose removal most improves the fit is detached.

### Single-flyover estimation (`fit.py`)

The retarded-time model is fitted with robust multi-start least squares. Each feature of the curve fixes
one parameter:

| Feature of the curve | Parameter |
|---|---|
| (f₊ − f₋)/(f₊ + f₋) = v/c | speed |
| slope at the inflection, −f0·v²/(c·d) | closest-approach distance |
| inflection time, τ_ca + d/c | time of closest approach |
| centre frequency | blade-passage frequency f0 |

A fit is used only if all of these hold:

- the ridge brackets the closest approach;
- the residual is small;
- no parameter sits at a bound;
- the speed is plausible for a light aircraft (> 33 m/s);
- the distance is within the search range.

### Flight tracks (`adsb.py`, `trajectory.py`)

- **Input.** readsb trace files (e.g. the adsb.lol `globe_history` archive) or any table with time,
  position, altitude and velocity. OpenSky column names are mapped automatically.
- **Type filter.** Piston and turboprop types are kept by ICAO type designator; unknown types are kept.
- **Smoothing.** GNSS positions and velocities are fused with a Rauch–Tung–Striebel smoother.
- **Gaps.** Dropouts shorter than 60 s are bridged. Longer gaps split the track.

### Search (`search.py`)

1. **Grid.** 200-m cells over a 60 × 60 km area, placed on the terrain surface in a local
   East-North-Up frame.
2. **Passes.** Every closest approach of every propeller track to every cell within 8 km, with its
   arrival time, slant range and speed. This step is compiled with numba: about 10⁶ passes per second.
3. **Matching.**
   - The closest flyover must match a pass within ×0.6–1.6 in distance and ×0.8–1.25 in speed, inside
     the clock prior. That pass fixes the clock epoch.
   - Each further flyover must match another pass at the same cell within ±15 s of that epoch.
4. **Silence constraint.** A cell is rejected if another propeller aircraft passed closer than the
   farthest matched flyover during the recording and no tone was detected at that time.
5. **Doppler likelihood.** The full curve predicted from each matched track is compared with the
   measured ridge (RMS misfit in Hz). Rest frequencies are solved in closed form.
6. **Refinement.** Nelder–Mead over position, clock epoch, wind and sound-speed offset; the last two are
   nuisance parameters with weak priors. The output includes a misfit surface and approximate 68/95 %
   regions.

The result is flagged `ambiguous` when a position more than 500 m away fits within 10 %, and
`low_confidence` when the best misfit exceeds 0.8 Hz.

---

## Results on synthetic scenarios

Twelve scenarios with independent traffic, terrain, receiver position and clock offset. In each, the
estimator assumes 10 °C and still air (the truth is 6 °C with a 6 m/s wind), and the recording start time
is known only to ±15 min (actual offsets up to ±9 min).

| Seed | Flyovers used | Error | Clock error | Misfit (Hz) | Fitted air temperature |
|---:|:---:|---:|---:|---:|---:|
| 1 | 2 | 63 m | −0.2 s | 0.27 | 15.6 °C |
| 2 | 3 | 87 m | −0.3 s | 0.37 | 10.1 °C |
| 3 | 2 | 19 m | +0.2 s | 0.37 | 6.0 °C |
| 4 | 2 | 25 m | +0.0 s | 0.36 | 7.5 °C |
| 5 | 4 | 71 m | +0.8 s | 0.38 | 7.0 °C |
| 6 | 2 | 83 m | −1.1 s | 0.21 | 13.4 °C |
| 7 | 3 | 15 m | +0.2 s | 0.44 | 3.8 °C |
| 8 | 2 | 27 m | +0.1 s | 0.35 | 9.2 °C |
| 9 | 4 | 53 m | −0.5 s | 0.43 | 9.4 °C |
| 10 | 3 | 29 m | −0.5 s | 0.33 | 3.4 °C |
| 11 | 3 | 13 m | −0.3 s | 0.35 | 7.9 °C |
| 12 | 1 | 2,855 m | +0.8 s | 0.24 | 9.1 °C (flagged ambiguous) |

- **Eleven of twelve scenarios** are located within 13–87 m (median 29 m), with the clock recovered to
  within 1.1 s.
- **Seed 12** contains only one usable flyover, which constrains the position to a line. It is reported
  as ambiguous.
- **Atmosphere.** Fitted temperatures scatter widely around the true 6 °C. Wind and sound speed are
  nuisance parameters, not measurements.
- **Runtime.** A full run takes 20–45 s on a laptop.

Synthetic recordings are a best case. They omit codec filtering, refraction, ground reflections,
incomplete ADS-B coverage and aircraft without transponders.

---

## Using recorded data

You need four inputs:

- **A recording.** One or more propeller aircraft passing within a few kilometres, plus the date and an
  approximate start time. Two or more flyovers, or a clock accurate to a few seconds, give much stronger
  results.
- **ADS-B for the day and area.** For example, the [adsb.lol](https://adsb.lol) daily `globe_history`
  archive (ODbL), distributed as split `.tar` files. Extract a region:

  ```bash
  python -m doppler_geo convert-readsb v2026.11.01-planes-readsb-prod-0.tar.aa v2026.11.01-planes-readsb-prod-0.tar.ab --bbox 37.9,39.0,-98.8,-97.6 --start 2026-11-01T15:00:00Z --end 2026-11-01T17:00:00Z --out flights.parquet
  ```

- **A search centre.** The default search area is ±30 km around it.
- **Weather and terrain.**
  - Air temperature, for example from the nearest airport's METAR.
  - An elevation model as GeoTIFF (SRTM or Copernicus GLO-30), via the `dem` extra.
  - Because ADS-B altitudes are ellipsoidal, add the local geoid offset to the elevation model: about
    −30 m over most of the US.

Inspect a recording:

```bash
python -m doppler_geo analyze recording.wav --temp 12 --out out/inspect
```

Run the full search:

```bash
python -m doppler_geo locate recording.wav --adsb flights.parquet --center 38.45,-98.20 --start 2026-11-01T15:42:00Z --start-unc 600 --temp 12 --terrain dem.tif --geoid-offset -28 --out out/run
```

Audio formats readable by libsndfile load directly. Other containers, including video files, need
`ffmpeg` on the path.

---

## Limitations

- **Requirements.** The method needs nearby propeller traffic, a known date and approximate time, and
  ADS-B coverage of the aircraft involved. It does not apply to arbitrary recordings.
- **Two-flyover ambiguity.** With an unknown clock offset, two flyovers can admit a second solution. At
  that position both aircraft pass at similar distances, and both are heard shifted by the same amount,
  which the clock offset absorbs. A third flyover, the silence constraint, a reliable clock or a smaller
  search area removes it, and the result is flagged when it remains.
- **One flyover** constrains the receiver to lines parallel to the track (a mirror pair), not to a point.
- **Crossing tones.** Where two aircraft's tones cross in the spectrogram, a ridge can still be assigned
  to the wrong source. Inspect `fig2_spectrogram` before trusting a result.
- **The silence constraint** is only as complete as the ADS-B data. An aircraft with poor coverage, or with
  a quieter tone than the matched ones, can wrongly reject the true position.
- **Uncertainty regions** reflect measurement noise only. They exclude model error such as wind
  structure, refraction, ADS-B latency and temperature.

---

## Project layout

```
doppler_geo/
  physics.py     Doppler shift with retarded time (closed form; numerical with wind)
  geo.py         WGS84 / ECEF / local ENU conversions, terrain models
  spectral.py    audio loading, whitened spectrogram, ridge tracing, harmonic grouping
  fit.py         single-flyover fit, quality flags, de-duplication
  adsb.py        readsb trace parser, table loader, aircraft-type filter
  trajectory.py  RTS track smoothing and resampling
  search.py      grid, passes, matching, silence constraint, likelihood, refinement
  _kernels.py    numba-compiled loops (passes, silence check, smoother)
  synth.py       synthetic scenarios: traffic, terrain, audio rendering, ADS-B sampling
  viz.py         figures (PDF + PNG), captions, KML
  pipeline.py    analyze / locate / demo orchestration
  cli.py         python -m doppler_geo {demo, analyze, locate, convert-readsb}
notebooks/
  doppler_geolocation.ipynb   executed notebook
  figures/                    notebook figures (PDF + PNG)
paper/
  doppler_geolocation.tex     LaTeX source (pdflatex, two passes)
  doppler_geolocation.pdf     compiled document
  figures/                    figure PDFs used by the paper
tests/
```

Run the tests with `python -m pytest -q`. The end-to-end test takes about a minute.

---

## References

- B. G. Ferguson, *J. Acoust. Soc. Am.* **92** (1992), [doi:10.1121/1.403934](https://doi.org/10.1121/1.403934).
  Single-sensor estimation of propeller-aircraft speed and altitude from the Doppler-shifted
  blade-passage tone.
- B. G. Ferguson and B. G. Quinn, *J. Acoust. Soc. Am.* **96** (1994), [doi:10.1121/1.410320](https://doi.org/10.1121/1.410320).
- H. H. Hubbard (ed.), *Aeroacoustics of Flight Vehicles: Theory and Practice*, NASA RP-1258 (1991).
- H. E. Rauch, F. Tung and C. T. Striebel, *AIAA J.* **3**, 1445 (1965).
- J. A. Nelder and R. Mead, *Comput. J.* **7**, 308 (1965).
- ADS-B data: [adsb.lol](https://adsb.lol) (ODbL), readsb trace format.

## License

MIT; see [LICENSE](LICENSE). ADS-B data from adsb.lol is licensed separately under the ODbL.
