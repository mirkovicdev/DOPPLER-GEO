"""Audio -> whitened spectrogram -> tone ridges -> harmonic groups (one group per flyover).

A propeller radiates a tone at its blade-passage frequency (blades x RPM / 60, typically
50-150 Hz) plus harmonics. The tone is often inaudible in a recording, but in a long-window
spectrogram it appears as a narrow line that descends in an S-curve as the aircraft passes.

Whitening: each frequency bin has its median over time subtracted (in dB), which removes
stationary lines such as mains hum, fridges and AC units. Then each frame has its median over
frequency subtracted, which removes broadband bursts (wind gusts, cars, jets). What remains
is "spectral contrast": how far a bin sticks out above its own typical level.

Ridge tracing (hysteresis, like Canny edges): start from strong seeds, follow the line forward
and backward with a slope-predicted, slope-widened search window, stop when the running
contrast fades into noise, trim the weak tail.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly


def load_audio(path, target_fs: float = 1000.0):
    """Load any audio file (or video file if ffmpeg is installed) as mono float at target_fs."""
    import soundfile as sf

    try:
        x, fs = sf.read(str(path), always_2d=True)
        x = x.mean(axis=1)
    except (RuntimeError, sf.LibsndfileError if hasattr(sf, "LibsndfileError") else RuntimeError):
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"can't decode {path}; install ffmpeg or convert it to WAV/FLAC first")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(int(target_fs)),
                              "-f", "f32le", "-"], check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.float32).astype(float), float(int(target_fs))
    return resample(x, fs, target_fs), float(target_fs)


def resample(x, fs, target_fs):
    if abs(fs - target_fs) < 1e-9:
        return np.asarray(x, float)
    fr = Fraction(target_fs / fs).limit_denominator(1000)
    return resample_poly(np.asarray(x, float), fr.numerator, fr.denominator)


@dataclass
class Spectrogram:
    t: np.ndarray          # frame centre times (s, recording-relative)
    f: np.ndarray          # frequencies (Hz)
    power_db: np.ndarray   # (n_freq, n_frames)
    contrast_db: np.ndarray

    @property
    def hop(self) -> float:
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else 1.0

    @property
    def df(self) -> float:
        return float(self.f[1] - self.f[0])


def compute_spectrogram(x, fs, win_s: float = 2.0, hop_s: float = 0.5, fmin: float = 30.0,
                        fmax: float = 400.0, pad: int = 4, chunk: int = 512) -> Spectrogram:
    """Hann-window STFT (computed in chunks, cropped to [fmin, fmax]) and its whitened contrast."""
    x = np.asarray(x, float)
    nper = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    nfft = nper * pad
    if len(x) < nper:
        raise ValueError("recording shorter than one analysis window")
    win = np.hanning(nper)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    starts = np.arange(0, len(x) - nper + 1, hop)
    out = np.empty((band.sum(), len(starts)), np.float32)
    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        frames = np.lib.stride_tricks.sliding_window_view(x, nper)[s] * win
        spec = np.fft.rfft(frames, n=nfft, axis=1)[:, band]
        out[:, i:i + len(s)] = (10 * np.log10(np.abs(spec) ** 2 + 1e-20)).T
    t = (starts + nper / 2) / fs
    return Spectrogram(t, freqs[band], out, whiten(out))


def whiten(power_db):
    c = power_db - np.median(power_db, axis=1, keepdims=True)
    return c - np.median(c, axis=0, keepdims=True)


@dataclass
class Ridge:
    t: np.ndarray
    f: np.ndarray
    snr: np.ndarray  # contrast (dB) at each point

    @property
    def t0(self):
        return float(self.t[0])

    @property
    def t1(self):
        return float(self.t[-1])

    @property
    def duration(self):
        return self.t1 - self.t0


@dataclass
class HarmonicGroup:
    """Ridges from one source; `t, f, w` are all harmonics rescaled to the fundamental."""

    ridges: list
    harmonics: list          # harmonic number of each ridge
    t: np.ndarray
    f: np.ndarray
    w: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def t0(self):
        return float(self.t.min())

    @property
    def t1(self):
        return float(self.t.max())


def _trace(C, used, fi, ti, step, f_axis, df, low_db, keep_db, max_miss, max_slope_hz_per_frame):
    """Follow one ridge from (fi, ti) in time direction `step` (+1 / -1). Returns (frames, freqs, values).
    Pixels already claimed by a stronger ridge count as misses, so the trace coasts through crossings."""
    F, T = C.shape
    ts, fs, vs = [ti], [f_axis[fi]], [C[fi, ti]]
    misses = 0
    t = ti
    while True:
        t += step
        if t < 0 or t >= T:
            break
        slope = 0.0
        if len(ts) >= 3:
            k = min(len(ts), 8)
            tt = np.asarray(ts[-k:], float)
            if np.ptp(tt) > 0:
                slope = float(np.clip(np.polyfit(tt, fs[-k:], 1)[0], -max_slope_hz_per_frame, max_slope_hz_per_frame))
        pred = fs[-1] + slope * (t - ts[-1])
        w = 0.8 + 0.6 * abs(slope) + 0.3 * misses
        lo = max(1, int((pred - w - f_axis[0]) / df))
        hi = min(F - 2, int((pred + w - f_axis[0]) / df) + 1)
        if hi <= lo:
            break
        seg = C[lo:hi + 1, t]
        # among local maxima, prefer the one closest to the prediction (continuity beats loudness)
        peaks = np.flatnonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1 if len(seg) > 2 else np.array([], int)
        b = -1
        if len(peaks):
            sigma = 0.4 + 0.25 * abs(slope) + 0.2 * misses
            dev = (f_axis[lo + peaks] - pred) / sigma
            score = seg[peaks] - 3.0 * dev * dev
            b = lo + int(peaks[np.argmax(score)])
        val = C[b, t] if b >= 0 else -np.inf
        if b >= 0 and val >= low_db and not used[b, t]:
            a, c = C[b - 1, t], C[b + 1, t]
            den = a - 2 * val + c
            off = 0.5 * (a - c) / den if den < 0 else 0.0
            ts.append(t)
            fs.append(f_axis[b] + off * df)
            vs.append(val)
            misses = 0
        else:
            misses += 1
            if misses > max_miss:
                break
        if len(vs) >= 12 and np.mean(vs[-12:]) < keep_db:
            break
    while len(vs) > 1 and vs[-1] < keep_db:  # trim the weak tail
        ts.pop()
        fs.pop()
        vs.pop()
    return ts, fs, vs


def track_ridges(spec: Spectrogram, seed_db: float = 11.0, keep_db: float = 6.5, low_db: float = 3.0,
                 max_miss_s: float = 4.0, max_slope_hz_s: float = 12.0, min_dur_s: float = 20.0,
                 min_median_db: float = 7.0, min_fill: float = 0.5, mask_hz: float = 2.0,
                 merge_gap_s: float = 20.0, merge_tol_hz: float = 3.0) -> list[Ridge]:
    """Seeded bidirectional ridge tracing on the whitened spectrogram."""
    C = spec.contrast_db
    C3 = uniform_filter1d(C, 3, axis=1)
    F, T = C.shape
    df, hop = spec.df, spec.hop
    used = np.zeros(C.shape, bool)
    seeds = np.argwhere(C3[1:-1] >= seed_db) + [1, 0]
    vals = C3[seeds[:, 0], seeds[:, 1]]
    is_peak = (vals >= C3[seeds[:, 0] - 1, seeds[:, 1]]) & (vals >= C3[seeds[:, 0] + 1, seeds[:, 1]])
    seeds, vals = seeds[is_peak], vals[is_peak]
    mb = max(1, int(round(mask_hz / df)))
    max_miss = max(1, int(round(max_miss_s / hop)))
    slope_lim = max_slope_hz_s * hop
    ridges = []
    for fi, ti in seeds[np.argsort(-vals)]:
        if used[fi, ti]:
            continue
        fwd = _trace(C, used, fi, ti, +1, spec.f, df, low_db, keep_db, max_miss, slope_lim)
        bwd = _trace(C, used, fi, ti, -1, spec.f, df, low_db, keep_db, max_miss, slope_lim)
        frames = bwd[0][:0:-1] + fwd[0]
        freqs = np.array(bwd[1][:0:-1] + fwd[1])
        snr = np.array(bwd[2][:0:-1] + fwd[2])
        all_frames = np.arange(frames[0], frames[-1] + 1)
        for fr, fq in zip(all_frames, np.interp(all_frames, frames, freqs)):
            b = int((fq - spec.f[0]) / df)
            used[max(0, b - mb):b + mb + 1, fr] = True
        if len(frames) < 10:
            continue
        ridges.append(Ridge(spec.t[np.array(frames)], freqs, snr))
    out = []
    for r in _merge_fragments(ridges, merge_gap_s, merge_tol_hz):
        n_expected = r.duration / hop + 1
        if r.duration >= min_dur_s and len(r.t) / n_expected >= min_fill and np.median(r.snr) >= min_median_db:
            out.append(r)
    return sorted(out, key=lambda r: r.t0)


def _merge_fragments(ridges, gap_s, tol_hz):
    """Join a ridge that ends with one that starts shortly after at the extrapolated frequency
    (a flyover broken at its steepest point or at a crossing)."""
    ridges = sorted(ridges, key=lambda r: r.t0)
    starts = np.array([r.t0 for r in ridges])
    nxt = [-1] * len(ridges)
    taken = [False] * len(ridges)
    for i, a in enumerate(ridges):
        k = min(len(a.t), 12)
        slope = float(np.clip(np.polyfit(a.t[-k:], a.f[-k:], 1)[0], -10, 2)) if k >= 3 else 0.0
        best, best_err = -1, np.inf
        for j in range(np.searchsorted(starts, a.t1, "right"), np.searchsorted(starts, a.t1 + gap_s, "right")):
            if taken[j]:
                continue
            gap = ridges[j].t0 - a.t1
            err = abs(ridges[j].f[0] - (a.f[-1] + slope * gap))
            if err < min(best_err, tol_hz + 0.4 * abs(slope) * gap):
                best, best_err = j, err
        if best >= 0:
            nxt[i] = best
            taken[best] = True
    out = []
    for i in range(len(ridges)):
        if taken[i]:
            continue
        chain = [i]
        while nxt[chain[-1]] >= 0:
            chain.append(nxt[chain[-1]])
        rs = [ridges[j] for j in chain]
        out.append(Ridge(np.concatenate([r.t for r in rs]), np.concatenate([r.f for r in rs]),
                         np.concatenate([r.snr for r in rs])))
    return out


def group_harmonics(ridges: list[Ridge], ratio_tol: float = 0.012, min_overlap: float = 0.4,
                    min_swing: float = 0.04) -> list[HarmonicGroup]:
    """Group ridges whose frequency ratio stays constant at a small-integer ratio (same propeller).

    The ratio test only means something while the pitch is changing: two unrelated planes on flat
    plateaus can sit at a constant 3:2 by chance. So the overlap must contain a >= `min_swing`
    relative frequency change."""
    n = len(ridges)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = ridges[i], ridges[j]
            lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
            if hi - lo < min_overlap * min(a.duration, b.duration):
                continue
            tt = np.linspace(lo, hi, 50)
            fa = np.interp(tt, a.t, a.f)
            if np.ptp(np.log(fa)) < min_swing:
                continue
            ratio = np.interp(tt, b.t, b.f) / fa
            if ratio.mean() < 1:
                ratio = 1 / ratio
            rm = np.median(ratio)
            if np.std(ratio) / rm > ratio_tol:
                continue
            if any(abs(rm - p / q) / rm < ratio_tol for q in (1, 2, 3) for p in range(q + 1, 6 * q + 1)):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(ridges[i])
    out = []
    for members in groups.values():
        members.sort(key=lambda r: np.median(r.f))
        base = np.median(members[0].f)
        rel = [np.median(r.f) / base for r in members]
        k = next((k for k in (1, 2, 3) if all(abs(x * k - round(x * k)) < 0.06 for x in rel)), 1)
        harm = [int(round(x * k)) for x in rel]
        t = np.concatenate([r.t for r in members])
        f = np.concatenate([r.f / h for r, h in zip(members, harm)])
        w = np.concatenate([np.clip(r.snr, 0, 30) for r in members])
        o = np.argsort(t, kind="stable")
        out.append(HarmonicGroup(members, harm, t[o], f[o], w[o] / w.mean()))
    return sorted(out, key=lambda g: g.t0)
