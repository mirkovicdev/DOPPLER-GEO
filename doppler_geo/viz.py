"""Publication-style figures (vector PDF + 300 dpi PNG), numbered captions, and a KML export.

Figures carry no titles: as in a paper, each is described by its caption (see `captions`).
Sizes follow a two-column journal layout: 3.4 in (single column) and 7.0 in (full width).
Functions return matplotlib Figure objects, so they display inline in a notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .physics import doppler_curve, flyby_frequency, speed_of_sound

W1, W2 = 3.4, 7.0
RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "legend.frameon": False, "legend.handlelength": 1.6, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "image.interpolation": "nearest",
}
SERIES = ["#1f4e9c", "#c0392b", "#2e7d32", "#8e44ad", "#d35400", "#00838f"]
TRUE_STYLE = dict(marker="+", ms=9, mew=1.4, color="#c0392b", ls="")
EST_STYLE = dict(marker="x", ms=6, mew=1.2, color="black", ls="")


def use_style():
    matplotlib.rcParams.update(RC)


def _panel(ax, letter, x=-0.02, y=1.015):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def save(fig, stem):
    """Write <stem>.pdf (vector) and <stem>.png (300 dpi)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)


# ---------------------------------------------------------------- Fig. 2  spectrogram

def fig_spectrogram(spec, flyovers):
    use_style()
    fig = Figure(figsize=(W2, 4.9), layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ext = [spec.t[0] / 60, spec.t[-1] / 60, spec.f[0], spec.f[-1]]
    p = spec.power_db - np.median(spec.power_db)
    lo, hi = np.percentile(p, [2, 99.8])
    im = ax1.imshow(p, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
    fig.colorbar(im, ax=ax1, pad=0.01, aspect=18).set_label("PSD (dB re median)")
    im = ax2.imshow(spec.contrast_db, origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=20,
                    rasterized=True)
    fig.colorbar(im, ax=ax2, pad=0.01, aspect=18).set_label("Spectral contrast (dB)")
    k = 0
    for fl in flyovers:
        if fl.quality != "ok":
            for r in fl.ridges:
                ax2.plot(r.t / 60, r.f, ".", ms=0.8, color="#9e9e9e", rasterized=True)
            continue
        col = SERIES[k % len(SERIES)]
        tt = np.linspace(fl.t.min(), fl.t.max(), 400)
        for r, h in zip(fl.ridges, fl.harmonics):
            ax2.plot(r.t / 60, r.f, "o", ms=1.0, mew=0, color=col, rasterized=True)
            ax2.plot(tt / 60, h * fl.model(tt), "-", lw=0.7, color=col)
        top = max(fl.harmonics) * fl.f0
        ax2.annotate(fl.name, (fl.t_arr / 60, top), xytext=(4, 6), textcoords="offset points", fontsize=8,
                     color=col)
        k += 1
    for ax in (ax1, ax2):
        ax.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (min)")
    _panel(ax1, "a")
    _panel(ax2, "b")
    return fig


# ---------------------------------------------------------------- Fig. 1  Doppler model

def fig_doppler_model(c=None):
    use_style()
    c = c or speed_of_sound(15)
    fig = Figure(figsize=(W2, 4.4), layout="constrained")
    axs = fig.subplots(2, 2)
    x = np.linspace(-60, 60, 1201)
    f0 = 85.0

    ax = axs[0, 0]
    for d, col in zip((300, 1000, 3000), SERIES):
        ax.plot(x, flyby_frequency(x + d / c, f0, 55, d, 0, c), color=col, label=fr"$d = {d / 1000:g}$ km")
    ax.legend(loc="upper right")
    ax.text(0.03, 0.06, r"$v = 55\ \mathrm{m\,s^{-1}}$", transform=ax.transAxes, fontsize=7.5)

    ax = axs[0, 1]
    for v, col in zip((40, 60, 100), SERIES):
        ax.plot(x, flyby_frequency(x + 1000 / c, f0, v, 1000, 0, c), color=col,
                label=fr"$v = {v}\ \mathrm{{m\,s^{{-1}}}}$")
    ax.legend(loc="upper right")
    ax.text(0.03, 0.06, r"$d = 1$ km", transform=ax.transAxes, fontsize=7.5)

    v, d = 110.0, 500.0
    ret = flyby_frequency(x, f0, v, d, 0, c)
    inst = flyby_frequency(x, f0, v, d, 0, c, retarded=False)
    ax = axs[1, 0]
    ax.plot(x, ret, color=SERIES[0], label="retarded time")
    ax.plot(x, inst, color=SERIES[1], ls="--", label="instantaneous")
    ax.legend(loc="upper right")
    ax.text(0.03, 0.06, r"$v = 110\ \mathrm{m\,s^{-1}},\ d = 0.5$ km", transform=ax.transAxes, fontsize=7.5)
    ax = axs[1, 1]
    ax.plot(x, inst - ret, color="black")
    ax.axhline(0, color="#9e9e9e", lw=0.5)
    ax.set_ylabel(r"$f_\mathrm{inst} - f_\mathrm{ret}$ (Hz)")

    for a, letter in zip(axs.flat, "abcd"):
        # (a, b): time from the arrival of the closest-approach sound; (c, d): from its emission
        a.set_xlabel(r"$t - t_\mathrm{ca}$ (s)" if letter in "ab" else r"$t - \tau_\mathrm{ca}$ (s)")
        _panel(a, letter)
    for a in (axs[0, 0], axs[0, 1], axs[1, 0]):
        a.set_ylabel(r"$f$ (Hz)")
    return fig


# ---------------------------------------------------------------- Fig. 3  single-flyover fits

def fig_flyover_fits(flyovers):
    use_style()
    good = [f for f in flyovers if f.quality == "ok"]
    n = max(len(good), 1)
    fig = Figure(figsize=(W2, 3.3), layout="constrained")
    axs = fig.subplots(2, n, sharex="col", height_ratios=[2.2, 1], squeeze=False)
    for j, fl in enumerate(good):
        x = fl.t - fl.t_arr
        top, bot = axs[0, j], axs[1, j]
        top.plot(x, fl.f, "o", ms=1.2, mew=0, color="black", rasterized=True, label="ridge")
        xx = np.linspace(x.min(), x.max(), 500)
        top.plot(xx, fl.model(xx + fl.t_arr), color=SERIES[1], lw=1.0, label="model")
        s = fl.sigma
        txt = (fr"$\hat f_0 = {fl.f0:.2f}\pm{s['f0']:.2f}$ Hz" "\n"
               fr"$\hat v = {fl.v:.1f}\pm{s['v']:.1f}\ \mathrm{{m\,s^{{-1}}}}$" "\n"
               fr"$\hat d = {fl.d:.0f}\pm{s['d']:.0f}$ m")
        top.text(0.97, 0.95, txt, transform=top.transAxes, ha="right", va="top", fontsize=7)
        top.set_title(fl.name, fontsize=8.5)
        res = fl.f - fl.model(fl.t)
        bot.plot(x, res, "o", ms=1.0, mew=0, color="black", rasterized=True)
        bot.axhline(0, color=SERIES[1], lw=0.8)
        lim = max(1.0, float(np.percentile(np.abs(res), 99)) * 1.2)
        bot.set_ylim(-lim, lim)
        bot.set_xlabel(r"$t - \hat t_\mathrm{ca}$ (s)")
        if j == 0:
            top.set_ylabel(r"$f / h$ (Hz)")
            bot.set_ylabel("Residual (Hz)")
            top.legend(loc="lower left", markerscale=3)
        _panel(top, "abcdef"[j])
    return fig


# ---------------------------------------------------------------- Fig. 4  candidate reduction

def fig_reduction(funnel):
    use_style()
    fig = Figure(figsize=(W1, 2.4), layout="constrained")
    ax = fig.subplots()
    n = np.array([max(f[2], 1) for f in funnel], float)
    x = np.arange(len(n))
    ax.semilogy(x, n, "o-", color="black", ms=3.5, lw=0.8)
    for xi, ni in zip(x, n):
        ax.annotate(f"{int(ni):,}", (xi, ni), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x, [f[1] for f in funnel])
    ax.minorticks_off()
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_ylim(0.5, n.max() * 6)
    ax.set_xlim(-0.4, len(n) - 0.6)
    ax.set_ylabel("Candidate cells")
    ax.set_xlabel("Constraint applied (cumulative)")
    return fig


# ---------------------------------------------------------------- Fig. 5  spatial constraints

def _zoom_box(result, key, min_half=4.0):
    g = result.grid
    xy = g.cell_enu(result.stage_cells[key])[:, :2] / 1000
    if len(xy) == 0:
        cx, cy = result.best["e"] / 1000, result.best["n"] / 1000
        return [cx - min_half, cx + min_half, cy - min_half, cy + min_half]
    cx, cy = np.median(xy, axis=0)
    half = max(min_half, 1.15 * float(np.max(np.abs(xy - [cx, cy]))))
    return [cx - half, cx + half, cy - half, cy + half]


def _marks(ax, best, truth):
    if truth is not None:
        ax.plot(truth["e"] / 1000, truth["n"] / 1000, label="true position", **TRUE_STYLE)
    ax.plot(best["e"] / 1000, best["n"] / 1000, label="estimate", **EST_STYLE)


def fig_constraints(result, truth=None):
    use_style()
    g = result.grid
    r = g.res / 2000
    ext = [g.e[0] / 1000 - r, g.e[-1] / 1000 + r, g.n[0] / 1000 - r, g.n[-1] / 1000 + r]
    keys = list(result.stage_cells)
    first, matched, silent = keys[0], keys[-2] if len(keys) > 1 else keys[0], keys[-1]
    zoom = _zoom_box(result, matched)

    fig = Figure(figsize=(W2, 6.6), layout="constrained")
    axs = fig.subplots(2, 2)
    ax = axs[0, 0]
    for tr in result.tracks:
        ax.plot(tr.pos[:, 0] / 1000, tr.pos[:, 1] / 1000, color="#b0b0b0", lw=0.35)
    for k, tid in enumerate(result.best["track_ids"]):
        tr = result.tracks[tid]
        ax.plot(tr.pos[:, 0] / 1000, tr.pos[:, 1] / 1000, color=SERIES[k % len(SERIES)], lw=1.1,
                label=f"{result.events[k].name}: {tr.label} ({tr.type_code})")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _marks(ax, result.best, truth)
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1.2, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)

    ax = axs[0, 1]
    ax.imshow(g.mask(result.stage_cells[first]), origin="lower", extent=ext,
              cmap=ListedColormap(["white", "#7f9fcf"]), rasterized=True)
    ax.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                           fill=False, ec="black", lw=0.6, ls="--"))
    _marks(ax, result.best, truth)

    ax = axs[1, 0]
    xy = g.cell_enu(result.stage_cells[matched])[:, :2] / 1000
    ax.plot(xy[:, 0], xy[:, 1], "s", ms=3.2, mew=0, color="#7f9fcf")
    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    _marks(ax, result.best, truth)

    ax = axs[1, 1]
    cells = result.stage_cells[silent]
    xy = g.cell_enu(cells)[:, :2] / 1000
    score = result.cell_score.ravel()[cells]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=score, s=9, marker="s", cmap="viridis_r", linewidths=0,
                    vmin=np.nanmin(score) if np.isfinite(score).any() else 0)
    fig.colorbar(sc, ax=ax, pad=0.01, aspect=25).set_label("RMS Doppler misfit (Hz)")
    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    _marks(ax, result.best, truth)
    ax.legend(loc="lower right", fontsize=6.5)

    for a, letter in zip(axs.flat, "abcd"):
        a.set_aspect("equal")
        a.set_xlabel("Easting (km)")
        a.set_ylabel("Northing (km)")
        _panel(a, letter)
    return fig


# ---------------------------------------------------------------- Fig. 6  misfit surface

def fig_misfit_surface(result, truth=None):
    from matplotlib.ticker import MaxNLocator

    use_style()
    best = result.best
    fig = Figure(figsize=(W2, 3.1), layout="constrained")
    axs = fig.subplots(1, 2)
    for ax, s, letter in ((axs[0], result.surface, "a"), (axs[1], result.surface["fine"], "b")):
        e, n = s["e"] - best["e"], s["n"] - best["n"]
        levels = MaxNLocator(12).tick_values(s["rms"].min(), np.percentile(s["rms"], 97))
        cf = ax.contourf(e, n, s["rms"], levels=levels, cmap="viridis_r", extend="max")
        fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.92).set_label("RMS Doppler misfit (Hz)")
        if letter == "b":
            cs = ax.contour(e, n, s["dchi2"], levels=[2.30, 6.18], colors="white", linewidths=[0.9, 0.7],
                            linestyles=["-", "--"])
            ax.clabel(cs, fmt={2.30: "68%", 6.18: "95%"}, fontsize=6.5)
        else:
            h = result.surface["fine"]["e"][-1] - best["e"]
            ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, ec="white", lw=0.6, ls="--"))
        ax.plot(0, 0, label="estimate", **{**EST_STYLE, "color": "white"})
        if truth is not None:
            te, tn = truth["e"] - best["e"], truth["n"] - best["n"]
            if abs(te) <= e[-1] and abs(tn) <= n[-1]:
                ax.plot(te, tn, label="true position", **TRUE_STYLE)
        ax.set_aspect("equal")
        ax.set_xlabel("East offset (m)")
        ax.set_ylabel("North offset (m)")
        _panel(ax, letter)
    axs[1].legend(loc="lower left", fontsize=6.5, labelcolor="white")
    return fig


# ---------------------------------------------------------------- Fig. 7  ADS-B prediction vs data

def _predicted(result, cand, k, t, c, wind):
    obs = np.array([cand["e"], cand["n"], cand["up"]])
    tr = result.tracks[cand["track_ids"][k]]
    cw = np.asarray(cand["wind_enu"]) if cand.get("wind_enu") is not None else wind
    cur = doppler_curve(tr, obs, cand["t0"] + t.min(), cand["t0"] + t.max(), cand.get("c", c), cw)
    if cur is None:
        return np.full_like(t, np.nan)
    return cand["f0_hz"][k] * np.interp(cand["t0"] + t, cur[0], cur[1], left=np.nan, right=np.nan)


def fig_adsb_match(result, c, wind=None):
    use_style()
    evs, best = result.events, result.best
    order = np.argsort([ev.t_arr for ev in evs])
    alt = next((cd for cd in result.candidates[1:]
                if np.hypot(cd["e"] - best["e"], cd["n"] - best["n"]) > 500), None)
    n = len(evs)
    fig = Figure(figsize=(W2, 3.4), layout="constrained")
    axs = fig.subplots(2, n, sharex="col", height_ratios=[2.2, 1], squeeze=False)
    for j, k in enumerate(order):
        ev = evs[k]
        top, bot = axs[0, j], axs[1, j]
        top.plot(ev.t, ev.f, "o", ms=1.2, mew=0, color="black", rasterized=True, label="measured")
        tt = np.linspace(ev.t.min(), ev.t.max(), 600)
        top.plot(tt, _predicted(result, best, k, tt, c, wind), color=SERIES[1], lw=1.0, label="ADS-B, estimate")
        if alt is not None:
            top.plot(tt, _predicted(result, alt, k, tt, c, wind), color=SERIES[0], lw=0.9, ls="--",
                     label="ADS-B, runner-up")
        res = ev.f - _predicted(result, best, k, ev.t, c, wind)
        bot.plot(ev.t, res, "o", ms=1.0, mew=0, color="black", rasterized=True)
        bot.axhline(0, color=SERIES[1], lw=0.8)
        lim = max(1.0, float(np.nanpercentile(np.abs(res), 99)) * 1.2)
        bot.set_ylim(-lim, lim)
        rms = float(np.sqrt(np.nanmean(res**2)))
        bot.text(0.97, 0.9, f"rms {rms:.2f} Hz", transform=bot.transAxes, ha="right", va="top", fontsize=7)
        tr = result.tracks[best["track_ids"][k]]
        top.set_title(f"{ev.name}  ↔  {tr.label} ({tr.type_code})", fontsize=8.5)
        bot.set_xlabel("Time in recording (s)")
        if j == 0:
            top.set_ylabel(r"$f / h$ (Hz)")
            bot.set_ylabel("Residual (Hz)")
            top.legend(loc="lower left", markerscale=3, fontsize=6.5)
        _panel(top, "abcdef"[j])
    return fig


# ---------------------------------------------------------------- Fig. 8  Monte Carlo

def fig_monte_carlo(df):
    """df columns: seed, error_m, ambiguous (bool)."""
    use_style()
    fig = Figure(figsize=(W1, 2.4), layout="constrained")
    ax = fig.subplots()
    ok = ~df["ambiguous"].to_numpy(bool)
    ax.semilogy(df["seed"][ok], df["error_m"][ok], "o", color="black", ms=4, label="unambiguous")
    if (~ok).any():
        ax.semilogy(df["seed"][~ok], df["error_m"][~ok], "o", mfc="white", mec=SERIES[1], ms=4.5,
                    label="flagged ambiguous")
    med = float(np.median(df["error_m"][ok])) if ok.any() else np.nan
    ax.axhline(med, color="#9e9e9e", lw=0.6, ls="--")
    ax.text(df["seed"].min() - 0.4, med, f" median {med:.0f} m", va="bottom", ha="left", fontsize=7, color="#606060")
    ax.axhline(200, color="#9e9e9e", lw=0.6, ls=":")
    ax.set_xticks(df["seed"])
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlabel("Scenario (random seed)")
    ax.set_ylabel("Localisation error (m)")
    ax.legend(loc="upper left", fontsize=6.5)
    return fig


# ---------------------------------------------------------------- captions

def captions(analysis, result, truth=None, win_s=2.0, hop_s=0.5):
    good = [f for f in analysis.flyovers if f.quality == "ok"]
    best = result.best
    err = ""
    if truth is not None:
        err = f" The true receiver position (+) lies {np.hypot(truth['e'] - best['e'], truth['n'] - best['n']):.0f} m from the estimate (×)."
    return {
        "fig1": ("Figure 1. Received blade-passage frequency for a straight, level flyover with f₀ = 85 Hz. "
                 "(a) The slope at closest approach scales as 1/d. (b) The total frequency excursion scales with v/c. "
                 "(c) Retarded-time model versus the instantaneous model, which evaluates the geometry at the "
                 "arrival time; (d) their difference."),
        "fig2": (f"Figure 2. (a) Power spectral density of the {analysis.spec.t[-1] / 60:.0f}-min recording "
                 f"({win_s:g}-s Hann window, {hop_s:g}-s hop). (b) Spectral contrast after per-bin and per-frame "
                 "median normalisation. Coloured markers: traced tonal ridges of the usable flyovers "
                 f"({', '.join(f.name for f in good)}); lines: fitted retarded-time Doppler model for each harmonic; "
                 "grey markers: ridges rejected as partial, duplicate or implausible."),
        "fig3": ("Figure 3. Single-flyover fits. Top: ridge frequencies divided by harmonic number (dots) and the "
                 "fitted retarded-time model; parameter uncertainties are 1σ, inflated by the window/hop "
                 "oversampling factor. Bottom: residuals."),
        "fig4": ("Figure 4. Number of candidate 200-m grid cells remaining after each cumulative constraint: "
                 + "; ".join(f"{f[1]} = {f[0]}" for f in result.funnel[1:-1]) + "."),
        "fig5": ("Figure 5. Spatial constraints. (a) Propeller-aircraft ADS-B tracks in the search window; tracks "
                 "matched to the observed flyovers are coloured. (b) Cells admissible for the first flyover "
                 "(dashed box: extent of c and d). (c) Cells consistent with all flyovers under a common clock "
                 "epoch. (d) Cells surviving the silence constraint, coloured by RMS Doppler misfit." + err),
        "fig6": (f"Figure 6. RMS Doppler misfit as a function of receiver position. (a) 2 km × 2 km around "
                 f"the estimate (dashed box: extent of b). (b) Zoom with approximate 68 % and 95 % confidence "
                 f"regions (Δχ² = 2.30, 6.18; residuals de-weighted by the spectrogram "
                 f"window/hop ratio); equivalent radii {result.surface['radius68_m']:.0f} m and "
                 f"{result.surface['radius95_m']:.0f} m. These regions reflect measurement noise only, not "
                 "model error (wind structure, refraction, ADS-B latency)." + err),
        "fig7": ("Figure 7. Measured ridge frequencies (dots) against the frequencies predicted from the matched "
                 "ADS-B trajectories at the estimated receiver position (solid) and at the best-scoring "
                 "alternative more than 500 m away (dashed). Bottom: residuals at the estimate."),
    }


# ---------------------------------------------------------------- KML

def export_kml(result, frame, path, truth=None):
    def coords(enu):
        lat, lon, h = frame.to_geodetic(enu)
        return " ".join(f"{lo:.6f},{la:.6f},{hh:.0f}" for la, lo, hh in zip(lat, lon, h))

    best = result.best
    used = set(best["track_ids"])
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             "<name>doppler_geo result</name>",
             '<Style id="hit"><LineStyle><color>ff2b39c0</color><width>3</width></LineStyle></Style>',
             '<Style id="other"><LineStyle><color>66b0b0b0</color><width>1</width></LineStyle></Style>',
             '<Style id="ring"><LineStyle><color>ffffffff</color><width>2</width></LineStyle>'
             '<PolyStyle><color>33ffffff</color></PolyStyle></Style>']
    parts.append(f'<Placemark><name>estimate ({best["rms_hz"]:.2f} Hz)</name>'
                 f'<Point><coordinates>{best["lon"]:.6f},{best["lat"]:.6f},0</coordinates></Point></Placemark>')
    r68 = result.surface["radius68_m"]
    if r68 > 0:
        a = np.linspace(0, 2 * np.pi, 73)
        ring = np.stack([best["e"] + r68 * np.cos(a), best["n"] + r68 * np.sin(a), np.full_like(a, best["up"])], -1)
        parts.append(f'<Placemark><name>68% region</name><styleUrl>#ring</styleUrl><Polygon><outerBoundaryIs>'
                     f'<LinearRing><coordinates>{coords(ring)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
    if truth is not None:
        parts.append(f'<Placemark><name>true position</name>'
                     f'<Point><coordinates>{truth["lon"]:.6f},{truth["lat"]:.6f},0</coordinates></Point></Placemark>')
    for c in result.candidates[1:]:
        parts.append(f'<Placemark><name>alternative ({c["rms_hz"]:.2f} Hz)</name>'
                     f'<Point><coordinates>{c["lon"]:.6f},{c["lat"]:.6f},0</coordinates></Point></Placemark>')
    for i, tr in enumerate(result.tracks):
        style = "hit" if i in used else "other"
        p = tr.pos[:: max(1, int(5 / tr.dt))]
        parts.append(f'<Placemark><name>{tr.label} {tr.type_code}</name><styleUrl>#{style}</styleUrl>'
                     f'<LineString><altitudeMode>absolute</altitudeMode><coordinates>{coords(p)}</coordinates>'
                     f'</LineString></Placemark>')
    parts.append("</Document></kml>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def make_all(out, analysis, result, frame, c, truth=None, wind=None, log=print):
    """Write fig1..fig7 (PDF + PNG), captions.md and result.kml into <out>/figures."""
    figdir = Path(out) / "figures"
    figs = {
        "fig1_doppler_model": fig_doppler_model(c),
        "fig2_spectrogram": fig_spectrogram(analysis.spec, analysis.flyovers),
        "fig3_flyover_fits": fig_flyover_fits(analysis.flyovers),
        "fig4_candidate_reduction": fig_reduction(result.funnel),
        "fig5_spatial_constraints": fig_constraints(result, truth),
        "fig6_misfit_surface": fig_misfit_surface(result, truth),
        "fig7_adsb_match": fig_adsb_match(result, c, wind),
    }
    for stem, fig in figs.items():
        save(fig, figdir / stem)
        log(f"[figure]  figures/{stem}.pdf/.png")
    caps = captions(analysis, result, truth)
    (figdir / "captions.md").write_text("\n\n".join(caps.values()) + "\n", encoding="utf-8")
    export_kml(result, frame, Path(out) / "result.kml", truth)
    log("[figure]  figures/captions.md, result.kml")
