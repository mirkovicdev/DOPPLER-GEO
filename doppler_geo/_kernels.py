"""Hot loops, compiled with numba when available (pure-Python fallback is correct but slow)."""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def rts_cv(t, zp, zv, has_v, rp, rv, q):
    """Rauch-Tung-Striebel smoother, 1-axis constant-velocity model, irregular sampling.

    Fuses position measurements zp (variance rp) with optional velocity measurements zv
    (variance rv) under white-acceleration process noise q (m^2/s^3). Returns (n, 2): [pos, vel].
    """
    n = t.size
    xf = np.empty((n, 2))
    pf = np.empty((n, 3))   # P00, P01, P11 after update
    xpr = np.empty((n, 2))
    ppr = np.empty((n, 3))  # predicted
    x0 = zp[0]
    x1 = zv[0] if has_v[0] else 0.0
    p00 = rp
    p01 = 0.0
    p11 = rv if has_v[0] else 1.0e4
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            x0 = x0 + dt * x1
            p00 = p00 + 2.0 * dt * p01 + dt * dt * p11 + q * dt ** 3 / 3.0
            p01 = p01 + dt * p11 + q * dt * dt / 2.0
            p11 = p11 + q * dt
        xpr[k, 0] = x0
        xpr[k, 1] = x1
        ppr[k, 0] = p00
        ppr[k, 1] = p01
        ppr[k, 2] = p11
        if k > 0:
            s = p00 + rp
            k0 = p00 / s
            k1 = p01 / s
            y = zp[k] - x0
            x0 += k0 * y
            x1 += k1 * y
            p00, p01, p11 = (1.0 - k0) * p00, (1.0 - k0) * p01, p11 - k1 * p01
            if has_v[k]:
                s = p11 + rv
                k0 = p01 / s
                k1 = p11 / s
                y = zv[k] - x1
                x0 += k0 * y
                x1 += k1 * y
                p00, p01, p11 = p00 - k0 * p01, (1.0 - k1) * p01, (1.0 - k1) * p11
        xf[k, 0] = x0
        xf[k, 1] = x1
        pf[k, 0] = p00
        pf[k, 1] = p01
        pf[k, 2] = p11
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        dt = t[k + 1] - t[k]
        a00 = pf[k, 0] + dt * pf[k, 1]
        a01 = pf[k, 1]
        a10 = pf[k, 1] + dt * pf[k, 2]
        a11 = pf[k, 2]
        q00, q01, q11 = ppr[k + 1, 0], ppr[k + 1, 1], ppr[k + 1, 2]
        det = q00 * q11 - q01 * q01
        i00, i01, i11 = q11 / det, -q01 / det, q00 / det
        c00 = a00 * i00 + a01 * i01
        c01 = a00 * i01 + a01 * i11
        c10 = a10 * i00 + a11 * i01
        c11 = a10 * i01 + a11 * i11
        d0 = xs[k + 1, 0] - xpr[k + 1, 0]
        d1 = xs[k + 1, 1] - xpr[k + 1, 1]
        xs[k, 0] = xf[k, 0] + c00 * d0 + c01 * d1
        xs[k, 1] = xf[k, 1] + c10 * d0 + c11 * d1
    return xs


@njit(cache=True)
def find_passes(e0, n0, res, nx, ny, up, tau, px, py, pz, vx, vy, vz, rmax, c, track_id,
                o_cell, o_track, o_tau, o_tarr, o_r, o_speed, count):
    """Find every closest approach (local minimum of 3D slant range) of one track to every grid cell.

    The grid is regular in (east, north) with ground height `up[iy, ix]`. Each pass is refined
    with a parabola through the three samples around the minimum. Returns the new pass count
    (may exceed the buffer size, in which case the caller re-runs with bigger buffers).
    """
    nk = tau.size
    if nk < 3:
        return count
    step = tau[1] - tau[0]
    ir = int(rmax / res) + 1
    rmax2 = rmax * rmax
    cap = o_cell.size
    for k in range(1, nk - 1):
        vh = math.sqrt(vx[k] * vx[k] + vy[k] * vy[k])
        margin = vh * step + 2.0 * res + 300.0
        cx = int(round((px[k] - e0) / res))
        cy = int(round((py[k] - n0) / res))
        if cx < -ir or cx >= nx + ir or cy < -ir or cy >= ny + ir:
            continue
        for iy in range(max(0, cy - ir), min(ny, cy + ir + 1)):
            gy = n0 + iy * res
            dy = gy - py[k]
            for ix in range(max(0, cx - ir), min(nx, cx + ir + 1)):
                gx = e0 + ix * res
                dx = gx - px[k]
                dh2 = dx * dx + dy * dy
                if dh2 > rmax2:
                    continue
                if vh > 1.0:
                    along = (dx * vx[k] + dy * vy[k]) / vh
                    if along > margin or along < -margin:
                        continue
                gz = up[iy, ix]
                dz = gz - pz[k]
                r2 = dh2 + dz * dz
                if r2 > rmax2:
                    continue
                ax = gx - px[k - 1]
                ay = gy - py[k - 1]
                az = gz - pz[k - 1]
                r2m = ax * ax + ay * ay + az * az
                if r2 > r2m:
                    continue
                bx = gx - px[k + 1]
                by = gy - py[k + 1]
                bz = gz - pz[k + 1]
                r2p = bx * bx + by * by + bz * bz
                if r2 >= r2p:
                    continue
                rm = math.sqrt(r2m)
                r0 = math.sqrt(r2)
                rp = math.sqrt(r2p)
                den = rm - 2.0 * r0 + rp
                off = 0.0
                if den > 0.0:
                    off = 0.5 * (rm - rp) / den
                    off = min(0.5, max(-0.5, off))
                rmin = r0 - 0.25 * (rm - rp) * off
                tmin = tau[k] + off * step
                if count < cap:
                    o_cell[count] = iy * nx + ix
                    o_track[count] = track_id
                    o_tau[count] = tmin
                    o_tarr[count] = tmin + rmin / c
                    o_r[count] = rmin
                    o_speed[count] = math.sqrt(vh * vh + vz[k] * vz[k])
                count += 1
    return count


@njit(cache=True)
def count_violations(h_cell, h_lo, h_hi, h_t0, h_rthr, h_ids, h_track, h_tarr,
                     cell_start, p_track, p_tarr, p_r, heard_lo, heard_hi, excl_dt):
    """Silence constraint. For each hypothesis count the ADS-B passes at its cell that
    (a) arrive inside the checked window, (b) come closer than the threshold slant range,
    (c) are not one of the matched passes, and (d) do not coincide with any detected tone ridge
    (heard_lo/hi, recording-relative). Such aircraft should have been audible but were not detected."""
    nh = h_cell.size
    m = h_ids.shape[1]
    nheard = heard_lo.size
    out = np.zeros(nh, np.int64)
    for h in range(nh):
        cidx = h_cell[h]
        cnt = 0
        for q in range(cell_start[cidx], cell_start[cidx + 1]):
            ta = p_tarr[q]
            if ta < h_lo[h] or ta > h_hi[h]:
                continue
            if p_r[q] >= h_rthr[h]:
                continue
            skip = False
            for j in range(m):
                if q == h_ids[h, j] or (p_track[q] == h_track[h, j] and abs(ta - h_tarr[h, j]) < excl_dt):
                    skip = True
                    break
            if not skip:
                rel = ta - h_t0[h]
                for j in range(nheard):
                    if heard_lo[j] <= rel <= heard_hi[j]:
                        skip = True
                        break
            if not skip:
                cnt += 1
        out[h] = cnt
    return out
