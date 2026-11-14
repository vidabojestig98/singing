"""Pitch-stability features: what physically separates singing from speech.

Speech pitch glides continuously; singing holds notes (long steady pitch runs, often with vibrato).
A vectorised YIN tracker gives f0 at 100 fps; `window_features` summarises it over each
YAMNet frame (0.96 s) so it can be appended to the embedding before the classifier head.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 1024          # 64 ms analysis frame
WIN = 640             # difference-function window (40 ms)
HOPF = 160            # 10 ms
FMIN, FMAX = 70.0, 1000.0
TAU_MIN, TAU_MAX = int(SR / FMAX), int(SR / FMIN)       # 16 .. 228
N_FEATS = 8
FRAMES_PER_WIN = 96   # 0.96 s of 10 ms frames
HOP_FRAMES = 48       # 0.48 s


def yin(wave: np.ndarray):
    """Returns (f0_hz[n], harmonicity[n]) at 100 fps. f0 = 0 where unvoiced."""
    wave = np.asarray(wave, np.float32)
    if wave.size < FRAME + HOPF:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    n = 1 + (wave.size - FRAME) // HOPF
    out_f0 = np.zeros(n, np.float32)
    out_h = np.zeros(n, np.float32)
    B = 2048                                    # frames per batch (bounded memory)
    taus = np.arange(TAU_MAX + 1)
    for b0 in range(0, n, B):
        idx = np.arange(FRAME)[None, :] + HOPF * np.arange(b0, min(n, b0 + B))[:, None]
        x = wave[idx].astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        a = x[:, :WIN]
        # r(tau) = sum_j a_j * x_{j+tau}  via FFT cross-correlation
        fa = np.fft.rfft(a, n=2048, axis=1)
        fx = np.fft.rfft(x, n=2048, axis=1)
        r = np.fft.irfft(np.conj(fa) * fx, n=2048, axis=1)[:, : TAU_MAX + 1]
        e0 = (a**2).sum(axis=1, keepdims=True)
        cs = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x**2, axis=1)], axis=1)
        e_tau = cs[:, WIN + taus] - cs[:, taus]                       # energy of x[tau : tau+WIN]
        d = e0 + e_tau - 2 * r
        d[:, 0] = 0
        csum = np.cumsum(d, axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(csum[:, 1:], 1e-12)
        sub = cmnd[:, TAU_MIN : TAU_MAX]
        k = sub.argmin(axis=1)
        mn = sub[np.arange(sub.shape[0]), k]
        # first dip under threshold is more octave-safe than the global minimum
        below = sub < 0.2
        first = below.argmax(axis=1)
        has = below.any(axis=1)
        k = np.where(has, first, k)
        # slide to the local minimum following the first crossing
        for _ in range(8):
            nxt = np.minimum(k + 1, sub.shape[1] - 1)
            k = np.where(has & (sub[np.arange(len(k)), nxt] < sub[np.arange(len(k)), k]), nxt, k)
        mn = sub[np.arange(sub.shape[0]), k]
        tau = (k + TAU_MIN).astype(np.float64)
        # parabolic interpolation
        i = np.clip(k + TAU_MIN, 1, TAU_MAX - 1)
        ar = np.arange(len(i))
        y0, y1, y2 = cmnd[ar, i - 1], cmnd[ar, i], cmnd[ar, i + 1]
        den = y0 - 2 * y1 + y2
        shift = np.divide(0.5 * (y0 - y2), den, out=np.zeros_like(den), where=np.abs(den) > 1e-9)
        tau = np.where(np.abs(tau - i) < 1, i + np.clip(shift, -1, 1), tau)
        voiced = (mn < 0.3) & (e0[:, 0] > 1e-7 * WIN)
        out_f0[b0 : b0 + len(k)] = np.where(voiced, SR / tau, 0.0)
        out_h[b0 : b0 + len(k)] = np.clip(1.0 - mn, 0, 1)
    return out_f0, out_h


def _runs(mask: np.ndarray):
    """(start, length) of True runs."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s, e - s))


def window_features(wave: np.ndarray, n_windows: int) -> np.ndarray:
    """(n_windows, N_FEATS) float32, aligned with YAMNet frames (hop 0.48 s, window 0.96 s)."""
    f0, harm = yin(wave)
    F = np.zeros((n_windows, N_FEATS), np.float32)
    semis = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1.0) / 55.0), np.nan)
    for w in range(n_windows):
        a, b = w * HOP_FRAMES, w * HOP_FRAMES + FRAMES_PER_WIN
        s = semis[a:b]
        if s.size < 10:
            continue
        v = ~np.isnan(s)
        F[w, 0] = v.mean()                                           # voiced fraction
        F[w, 1] = harm[a:b].mean()
        if v.sum() < 6:
            continue
        dsemi = np.abs(np.diff(s))
        ok = ~np.isnan(dsemi)
        F[w, 2] = np.median(dsemi[ok]) if ok.any() else 0.0           # typical frame-to-frame glide
        steady = np.zeros(s.size, bool)
        steady[1:] = ok & (dsemi < 0.35)                              # < 35 cents per 10 ms
        steady[:-1] |= (ok & (dsemi < 0.35))
        runs = _runs(steady)
        F[w, 3] = sum(l for _, l in runs if l >= 12) / s.size          # in steady runs >= 120 ms
        F[w, 4] = sum(l for _, l in runs if l >= 25) / s.size          # in steady runs >= 250 ms
        F[w, 5] = (max([l for _, l in runs]) if runs else 0) / FRAMES_PER_WIN
        F[w, 6] = np.nanstd(s)                                        # pitch spread (semitones)
        F[w, 7] = np.nanmedian(s) / 40.0                              # register (female/male/child)
    return F
