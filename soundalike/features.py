"""Audio features: a numpy re-implementation of Kaldi-style log-mel filterbanks
(`torchaudio.compliance.kaldi.fbank`) so no PyTorch is needed at runtime."""
from __future__ import annotations

import functools

import numpy as np

EPS = float(np.finfo(np.float32).eps)


def _mel(f):
    return 1127.0 * np.log1p(f / 700.0)


@functools.lru_cache(maxsize=8)
def _mel_banks(num_bins: int, n_fft: int, sr: int, low: float, high: float) -> np.ndarray:
    """Kaldi get_mel_banks: triangular filters evenly spaced on the mel scale.
    Returns (num_bins, n_fft // 2 + 1) with a zero Nyquist column."""
    nyq = sr / 2.0
    if high <= 0:
        high += nyq
    n_bins_fft = n_fft // 2
    fft_bin_width = sr / n_fft
    mel_low, mel_high = _mel(low), _mel(high)
    delta = (mel_high - mel_low) / (num_bins + 1)
    b = np.arange(num_bins)[:, None]
    left = mel_low + b * delta
    center = mel_low + (b + 1.0) * delta
    right = mel_low + (b + 2.0) * delta
    mel = _mel(fft_bin_width * np.arange(n_bins_fft))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    bank = np.maximum(0.0, np.minimum(up, down))
    return np.pad(bank, ((0, 0), (0, 1))).astype(np.float32)


@functools.lru_cache(maxsize=4)
def _window(kind: str, n: int) -> np.ndarray:
    k = np.arange(n)
    if kind == "hamming":
        return (0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))).astype(np.float32)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))
    if kind == "hanning":
        return hann.astype(np.float32)
    if kind == "povey":
        return (hann**0.85).astype(np.float32)
    raise ValueError(kind)


def kaldi_fbank(
    wave: np.ndarray,
    sr: int = 16000,
    num_mels: int = 80,
    window: str = "hamming",
    frame_len_ms: float = 25.0,
    shift_ms: float = 10.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    preemph: float = 0.97,
    scale: float = 1.0,
) -> np.ndarray:
    """Log mel filterbank, shape (frames, num_mels), float32. Mirrors Kaldi with
    snip_edges=True, dither=0, remove_dc_offset=True, power spectrum."""
    wave = np.asarray(wave, dtype=np.float32) * np.float32(scale)
    win = int(round(sr * frame_len_ms / 1000.0))
    hop = int(round(sr * shift_ms / 1000.0))
    if wave.size < win:
        return np.zeros((0, num_mels), np.float32)
    n_frames = 1 + (wave.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wave[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)  # Kaldi replicates sample 0
    frames = frames - preemph * prev
    frames = frames * _window(window, win)[None, :]
    n_fft = 1 << (win - 1).bit_length()
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = power.astype(np.float32) @ _mel_banks(num_mels, n_fft, sr, low_freq, high_freq).T
    return np.log(np.maximum(mel, EPS)).astype(np.float32)
