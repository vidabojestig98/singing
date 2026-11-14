import numpy as np

from soundalike.features import kaldi_fbank
from soundalike.pitch import window_features, yin
from soundalike.singing import HOP, SingingDetector


def test_fbank_shape_and_tone_energy_peak():
    sr = 16000
    t = np.arange(sr) / sr
    fb = kaldi_fbank(np.sin(2 * np.pi * 1000 * t), sr, num_mels=80)
    assert fb.shape == (1 + (sr - 400) // 160, 80)
    # mel bin of 1 kHz should hold the most energy; 1127*ln(1+f/700)
    mel = lambda f: 1127 * np.log1p(f / 700)  # noqa: E731
    expected = (mel(1000) - mel(20)) / ((mel(8000) - mel(20)) / 81) - 1
    assert abs(int(fb.mean(0).argmax()) - expected) <= 2


def test_yin_tracks_pitch_and_flags_noise_unvoiced():
    sr = 16000
    t = np.arange(sr * 2) / sr
    f0, _ = yin(np.concatenate([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)]).astype(np.float32))
    assert abs(np.median(f0[20:180]) - 220) < 2 and abs(np.median(f0[220:380]) - 330) < 3
    noise = np.random.default_rng(0).standard_normal(sr * 2).astype(np.float32) * 0.1
    f0n, _ = yin(noise)
    assert (f0n > 0).mean() < 0.15


def test_pitch_features_separate_held_notes_from_glides():
    sr = 16000
    t = np.arange(sr * 6) / sr
    held = np.sin(2 * np.pi * np.repeat([220, 262, 330, 262, 220, 196], sr) * t).astype(np.float32)   # notes
    phase = 2 * np.pi * np.cumsum(150 + 80 * np.sin(2 * np.pi * 1.7 * t)) / sr                         # constant glide
    glide = np.sin(phase).astype(np.float32)
    n = 1 + (len(t) - 15360) // 7680
    fh, fg = window_features(held, n), window_features(glide, n)
    assert fh[:, 4].mean() > 0.5 > fg[:, 4].mean()        # steady >= 250 ms fraction


def test_segmentation_hysteresis_gap_merge_and_min_len():
    p = np.zeros(100)
    p[10:30] = 0.9          # 20 frames  ~ 9.6 s
    p[33:50] = 0.9          # gap of 3 frames (1.44 s) -> merged
    p[60:62] = 0.95         # too short
    p[80:90] = 0.45         # never crosses `on`
    segs = SingingDetector.segment(p, on=0.55, off=0.35, min_len=2.5, max_gap=1.5)
    assert len(segs) == 1
    assert abs(segs[0].start - (10 * HOP + 0.24)) < 0.6 and segs[0].end > 49 * HOP
    assert SingingDetector.segment(np.zeros(0)) == []
