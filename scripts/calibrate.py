"""Calibrate similarity thresholds from REAL singers (VocalSet).

Builds ~clip-level embeddings (mean of 3 s windows over 12-25 s crops, like the pipeline) for every
recording, then looks at cosine similarity of same-singer vs different-singer pairs
(recordings of different songs/exercises only) and at window-level similarity inside one recording.

Prints: distributions, the thresholds achieving target error rates, and a logistic mapping
cosine -> P(same singer) that the UI uses for its "match %" display.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundalike.audio import load_audio  # noqa: E402
from soundalike.embed import VoiceEmbedder, l2norm  # noqa: E402

emb = VoiceEmbedder("models/wespeaker_campplus_lm.onnx")
files = sorted(glob.glob("data/testdata/vocalset/*/excerpts/*/*.wav"))
rows = []
for p in files:
    parts = p.split("/")
    singer, tech = parts[-4], parts[-2]
    song = re.sub(r"^[fm]\d+_", "", parts[-1])[:-4].split("_")[0]
    w = load_audio(p)[: 16000 * 40]
    _, E = emb.embed_windows(w, 3.0, 1.5)
    if len(E) < 2:
        continue
    rows.append((singer, song, tech, E))
print(len(rows), "recordings")
# window-level within-recording similarity (same singer, same song)
within = np.concatenate([(E @ E.T)[np.triu_indices(len(E), 1)] for _, _, _, E in rows])
# window-level across singers (different people)
rng = np.random.default_rng(0)
across = []
for _ in range(20000):
    a, b = rng.integers(len(rows), size=2)
    if rows[a][0] == rows[b][0]:
        continue
    ea, eb = rows[a][3], rows[b][3]
    across.append(ea[rng.integers(len(ea))] @ eb[rng.integers(len(eb))])
across = np.array(across)
# window-level same singer, DIFFERENT song
same_diffsong = []
for _ in range(20000):
    a, b = rng.integers(len(rows), size=2)
    if rows[a][0] != rows[b][0] or rows[a][1] == rows[b][1]:
        continue
    ea, eb = rows[a][3], rows[b][3]
    same_diffsong.append(ea[rng.integers(len(ea))] @ eb[rng.integers(len(eb))])
same_diffsong = np.array(same_diffsong)
q = lambda x: np.round(np.percentile(x, [1, 5, 25, 50, 75, 95, 99]), 3)  # noqa: E731
print("percentiles [1,5,25,50,75,95,99]")
print("  window  same singer, same recording :", q(within))
print("  window  same singer, other song      :", q(same_diffsong))
print("  window  different singers            :", q(across))

# clip-level (mean of windows) pairs across recordings
C = [(s, so, l2norm(E.mean(0))) for s, so, _, E in rows]
same, diff = [], []
for i in range(len(C)):
    for j in range(i + 1, len(C)):
        if C[i][1] == C[j][1]:
            continue
        (same if C[i][0] == C[j][0] else diff).append(float(C[i][2] @ C[j][2]))
same, diff = np.array(same), np.array(diff)
print("  clip    same singer, other song      :", q(same), len(same))
print("  clip    different singers            :", q(diff), len(diff))
for far in (0.01, 0.02, 0.05):
    t = np.quantile(diff, 1 - far)
    print(f"  clip-level threshold with {far:.0%} false-merge rate: {t:.3f} -> recall of same-singer pairs {np.mean(same >= t):.3f}")
for far in (0.01, 0.02, 0.05):
    t = np.quantile(across, 1 - far)
    print(f"  window-level: {far:.0%} of different-singer windows exceed {t:.3f}; same-singer other-song windows kept: {np.mean(same_diffsong >= t):.3f}")

# logistic calibration cosine -> P(same singer) on clip pairs (balanced)
x = np.concatenate([same, diff]); y = np.concatenate([np.ones(len(same)), np.zeros(len(diff))])
w = np.concatenate([np.full(len(same), 0.5 / len(same)), np.full(len(diff), 0.5 / len(diff))])
a, b = 10.0, -6.0
for _ in range(5000):
    p = 1 / (1 + np.exp(-(a * x + b)))
    ga = np.sum(w * (p - y) * x); gb = np.sum(w * (p - y))
    a -= 5 * ga; b -= 5 * gb
print(f"calibration: P(same singer) = sigmoid({a:.2f} * cos + {b:.2f})")
json.dump({"a": a, "b": b}, open("data/calibration.json", "w"))
