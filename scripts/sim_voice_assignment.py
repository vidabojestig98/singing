"""Simulate sequential voice assignment on REAL singers: how many voices appear, and how pure are they?
rules: centroid (mean of members)  |  max (best single clip)  |  top2 (mean of the best two clips of a voice)"""
import glob, os, re, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from soundalike.audio import load_audio
from soundalike.embed import VoiceEmbedder, l2norm
emb = VoiceEmbedder("models/wespeaker_campplus_lm.onnx")
rows = []
for p in sorted(glob.glob("data/testdata/vocalset/*/excerpts/*/*.wav")):
    parts = p.split("/")
    if parts[-2] == "spoken": continue
    _, E = emb.embed_windows(load_audio(p)[:16000*40], 3.0, 1.5)
    if len(E) > 1: rows.append((parts[-4], l2norm(E.mean(0))))
print(len(rows), "recordings,", len({r[0] for r in rows}), "singers")
def run(rule, thr, seed):
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    voices = []                      # list of lists of row indices
    for i in order:
        e = rows[i][1]; best, bs = None, -1
        for vi, members in enumerate(voices):
            sims = np.array([float(e @ rows[j][1]) for j in members])
            if rule == "centroid": s = float(e @ l2norm(np.mean([rows[j][1] for j in members], 0)))
            elif rule == "max": s = sims.max()
            else: s = np.sort(sims)[-2:].mean()
            if s > bs: best, bs = vi, s
        if best is not None and bs >= thr: voices[best].append(i)
        else: voices.append([i])
    pure = sum(max(np.unique([rows[j][0] for j in m], return_counts=True)[1]) for m in voices) / len(rows)
    return len(voices), pure
print("%-9s %5s | %8s %8s   (20 singers = ideal)" % ("rule", "thr", "voices", "purity"))
for rule in ("centroid", "max", "top2"):
    for thr in (0.60, 0.65, 0.70, 0.75, 0.80):
        r = [run(rule, thr, s) for s in range(3)]
        print("%-9s %5.2f | %8.1f %8.3f" % (rule, thr, np.mean([x[0] for x in r]), np.mean([x[1] for x in r])))
