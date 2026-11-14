"""Honest evaluation of singing detection on REAL music mixes (MUSDB18 test songs).

Ground truth for "someone is singing" = the isolated vocal stem's loudness in the same
0.96 s window (>-40 dBFS; stems are cleanly bimodal, see README). The detector only ever sees
the full mixture. Also reports false-positive rates on speech and instrumental music.
"""
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundalike.audio import load_audio  # noqa: E402
from soundalike.singing import HOP_S, WIN_S, Head, SingingDetector, Yamnet, frame_features  # noqa: E402


def decode_stream(p, stream, sr=16000):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", f"0:{stream}", "-f", "f32le", "-ac", "1",
                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def vocal_truth(vocals, n):
    idx = np.arange(WIN_S)[None, :] + HOP_S * np.arange(n)[:, None]
    r = np.sqrt((vocals[idx] ** 2).mean(1) + 1e-12)
    return 20 * np.log10(r) > -40


def prf(pred, truth):
    tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    yam = Yamnet("models/yamnet.onnx", "models/yamnet_class_map.csv")
    head = Head.load("soundalike/assets/singing_head.npz") if os.path.exists("soundalike/assets/singing_head.npz") else None
    variants = {"native AudioSet only": SingingDetector(yam, None)}
    if head is not None:
        variants["trained head only"] = SingingDetector(yam, head, native_weight=0.0)
        variants["fused (shipped)"] = SingingDetector(yam, head)
    songs = sorted(glob.glob("data/testdata/musdb/*.mp4"))
    cache = []
    for p in songs:
        mixw, voc = decode_stream(p, 0), decode_stream(p, 4)
        scores, emb = frame_features(yam, mixw)
        cache.append((os.path.basename(p)[:40], scores, emb, vocal_truth(voc, len(scores))))
    print(f"MUSDB songs: {len(cache)}, frames: {sum(len(c[3]) for c in cache)}, "
          f"frames with vocals: {sum(c[3].sum() for c in cache)}")
    for name, det in variants.items():
        P, T = [], []
        per = []
        for song, scores, emb, truth in cache:
            native = scores[:, yam.sing_ix].max(1)
            if det.head is None:
                p = native
            else:
                p = np.maximum(det.head.proba(emb)[:, 0], det.native_weight * native)
            P.append(p); T.append(truth)
            pr, rc, f1 = prf(p > 0.5, truth)
            per.append(f"{f1:.2f}")
        P, T = np.concatenate(P), np.concatenate(T)
        print(f"\n[{name}]  per-song F1 @0.5: {' '.join(per)}")
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            pr, rc, f1 = prf(P > thr, T)
            print(f"   thr {thr:.1f}: precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}")
    # negatives / dry singing
    print("\nfalse-positive / recall on other real material (fraction of frames with P(singing)>0.5):")
    sets = {
        "speech (LibriSpeech)": sorted(glob.glob("data/testdata/speech/*.flac"))[:40],
        "instrumental music (FMA)": sorted(glob.glob("data/testdata/music/instrumental/*.ogg"))[::3],
        "VocalSet SUNG excerpts (head saw these singers)":
            sorted(glob.glob("data/testdata/vocalset/*/excerpts/straight/*.wav") + glob.glob("data/testdata/vocalset/*/excerpts/vibrato/*.wav")),
        "VocalSet SPOKEN excerpts (same singers talking the lyrics; should be ~0)":
            sorted(glob.glob("data/testdata/vocalset/*/excerpts/spoken/*.wav")),
    }
    for sname, files in sets.items():
        row = {}
        for name, det in variants.items():
            fr = []
            for f in files:
                p = det.frame_scores(load_audio(f)[: 16000 * 60])
                fr.append(p)
            row[name] = float((np.concatenate(fr) > 0.5).mean())
        print(f"  {sname}\n     " + "   ".join(f"{k}: {v:.3f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
