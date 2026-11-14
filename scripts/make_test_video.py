"""Assemble a test VIDEO from real recordings with a known timeline (ground truth).

  speech -> instrumental -> singer A (song 1) -> speech -> singer B -> instrumental
         -> singer A over music -> silence

Writes <out>.mp4 (H.264 + AAC, moving test picture) and <out>.json (the timeline).
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundalike.audio import encode_wav, load_audio  # noqa: E402

SR = 16000
D = "data/testdata"


def seg(path, start, dur, gain=1.0):
    w = load_audio(path)[int(start * SR) : int((start + dur) * SR)]
    w = w / (np.abs(w).max() + 1e-9) * 0.7 * gain
    return w.astype(np.float32)


def build(out_base, singer_a="female1", singer_b="male3", song_a="caro", song_b="row", song_a2="dona"):
    pa = glob.glob(f"{D}/vocalset/{singer_a}/excerpts/vibrato/*{song_a}*.wav")[0]
    pa2 = glob.glob(f"{D}/vocalset/{singer_a}/excerpts/straight/*{song_a2}*.wav")[0]
    pb = glob.glob(f"{D}/vocalset/{singer_b}/excerpts/straight/*{song_b}*.wav")[0]
    sp = sorted(glob.glob(f"{D}/speech_big/clean_*.flac"))[:8]
    ins = sorted(glob.glob(f"{D}/music/instrumental/*.ogg"))
    parts = [
        ("speech", np.concatenate([seg(p, 0, 6) for p in sp[:2]])[: 10 * SR], None),
        ("instrumental", seg(ins[3], 2, 10), None),
        ("singing", seg(pa, 1, 20), singer_a),
        ("speech", np.concatenate([seg(p, 0, 6) for p in sp[3:5]])[: 8 * SR], None),   # LibriSpeech
        ("singing", seg(pb, 1, 20), singer_b),
        ("instrumental", seg(ins[7], 3, 8), None),
    ]
    # singer A over accompaniment (vocals ~ +4 dB over the band)
    v = seg(pa2, 1, 16)
    m = seg(ins[11], 0, 16)
    mixed = v + 0.6 * m * (np.sqrt((v**2).mean()) / np.sqrt((m**2).mean()))
    parts.append(("singing+music", mixed / np.abs(mixed).max() * 0.7, singer_a))
    parts.append(("silence", np.zeros(4 * SR, np.float32), None))
    wave = np.concatenate([p[1] for p in parts])
    timeline, t = [], 0.0
    for kind, w, who in parts:
        timeline.append({"kind": kind, "singer": who, "start": round(t, 2), "end": round(t + len(w) / SR, 2)})
        t += len(w) / SR
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    wav = out_base + ".wav"
    encode_wav(wave, wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=15:duration={t:.2f}",
                    "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-c:a", "aac", "-shortest",
                    out_base + ".mp4"], check=True)
    os.remove(wav)
    json.dump(timeline, open(out_base + ".json", "w"), indent=1)
    return out_base + ".mp4", timeline


if __name__ == "__main__":
    path, tl = build(sys.argv[1] if len(sys.argv) > 1 else "data/testdata/synth/video1")
    print(path)
    for x in tl:
        print(x)
