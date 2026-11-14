"""Segment-level evaluation of the SHIPPED detector on real songs (MUSDB18 test, never used for training):
the app's own segmentation (smoothing + hysteresis + gap merge + min length) vs. frames where the isolated
vocal stem is audible."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from eval_singing_detection import decode_stream, prf, vocal_truth
from soundalike.pipeline import Params
from soundalike.singing import HOP, Head, SingingDetector, Yamnet, frame_features

yam = Yamnet("models/yamnet.onnx", "models/yamnet_class_map.csv")
head = Head.load("soundalike/assets/singing_head.npz")
det, pr = SingingDetector(yam, head), Params()
P, T, per, nseg = [], [], [], 0
for p in sorted(glob.glob("data/testdata/musdb/*.mp4")):
    mix, voc = decode_stream(p, 0), decode_stream(p, 4)
    _, X = frame_features(yam, mix)
    pf = head.proba(X)[:, 0]
    truth = vocal_truth(voc, len(pf)); t = np.arange(len(pf)) * HOP + HOP
    segs = det.segment(pf, on=pr.sing_on, off=pr.sing_off, min_len=pr.min_segment, max_gap=pr.max_gap)
    pred = np.zeros(len(pf), bool)
    for s in segs:
        pred |= (t >= s.start) & (t <= s.end)
    P.append(pred); T.append(truth); per.append(prf(pred, truth)[2]); nseg += len(segs)
p_, r_, f_ = prf(np.concatenate(P), np.concatenate(T))
print(f"songs={len(per)} frames={sum(map(len, T))}  precision {p_:.3f}  recall {r_:.3f}  F1 {f_:.3f}  segments={nseg}")
print("per-song F1:", " ".join(f"{x:.2f}" for x in per))
