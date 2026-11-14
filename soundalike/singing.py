"""Singing detection: where in an audio track does someone sing?

Pipeline per track:
  1. YAMNet (AudioSet CNN, ONNX) -> 1024-d embedding + 521 class scores for every 0.96 s
     window at a 0.48 s hop.
  2. A tiny trained head (multinomial logistic regression on the embeddings, see
     scripts/train_singing_head.py) -> P(singing / speech / music / other) per frame.
     It is combined with YAMNet's own AudioSet "Singing/Choir/..." scores, because the
     off-the-shelf scores are good for accompanied singing while dry a-cappella voices
     (which AudioSet under-represents) are what the head adds.
  3. Smoothing + hysteresis -> segments.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
HOP = 0.48          # seconds between YAMNet frames
WIN = 0.96          # YAMNet analysis window
HOP_S, WIN_S = int(HOP * SR), int(WIN * SR)
CLASSES = ["singing", "speech", "music", "other"]
PITCH_BOOST = 4.0   # pitch-stability features are few but decisive; keep the L2 prior from drowning them

NATIVE_SINGING = ["Singing", "Choir", "Yodeling", "Child singing", "Synthetic singing", "Opera"]
NATIVE_SPEECHISH = ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue"]


def frame_features(yamnet: "Yamnet", wave: np.ndarray):
    """Per YAMNet frame: (AudioSet scores[N,521], head input[N, 1024+8] = embedding ++ pitch-stability)."""
    from .pitch import window_features

    scores, emb = yamnet.run(wave)
    pf = window_features(np.asarray(wave, np.float32), len(emb))
    return scores, np.concatenate([emb, pf], axis=1).astype(np.float32)


class Yamnet:
    def __init__(self, model_path: str | Path, class_map: str | Path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        names = [r[2] for r in list(csv.reader(open(class_map)))[1:]]
        self.names = names
        ix = {n: i for i, n in enumerate(names)}
        self.sing_ix = [ix[n] for n in NATIVE_SINGING if n in ix]
        self.speech_ix = [ix[n] for n in NATIVE_SPEECHISH if n in ix]

    def run(self, wave: np.ndarray, chunk_frames: int = 600):
        """Returns (scores[N,521], emb[N,1024]); frame i spans [i*HOP, i*HOP+WIN]."""
        wave = np.asarray(wave, dtype=np.float32)
        if wave.size < WIN_S:
            wave = np.pad(wave, (0, WIN_S - wave.size))
        n_frames = 1 + (wave.size - WIN_S) // HOP_S
        scores, embs = [], []
        for f0 in range(0, n_frames, chunk_frames):
            f1 = min(n_frames, f0 + chunk_frames)
            seg = wave[f0 * HOP_S : (f1 - 1) * HOP_S + WIN_S]
            s, e, _ = self.sess.run(None, {"waveform": seg})  # ORT sessions are thread-safe
            scores.append(s[: f1 - f0])
            embs.append(e[: f1 - f0])
        return np.concatenate(scores), np.concatenate(embs)


@dataclass
class Head:
    mean: np.ndarray
    std: np.ndarray
    W: np.ndarray      # (n_classes, dim)
    b: np.ndarray

    @staticmethod
    def load(path: str | Path) -> "Head":
        z = np.load(path)
        return Head(z["mean"], z["std"], z["W"], z["b"])

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std, W=self.W, b=self.b)

    def proba(self, emb: np.ndarray) -> np.ndarray:
        z = ((emb - self.mean) / self.std) @ self.W.T + self.b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


@dataclass
class Segment:
    start: float
    end: float
    score: float

    @property
    def dur(self) -> float:
        return self.end - self.start


class SingingDetector:
    def __init__(self, yamnet: Yamnet, head: Head | None, native_weight: float = 1.0):
        self.yamnet = yamnet
        self.head = head
        self.native_weight = native_weight

    def frame_scores(self, wave: np.ndarray) -> np.ndarray:
        """P(singing) per YAMNet frame."""
        scores, feats = frame_features(self.yamnet, wave)
        native = scores[:, self.yamnet.sing_ix].max(axis=1)
        if self.head is None:
            return native
        p_head = self.head.proba(feats)[:, 0]
        # "or"-style fusion: either detector being confident is enough
        return np.maximum(p_head, self.native_weight * native)

    @staticmethod
    def segment(
        p: np.ndarray,
        on: float = 0.55,
        off: float = 0.35,
        min_len: float = 2.5,
        max_gap: float = 1.5,
        smooth: int = 3,
    ) -> list[Segment]:
        if p.size == 0:
            return []
        if smooth > 1 and p.size >= smooth:
            k = smooth // 2
            padded = np.pad(p, k, mode="edge")
            p_s = np.array([np.median(padded[i : i + smooth]) for i in range(p.size)])
        else:
            p_s = p
        active, runs, start = False, [], 0
        for i, v in enumerate(p_s):
            if not active and v >= on:
                active, start = True, i
            elif active and v < off:
                active = False
                runs.append((start, i))
        if active:
            runs.append((start, p_s.size))
        # merge short gaps
        merged: list[list[int]] = []
        gap_frames = int(round(max_gap / HOP))
        for a, b in runs:
            if merged and a - merged[-1][1] <= gap_frames:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out = []
        for a, b in merged:
            # frame i spans [i*HOP, i*HOP+WIN]; its "own" central hop is [i*HOP+0.24, i*HOP+0.72]
            s, e = a * HOP + (WIN - HOP) / 2, (b - 1) * HOP + (WIN + HOP) / 2
            if e - s >= min_len:
                out.append(Segment(s, e, float(p_s[a:b].mean())))
        return out

    def detect(self, wave: np.ndarray, **kw) -> tuple[list[Segment], np.ndarray]:
        p = self.frame_scores(wave)
        return self.segment(p, **kw), p
