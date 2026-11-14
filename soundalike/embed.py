"""Speaker/voice-timbre embeddings via ONNX (WeSpeaker). No PyTorch needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import kaldi_fbank

SR = 16000


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class VoiceEmbedder:
    """Maps a mono 16 kHz waveform to L2-normalised voice embeddings."""

    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        self.sess = _session(self.path)
        self.dim = self.sess.get_outputs()[0].shape[-1]

    def embed_waveform(self, wave: np.ndarray) -> np.ndarray:
        """One embedding for the whole waveform (any length >= ~0.5 s)."""
        fb = kaldi_fbank(wave, SR, num_mels=80, window="hamming", scale=32768.0)
        if fb.shape[0] < 30:
            raise ValueError("audio too short to embed")
        fb = fb - fb.mean(axis=0, keepdims=True)  # CMN, as in WeSpeaker
        e = self.sess.run(None, {"feats": fb[None]})[0][0]
        n = np.linalg.norm(e)
        return (e / max(n, 1e-9)).astype(np.float32)

    def embed_windows(self, wave: np.ndarray, win_s: float = 4.0, hop_s: float = 2.0, min_s: float = 1.5):
        """Sliding-window embeddings -> (starts_seconds, matrix[n, dim])."""
        win, hop = int(win_s * SR), int(hop_s * SR)
        starts = list(range(0, max(1, wave.size - win + 1), hop))
        if wave.size > win and (starts[-1] + win) < wave.size - int(min_s * SR):
            starts.append(wave.size - win)
        out, keep = [], []
        for s in starts:
            seg = wave[s : s + win]
            if seg.size < int(min_s * SR):
                continue
            out.append(self.embed_waveform(seg))
            keep.append(s / SR)
        if not out:
            return np.zeros(0), np.zeros((0, self.dim), np.float32)
        return np.array(keep), np.stack(out)


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-9)
