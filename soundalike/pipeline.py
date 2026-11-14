"""The engine: media file -> singing segments -> distinct voices -> clips in the database,
and query audio -> voice embedding -> best matches."""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .audio import SR, MediaInfo, load_audio, probe
from .clips import export_clip
from .db import Database
from .embed import VoiceEmbedder, l2norm
from .singing import HOP, Head, Segment, SingingDetector, Yamnet

log = logging.getLogger("soundalike")
PKG = Path(__file__).resolve().parent
BLOCK_SEC = 900  # 1875 YAMNet hops -> blocks stay on the frame grid


@dataclass
class Params:
    win_s: float = 3.0            # voice-embedding window
    hop_s: float = 1.5
    # cosine similarity (CAM++ embeddings) thresholds; calibrated with scripts/calibrate.py on VocalSet
    cluster_sim: float = 0.32     # windows of one video at least this similar (avg-linkage) = same voice
    voice_sim: float = 0.80       # a video's voice joins an existing DB voice only at/above this (mean of its 2 closest clips): precision over fragmentation, see scripts/sim_voice_assignment.py
    min_clip: float = 3.0
    max_clip: float = 30.0
    sing_on: float = 0.55
    sing_off: float = 0.35
    min_segment: float = 3.0
    max_gap: float = 1.5
    max_cluster_windows: int = 2500


@dataclass
class Run:
    start: float
    end: float
    label: int
    emb: np.ndarray
    score: float


@dataclass
class Analysis:
    info: MediaInfo
    segments: list[Segment]
    runs: list[Run]
    centroids: dict[int, np.ndarray] = field(default_factory=dict)   # label -> unit centroid
    seconds: dict[int, float] = field(default_factory=dict)


def _mode_filter(labels: list[int], radius: int = 2) -> list[int]:
    out = []
    for i in range(len(labels)):
        w = labels[max(0, i - radius) : i + radius + 1]
        vals, counts = np.unique(w, return_counts=True)
        best = vals[counts == counts.max()]
        out.append(labels[i] if labels[i] in best else int(best[0]))
    return out


class Engine:
    def __init__(self, data_dir: str | Path | None = None, models_dir: str | Path | None = None,
                 params: Params | None = None, voice_model: str = "wespeaker_campplus_lm.onnx"):
        self.params = params or Params()
        root = Path(os.environ.get("SOUNDALIKE_ROOT", PKG.parent))
        models = Path(models_dir or os.environ.get("SOUNDALIKE_MODELS", root / "models"))
        self.models_dir = models
        self.yamnet = Yamnet(models / "yamnet.onnx", models / "yamnet_class_map.csv")
        head_path = PKG / "assets" / "singing_head.npz"
        self.head = Head.load(head_path) if head_path.exists() else None
        if self.head is None:
            log.warning("no trained singing head found; using off-the-shelf AudioSet scores only")
        self.detector = SingingDetector(self.yamnet, self.head)
        self.embedder = VoiceEmbedder(models / voice_model)
        self.db = Database(data_dir or os.environ.get("SOUNDALIKE_DATA", root / "data" / "db"), self.embedder.dim)

    # ---- detection ---------------------------------------------------------------------------
    def detect(self, path: str | Path, info: MediaInfo | None = None) -> tuple[list[Segment], np.ndarray, np.ndarray | None]:
        """Singing segments for a whole file (any length). Returns (segments, frame_probs, wave_if_short)."""
        info = info or probe(path)
        pr = self.params
        if info.duration <= BLOCK_SEC + 1:
            wave = load_audio(path)
            p = self.detector.frame_scores(wave)
            wave_keep: np.ndarray | None = wave
        else:
            parts, wave_keep = [], None
            t = 0.0
            while t < info.duration:
                w = load_audio(path, start=t, dur=BLOCK_SEC + HOP)
                if w.size >= int(0.96 * SR):
                    f = self.detector.frame_scores(w)
                    parts.append(f[: int(BLOCK_SEC / HOP)] if t + BLOCK_SEC < info.duration else f)
                t += BLOCK_SEC
            p = np.concatenate(parts) if parts else np.zeros(0)
        segs = self.detector.segment(p, on=pr.sing_on, off=pr.sing_off, min_len=pr.min_segment, max_gap=pr.max_gap)
        return segs, p, wave_keep

    # ---- voices ------------------------------------------------------------------------------
    def analyze(self, path: str | Path) -> Analysis:
        info = probe(path)
        segs, _, wave = self.detect(path, info)
        if not segs:
            return Analysis(info, [], [])
        pr = self.params
        win_t, win_emb, win_seg = [], [], []
        for si, s in enumerate(segs):
            a, b = s.start, s.end
            if wave is not None:
                w = wave[int(a * SR) : int(b * SR)]
            else:
                w = load_audio(path, start=a, dur=b - a)
            ts, E = self.embedder.embed_windows(w, pr.win_s, pr.hop_s, min_s=1.5)
            for t, e in zip(ts, E):
                win_t.append(a + t); win_emb.append(e); win_seg.append(si)
        if not win_emb:
            return Analysis(info, segs, [])
        E = np.stack(win_emb)
        labels = self._cluster(E)
        # temporal smoothing inside each segment
        win_t = np.array(win_t); win_seg = np.array(win_seg)
        labels = np.array(labels)
        for si in range(len(segs)):
            idx = np.where(win_seg == si)[0]
            labels[idx] = _mode_filter(list(labels[idx]))
        runs: list[Run] = []
        for si, s in enumerate(segs):
            idx = np.where(win_seg == si)[0]
            if len(idx) == 0:
                continue
            # each window "owns" its central hop; first/last extend to the segment edges
            owned = []
            for k, i in enumerate(idx):
                c = win_t[i] + pr.win_s / 2
                a = s.start if k == 0 else c - pr.hop_s / 2
                b = s.end if k == len(idx) - 1 else c + pr.hop_s / 2
                owned.append((a, b, int(labels[i]), i))
            cur = [owned[0]]
            groups = []
            for o in owned[1:]:
                if o[2] == cur[-1][2]:
                    cur.append(o)
                else:
                    groups.append(cur); cur = [o]
            groups.append(cur)
            for g in groups:
                a, b, lab = g[0][0], g[-1][1], g[0][2]
                if b - a < pr.min_clip:
                    continue
                emb = l2norm(E[[o[3] for o in g]].mean(0))
                runs.append(Run(a, b, lab, emb, s.score))
        cents, secs = {}, {}
        for lab in set(int(l) for l in labels):
            cents[lab] = l2norm(E[labels == lab].mean(0))
        for r in runs:
            secs[r.label] = secs.get(r.label, 0.0) + (r.end - r.start)
        return Analysis(info, segs, runs, cents, secs)

    def _cluster(self, E: np.ndarray) -> list[int]:
        n = len(E)
        if n == 1:
            return [0]
        pr = self.params
        sub = np.arange(n)
        if n > pr.max_cluster_windows:
            sub = np.linspace(0, n - 1, pr.max_cluster_windows).astype(int)
        Z = linkage(E[sub], method="average", metric="cosine")
        lab_sub = fcluster(Z, t=1.0 - pr.cluster_sim, criterion="distance") - 1
        if len(sub) == n:
            return [int(x) for x in lab_sub]
        cents = np.stack([l2norm(E[sub][lab_sub == k].mean(0)) for k in range(lab_sub.max() + 1)])
        return [int(x) for x in (E @ cents.T).argmax(1)]

    # ---- ingestion ---------------------------------------------------------------------------
    def ingest(self, path: str | Path, source_id: int) -> dict:
        t0 = time.time()
        if not probe(path).has_audio:
            return {"status": "skipped", "error": "no audio track", "segments": 0, "clips": 0, "voices": 0,
                    "seconds": round(time.time() - t0, 1)}
        an = self.analyze(path)
        if not an.runs:
            return {"status": "no_singing", "segments": len(an.segments), "clips": 0, "voices": 0,
                    "seconds": round(time.time() - t0, 1)}
        pr = self.params
        label_voice: dict[int, int | None] = {}
        n_clips = 0
        voices_touched: set[int] = set()
        for r in an.runs:
            L = r.end - r.start
            pieces = max(1, math.ceil(L / pr.max_clip))
            for k in range(pieces):
                a = r.start + L * k / pieces
                b = r.start + L * (k + 1) / pieces
                emb = r.emb
                if r.label not in label_voice:
                    vid, sim = self.db.nearest_voice(an.centroids[r.label])
                    label_voice[r.label] = vid if (vid is not None and sim >= pr.voice_sim) else None
                cid, vid = self.db.add_clip(source_id, label_voice[r.label], a, b, emb, r.score, "video")
                label_voice[r.label] = vid
                try:
                    kind = export_clip(path, self.db.clip_file(cid), a, b, an.info)
                except Exception as e:  # keep going; one bad clip must not lose the whole video
                    log.warning("clip export failed (%s); dropping clip", e)
                    self.db.delete_clip(cid)
                    continue
                self.db.set_clip_path(cid, str(self.db.clip_file(cid).relative_to(self.db.root / "clips")), kind)
                voices_touched.add(vid)
                n_clips += 1
        return {"status": "done" if n_clips else "no_singing", "segments": len(an.segments), "clips": n_clips,
                "voices": len(voices_touched), "seconds": round(time.time() - t0, 1)}

    # ---- querying ----------------------------------------------------------------------------
    def analyze_query(self, path: str | Path, max_seconds: float = 180.0) -> dict:
        """Find the singing in a user's upload/recording and build one embedding per distinct voice."""
        info = probe(path)
        wave = load_audio(path, dur=max_seconds)
        if wave.size < 2 * SR:
            raise ValueError("The recording is too short — please sing for at least 3 seconds.")
        segs, p = self.detector.detect(wave, on=self.params.sing_on, off=self.params.sing_off,
                                       min_len=2.0, max_gap=self.params.max_gap)
        fallback = not segs
        if fallback:
            segs = [Segment(0.0, wave.size / SR, float(p.mean()) if p.size else 0.0)]
        win_emb = []
        for s in segs:
            w = wave[int(s.start * SR) : int(s.end * SR)]
            _, E = self.embedder.embed_windows(w, self.params.win_s, self.params.hop_s, min_s=1.5)
            if len(E):
                win_emb.append(E)
        if not win_emb:
            raise ValueError("Could not find enough voiced audio in the recording.")
        E = np.concatenate(win_emb)
        labels = np.array(self._cluster(E))
        voices = []
        for lab in sorted(set(labels), key=lambda l: -(labels == l).sum()):
            m = labels == lab
            voices.append({"embedding": l2norm(E[m].mean(0)), "windows": int(m.sum()),
                           "seconds": float(m.sum() * self.params.hop_s)})
        return {
            "duration": info.duration if info.duration else wave.size / SR,
            "singing_segments": [(round(s.start, 2), round(s.end, 2)) for s in segs],
            "singing_detected": not fallback,
            "singing_seconds": round(sum(s.end - s.start for s in segs), 1),
            "voices": voices,
        }

    def match(self, path: str | Path, top_voices: int = 5) -> dict:
        q = self.analyze_query(path)
        results = []
        for v in q["voices"][:3]:
            results.append({"query_voice_seconds": v["seconds"], "matches": self.db.match(v["embedding"], top_voices)})
        q["voices"] = [{k: x for k, x in v.items() if k != "embedding"} for v in q["voices"]]
        q["results"] = results
        return q
