"""ffmpeg helpers: probing, decoding to mono float32, clip export."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000


class MediaError(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@dataclass
class MediaInfo:
    duration: float
    has_video: bool
    has_audio: bool
    width: int = 0
    height: int = 0


def probe(path: str | Path) -> MediaInfo:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
    )
    if p.returncode != 0:
        raise MediaError(f"ffprobe failed: {p.stderr.decode(errors='replace')[:300]}")
    j = json.loads(p.stdout or b"{}")
    streams = j.get("streams", [])
    # attached cover art shows up as a "video" stream with disposition attached_pic
    vids = [
        s for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        and s.get("codec_name") not in ("mjpeg", "png") or (s.get("codec_type") == "video" and float(s.get("duration", 0) or 0) > 1)
    ]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    dur = 0.0
    try:
        dur = float(j.get("format", {}).get("duration") or 0)
    except ValueError:
        pass
    if not dur:
        for s in streams:
            try:
                dur = max(dur, float(s.get("duration") or 0))
            except ValueError:
                pass
    v = vids[0] if vids else {}
    return MediaInfo(dur, bool(vids), bool(auds), int(v.get("width", 0) or 0), int(v.get("height", 0) or 0))


def load_audio(path: str | Path, sr: int = SR, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """Decode any audio/video file to mono float32 in [-1, 1] at `sr`."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-vn", "-map", "0:a:0?", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 and not p.stdout:
        raise MediaError(f"ffmpeg decode failed: {p.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def encode_wav(wave: np.ndarray, path: str | Path, sr: int = SR) -> None:
    pcm = (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-", str(path)]
    p = subprocess.run(cmd, input=pcm, capture_output=True)
    if p.returncode != 0:
        raise MediaError(p.stderr.decode(errors="replace")[:300])
