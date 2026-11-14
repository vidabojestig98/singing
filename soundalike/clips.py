"""Cut a singing part out of a source as a standalone, browser-playable MP4 video clip."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .audio import MediaError, MediaInfo

X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"]
AAC = ["-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "44100"]


def export_clip(src: str | Path, out: str | Path, start: float, end: float, info: MediaInfo) -> str:
    """Writes `out` (mp4). Returns 'video' if the clip is a real video cut, 'audio-viz' if the
    source has no picture and a waveform animation was rendered instead."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    dur = max(0.5, end - start)
    base = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{dur:.3f}"]
    if info.has_video:
        cmd = base + ["-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale='min(640,iw)':-2:flags=bicubic,fps=fps=30", *X264, *AAC,
                      "-movflags", "+faststart", "-f", "mp4", str(tmp)]
        kind = "video"
    else:
        cmd = base + ["-filter_complex",
                      "[0:a:0]asplit[a][b];[b]showwaves=s=640x360:mode=cline:rate=25:colors=0x5b8cff,format=yuv420p[v]",
                      "-map", "[v]", "-map", "[a]", *X264, *AAC, "-movflags", "+faststart", "-f", "mp4", str(tmp)]
        kind = "audio-viz"
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
        tmp.unlink(missing_ok=True)
        if info.has_video:   # odd video stream (e.g. no decodable frames): fall back to waveform video
            return export_clip(src, out, start, end, MediaInfo(info.duration, False, info.has_audio))
        raise MediaError(f"clip export failed: {p.stderr.decode(errors='replace')[-300:]}")
    tmp.replace(out)
    return kind
