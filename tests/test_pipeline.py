"""End-to-end engine tests on REAL recordings (VocalSet singers, LibriSpeech speech, FMA instrumentals)."""
import glob

import numpy as np

from conftest import DATA, needs_data, vocalset


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


@needs_data
def test_video_singing_found_voices_split_and_clips_playable(engine, test_video):
    path, timeline = test_video
    sid = engine.db.add_source("upload", "upload:video1", url=path, title="video1")
    res = engine.ingest(path, sid)
    assert res["status"] == "done"
    truth = [(t["start"], t["end"]) for t in timeline if "singing" in t["kind"]]
    nonsing = [(t["start"], t["end"]) for t in timeline if "singing" not in t["kind"]]
    clips = engine.db.conn().execute("SELECT * FROM clips WHERE source_id=? ORDER BY start", (sid,)).fetchall()
    assert clips
    total = sum(c["end"] - c["start"] for c in clips)
    inside = sum(overlap((c["start"], c["end"]), s) for c in clips for s in truth)
    wrong = sum(overlap((c["start"], c["end"]), s) for c in clips for s in nonsing)
    covered = sum(overlap((c["start"], c["end"]), s) for c in clips for s in truth) / sum(e - s for s, e in truth)
    print(f"clips cover {covered:.0%} of true singing; {wrong:.1f}s of {total:.1f}s clip time is not singing")
    assert covered > 0.75                       # most of what was sung is found
    assert wrong / total < 0.20                 # speech / instrumental / silence mostly excluded
    # the two different singers must end up in different voices
    def voice_at(t):
        for c in clips:
            if c["start"] <= t <= c["end"]:
                return c["voice_id"]
    a, b = voice_at(27), voice_at(50)
    assert a is not None and b is not None and a != b
    # every clip is a real, playable mp4 with a picture track (the source had video)
    import subprocess
    for c in clips:
        f = engine.db.clip_file(c["id"])
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(f)],
                             capture_output=True, text=True).stdout
        assert "video" in out and "audio" in out and c["kind"] == "video"


@needs_data
def test_speech_only_and_instrumental_only_yield_no_clips(engine):
    for pat in ("speech_big/clean_*.flac", "music/instrumental/*.ogg"):
        f = sorted(glob.glob(str(DATA / pat)))[5]
        sid = engine.db.add_source("upload", f"upload:{f}", url=f, title=f)
        res = engine.ingest(f, sid)
        assert res["status"] == "no_singing" and res["clips"] == 0, (pat, res)


@needs_data
def test_recording_matches_the_right_singer(engine, tmp_path):
    """Enrol 8 real singers with two songs each; query with a third, unseen song."""
    singers = [f"female{i}" for i in (1, 2, 3, 4)] + [f"male{i}" for i in (1, 2, 3, 4)]
    from soundalike.pipeline import Engine
    eng = Engine(data_dir=tmp_path / "db")
    for s in singers:
        for song in ("caro", "dona"):
            f = vocalset(s, song, "vibrato" if song == "caro" else "straight")
            sid = eng.db.add_source("upload", f"upload:{f}", url=f, title=f"{s} {song}")
            assert eng.ingest(f, sid)["clips"] >= 1
    hits = 0
    for s in singers:
        q = vocalset(s, "row", "straight")
        res = eng.match(q)
        top = res["results"][0]["matches"][0]
        titles = {c["title"] for c in top["clips"]}
        hits += any(t.startswith(s) for t in titles)
    print(f"top-1 correct singer: {hits}/{len(singers)}")
    assert hits >= 7


@needs_data
def test_query_without_singing_falls_back_and_short_audio_rejected(engine, tmp_path):
    import pytest
    speech = sorted(glob.glob(str(DATA / "speech_big" / "clean_*.flac")))[2]
    r = engine.analyze_query(speech)
    assert r["voices"] and r["singing_detected"] is False                  # still embeds, but says it found no singing
    from soundalike.audio import encode_wav
    encode_wav(np.zeros(8000, np.float32) + 0.01, tmp_path / "short.wav")
    with pytest.raises(ValueError):
        engine.analyze_query(tmp_path / "short.wav")


@needs_data
def test_silent_video_is_skipped_not_failed(engine, tmp_path):
    import subprocess
    f = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
                    "-pix_fmt", "yuv420p", str(f)], check=True)
    sid = engine.db.add_source("upload", "upload:silent-test", url=str(f), title="silent")
    res = engine.ingest(f, sid)
    assert res["status"] == "skipped" and res["error"] == "no audio track"
