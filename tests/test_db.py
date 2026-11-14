import time

import numpy as np

from soundalike.db import Database


def unit(rng, d, n=None):
    x = rng.standard_normal((n, d) if n else d).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_clips_voices_match_and_delete(tmp_path):
    rng = np.random.default_rng(1)
    db = Database(tmp_path, 32)
    sid = db.add_source("url", "url:a", url="http://x/a.mp4", title="A", license="CC BY 4.0")
    assert db.add_source("url", "url:a") is None                      # duplicate refs are ignored
    voices = [unit(rng, 32) for _ in range(5)]
    ids = {}
    for v_i, v in enumerate(voices):
        for k in range(3):
            noisy = v + 0.15 * unit(rng, 32)
            cid, vid = db.add_clip(sid, None, k * 10.0, k * 10.0 + 8, noisy, 0.9, "video", new_voice_threshold=0.8)
            ids.setdefault(v_i, set()).add(vid)
    assert all(len(s) == 1 for s in ids.values())                      # clips of one voice joined one voice row
    assert len({next(iter(s)) for s in ids.values()}) == 5             # 5 distinct voices
    q = voices[3] + 0.1 * unit(rng, 32)
    best = db.match(q, top_voices=3)
    assert best[0]["voice_id"] == next(iter(ids[3])) and len(best[0]["clips"]) == 3
    # deleting a clip removes it from search and from the voice counters
    victim = best[0]["clips"][0]["id"]
    db.delete_clip(victim)
    assert victim not in [c["id"] for m in db.match(q, 3) for c in m["clips"]]
    assert db.stats()["clips"] == 14 and db.stats()["voices"] == 5


def test_queue_claim_requeue_and_cursor(tmp_path):
    db = Database(tmp_path, 8)
    for i in range(3):
        db.add_source("url", f"url:{i}", url=f"http://x/{i}")
    a = db.claim_source(); b = db.claim_source()
    assert a["id"] != b["id"]
    assert db.requeue_stuck() == 2                                     # crash recovery
    db.finish_source(a["id"], "failed", "boom")
    assert db.retry_failed(3) == 1
    db.set_cursor("zenodo", "video|singing", "7", False, 5)
    db.set_cursor("zenodo", "video|singing", "8", False, 3)
    c = db.get_cursor("zenodo", "video|singing")
    assert c["cursor"] == "8" and c["n_found"] == 8


def test_search_scans_in_blocks_and_keeps_scaling(tmp_path):
    """No hard cap on clips: 300k embeddings, searched in 200k-row blocks."""
    rng = np.random.default_rng(2)
    d = 64
    db = Database(tmp_path, d)
    n = 300_000
    E = unit(rng, d, n)
    with open(db.emb_path, "wb") as f:
        f.write(np.zeros((1, d), np.float32).tobytes())               # row 0 unused (ids start at 1)
        f.write(E.tobytes())
    q = E[123_456]
    t0 = time.time()
    hits = db.search_clips(q, k=5, block=100_000)
    dt = time.time() - t0
    assert hits[0][0] == 123_457 and abs(hits[0][1] - 1.0) < 1e-4
    assert dt < 3.0
