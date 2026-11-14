import glob
import io

from conftest import needs_data, vocalset


@needs_data
def test_api_flow(tmp_path):
    from fastapi.testclient import TestClient
    from soundalike.pipeline import Engine
    from soundalike.server import create_app

    eng = Engine(data_dir=tmp_path / "db")
    app = create_app(eng)
    c = TestClient(app)
    assert c.get("/api/health").json()["singing_head"] is True
    assert c.get("/").status_code == 200 and "Sound Alike Singer" in c.get("/").text
    # empty DB: matching still answers, with no matches and a clear database block
    q = vocalset("female1", "row")
    r = c.post("/api/match", files={"file": ("q.wav", open(q, "rb"), "audio/wav")})
    assert r.status_code == 200 and r.json()["database"]["clips"] == 0
    # ingest through the API (the crawler's workers pick it up)
    f = vocalset("female1", "caro", "vibrato")
    assert c.post("/api/ingest", files={"file": ("f1_caro.wav", open(f, "rb"), "audio/wav")}).status_code == 200
    import time
    t0 = time.time()
    while time.time() - t0 < 120 and c.get("/api/stats").json()["clips"] == 0:
        time.sleep(0.5)
    app.state.crawler.stop()
    assert c.get("/api/stats").json()["clips"] >= 1
    res = c.post("/api/match", files={"file": ("q.wav", open(q, "rb"), "audio/wav")}).json()
    m = res["results"][0]["matches"][0]
    assert 0 < m["percent"] <= 100 and m["clips"][0]["url"].startswith("/clips/")
    clip = c.get(m["clips"][0]["url"])
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4" and len(clip.content) > 5000
    rng = c.get(m["clips"][0]["url"], headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100              # seekable in the browser
    assert c.get("/api/voices").json()["total"] >= 1 and c.get("/api/sources").json()["sources"]
    assert c.post("/api/match", files={"file": ("x.wav", io.BytesIO(b"not audio at all"), "audio/wav")}).status_code in (400, 422)
