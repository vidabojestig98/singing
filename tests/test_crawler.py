"""Crawler against local mocks of the MediaWiki and Zenodo APIs that serve REAL media files."""
import glob
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import DATA, needs_data

FILES: dict[str, Path] = {}


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        base = f"http://127.0.0.1:{self.server.server_port}"
        if u.path.startswith("/files/"):
            data = FILES[urllib.parse.unquote(u.path[7:])].read_bytes()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data); return
        if u.path == "/w/api.php":                       # ---- MediaWiki
            def page(title, mtype="AUDIO", name=None):
                return {"title": title, "imageinfo": [{
                    "url": f"{base}/files/{name or title}", "descriptionurl": f"{base}/wiki/{title}", "mediatype": mtype,
                    "mime": "audio/x-wav", "duration": 20.0, "size": 1000,
                    "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}, "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                    "Artist": {"value": '<a href="x">Some <b>Singer</b></a>'}, "ObjectName": {"value": title.replace("File:", "")}}}]}
            if q.get("list") == "categorymembers":
                subs = {"Category:Singing": ["Category:Choirs"]}.get(q["cmtitle"], [])
                return self._json({"query": {"categorymembers": [{"title": s} for s in subs]}})
            if q.get("generator") == "categorymembers":
                t = q["gcmtitle"]
                if t == "Category:Singing":
                    if "gcmcontinue" not in q:
                        return self._json({"continue": {"gcmcontinue": "p2", "continue": "gcmcontinue||"},
                                           "query": {"pages": [page("File:sing_a.wav", name="sing_a.wav"), page("File:photo.jpg", "BITMAP")]}})
                    return self._json({"query": {"pages": [page("File:talk_a.wav", name="talk_a.wav")]}})
                if t == "Category:Choirs":
                    return self._json({"query": {"pages": [page("File:sing_b.wav", name="sing_b.wav")]}})
                return self._json({"query": {"pages": []}})
            return self._json({"query": {"pages": []}})   # searches: nothing
        if u.path == "/api/records":                    # ---- Zenodo
            pg = int(q.get("page", 1))
            def rec(i, key):
                return {"id": i, "doi": f"10.5281/zenodo.{i}", "links": {"self_html": f"https://zenodo.org/records/{i}"},
                        "metadata": {"title": f"Record {i}", "license": {"id": "cc-by-4.0"}, "creators": [{"name": "A. Person"}]},
                        "files": [{"key": key, "size": 100, "links": {"self": f"{base}/files/{key}"}},
                                  {"key": "data.zip", "size": 10, "links": {"self": f"{base}/files/data.zip"}}]}
            hits = [rec(11, "sing_c.wav")] if pg == 1 else []
            return self._json({"hits": {"total": 1, "hits": hits}})
        self.send_response(404); self.end_headers()


@pytest.fixture(scope="module")
def mock():
    FILES.clear()
    voc = sorted(glob.glob(str(DATA / "vocalset" / "*" / "excerpts" / "*" / "*.wav")))
    spe = sorted(glob.glob(str(DATA / "speech_big" / "clean_*.flac")))
    FILES["sing_a.wav"] = Path(voc[0]); FILES["sing_b.wav"] = Path(voc[40]); FILES["sing_c.wav"] = Path(voc[80])
    FILES["talk_a.wav"] = Path(spe[0])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@needs_data
def test_commons_paging_subcategories_and_filtering(mock):
    from soundalike.sources import CommonsSource
    s = CommonsSource(api=f"{mock}/w/api.php"); s.limiter.min_interval = 0
    p1 = s.discover("category:0:Singing", None)
    assert [i.ref for i in p1.items] == ["commons:File:sing_a.wav"]             # the JPG is dropped
    assert p1.new_queries == ["category:1:Choirs"]
    it = p1.items[0]
    assert it.author == "Some Singer" and it.license == "CC BY-SA 4.0" and it.duration == 20.0
    assert p1.next_cursor
    p2 = s.discover("category:0:Singing", p1.next_cursor)
    assert [i.ref for i in p2.items] == ["commons:File:talk_a.wav"] and p2.next_cursor is None


@needs_data
def test_crawl_end_to_end_with_resume(mock, engine, monkeypatch):
    from soundalike import sources
    from soundalike.crawler import Crawler
    from soundalike.sources import commons
    monkeypatch.setattr(commons, "SEED_CATEGORIES", ["Singing"])
    monkeypatch.setattr(commons, "SEED_SEARCH_TERMS", [])
    def mk():
        c = sources.CommonsSource(api=f"{mock}/w/api.php"); c.limiter.min_interval = 0
        z = sources.ZenodoSource(api=f"{mock}/api", types=("audio",), terms=["singing"]); z.limiter.min_interval = 0
        for s in (c, z):
            s.http.headers["User-Agent"] = "test"
        return [c, z]
    # first crawler: discovery only, a single pass of a few pages, then "crash"
    cr = Crawler(engine, mk(), workers=0, discover=True, work_dir=engine.db.root / "tmp")
    cr.start()
    import time
    t0 = time.time()
    while time.time() - t0 < 20 and engine.db.conn().execute("SELECT COUNT(*) FROM sources").fetchone()[0] < 2:
        time.sleep(0.2)
    cr.stop()
    found_before = engine.db.conn().execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    assert found_before >= 2
    # second crawler resumes from the persisted cursors and finishes everything
    cr2 = Crawler(engine, mk(), workers=2, discover=True, work_dir=engine.db.root / "tmp")
    cr2.start()
    t0 = time.time()
    while time.time() - t0 < 120:
        ex = engine.db.conn().execute("SELECT COUNT(*) FROM crawl_state WHERE exhausted=0").fetchone()[0]
        q = engine.db.conn().execute("SELECT COUNT(*) FROM sources WHERE status IN ('queued','working')").fetchone()[0]
        if ex == 0 and q == 0:
            break
        time.sleep(0.5)
    cr2.stop()
    rows = {r["ref"]: r for r in engine.db.conn().execute("SELECT * FROM sources")}
    assert set(rows) == {"commons:File:sing_a.wav", "commons:File:talk_a.wav", "commons:File:sing_b.wav", "zenodo:11/sing_c.wav"}
    assert rows["commons:File:talk_a.wav"]["status"] == "no_singing"              # speech only -> no clips
    for k in ("commons:File:sing_a.wav", "commons:File:sing_b.wav", "zenodo:11/sing_c.wav"):
        assert rows[k]["status"] == "done" and rows[k]["n_clips"] >= 1, (k, dict(rows[k]))
    assert rows["commons:File:sing_a.wav"]["license"] == "CC BY-SA 4.0"
    st = engine.db.stats()
    assert st["clips"] >= 3 and st["voices"] >= 2
    for c in engine.db.conn().execute("SELECT id, kind FROM clips"):
        f = engine.db.clip_file(c["id"])
        assert f.exists() and f.stat().st_size > 5000 and c["kind"] == "audio-viz"    # audio-only source -> waveform video
