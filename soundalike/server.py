"""HTTP API + web UI."""
from __future__ import annotations

import ipaddress
import json
import logging
import math
import os
import shutil
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .crawler import Crawler
from .pipeline import PKG, Engine
from .sources import default_sources

log = logging.getLogger("soundalike.server")
WEB = PKG.parent / "web"


def load_calibration() -> tuple[float, float]:
    """cosine -> P(same singer) mapping fitted on VocalSet by scripts/calibrate.py."""
    p = PKG / "assets" / "calibration.json"
    try:
        j = json.loads(p.read_text())
        return float(j["a"]), float(j["b"])
    except Exception:
        return 10.0, -6.0


class CrawlStart(BaseModel):
    sources: list[str] = ["zenodo", "commons"]
    workers: int = 2
    discover: bool = True       # local folders are deliberately CLI-only (`crawl --folder`), never settable over HTTP


def is_public_http_url(u: str) -> bool:
    """SSRF guard for user-supplied URLs: http(s) only, and every address the host resolves to must be public.
    (Set SOUNDALIKE_ALLOW_PRIVATE_URLS=1 for a trusted single-user setup that needs LAN media.)"""
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    if os.environ.get("SOUNDALIKE_ALLOW_PRIVATE_URLS") == "1":
        return True
    host = p.hostname.lower()
    try:                                    # IP literal: judge it directly
        return ipaddress.ip_address(host).is_global
    except ValueError:
        pass
    if host == "localhost" or host.endswith((".localhost", ".localdomain", ".local", ".internal", ".lan", ".home.arpa")) or "." not in host:
        return False
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return True    # not resolvable here (e.g. behind an egress proxy that resolves for us): nothing local to reach
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


class UrlList(BaseModel):
    urls: list[str]


def create_app(engine: Engine | None = None) -> FastAPI:
    engine = engine or Engine()
    app = FastAPI(title="Sound Alike Singer")
    app.state.engine = engine
    app.state.crawler = None
    cal_a, cal_b = load_calibration()
    db = engine.db

    def percent(cos: float) -> float:
        return round(100.0 / (1.0 + math.exp(-(cal_a * cos + cal_b))), 1)

    def clip_json(c: dict) -> dict:
        return {
            "id": c["id"], "url": f"/clips/{c['id']}.mp4", "similarity": round(c.get("similarity", 0.0), 4),
            "start": round(c["start"], 2), "end": round(c["end"], 2), "kind": c["kind"],
            "title": c.get("title"), "page_url": c.get("page_url"), "author": c.get("author"),
            "license": c.get("license"), "license_url": c.get("license_url"), "voice_id": c["voice_id"],
        }

    # ------------------------------------------------------------------ matching
    def save_upload(file: UploadFile) -> Path:
        suffix = Path(file.filename or "").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir()) as f:
            shutil.copyfileobj(file.file, f)
            return Path(f.name)

    @app.post("/api/match")
    def api_match(file: UploadFile = File(...)):
        path = save_upload(file)
        try:
            res = engine.match(path)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            log.exception("match failed")
            raise HTTPException(400, f"Could not read that file as audio/video: {e}")
        finally:
            path.unlink(missing_ok=True)
        for r in res["results"]:
            for m in r["matches"]:
                m["percent"] = percent(m["best"])
                m["clips"] = [clip_json(c) for c in m["clips"]]
        res["database"] = db.stats()
        return res

    # ------------------------------------------------------------------ database browsing
    @app.get("/api/stats")
    def api_stats():
        return db.stats()

    @app.get("/api/voices")
    def api_voices(limit: int = 24, offset: int = 0, min_clips: int = 1):
        limit, offset = max(1, min(limit, 200)), max(0, offset)
        c = db.conn()
        rows = c.execute(
            "SELECT id,label,n_clips,total_sec FROM voices WHERE n_clips>=? ORDER BY n_clips DESC, id LIMIT ? OFFSET ?",
            (min_clips, limit, offset)).fetchall()
        out = []
        for v in rows:
            clip = c.execute(
                "SELECT clips.*, sources.title, sources.page_url, sources.author, sources.license, sources.license_url "
                "FROM clips JOIN sources ON sources.id=clips.source_id WHERE voice_id=? AND deleted=0 "
                "ORDER BY score DESC LIMIT 1", (v["id"],)).fetchone()
            out.append({"id": v["id"], "label": v["label"], "n_clips": v["n_clips"], "seconds": round(v["total_sec"], 1),
                        "clip": clip_json(dict(clip)) if clip else None})
        total = c.execute("SELECT COUNT(*) FROM voices WHERE n_clips>=?", (min_clips,)).fetchone()[0]
        return {"total": total, "voices": out}

    @app.get("/api/voices/{vid}")
    def api_voice(vid: int, limit: int = 100, offset: int = 0):
        limit, offset = max(1, min(limit, 500)), max(0, offset)
        rows = db.conn().execute(
            "SELECT clips.*, sources.title, sources.page_url, sources.author, sources.license, sources.license_url "
            "FROM clips JOIN sources ON sources.id=clips.source_id WHERE voice_id=? AND deleted=0 "
            "ORDER BY clips.id LIMIT ? OFFSET ?", (vid, limit, offset)).fetchall()
        return {"voice_id": vid, "clips": [clip_json(dict(r)) for r in rows]}

    @app.get("/api/sources")
    def api_sources(status: str | None = None, limit: int = 50, offset: int = 0):
        limit, offset = max(1, min(limit, 500)), max(0, offset)
        q = ("SELECT id,kind,ref,title,page_url,license,duration,status,error,n_segments,n_clips,n_voices,finished_at "
             "FROM sources")
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        rows = db.conn().execute(q, [*args, limit, offset]).fetchall()
        return {"sources": [dict(r) for r in rows]}

    @app.get("/clips/{name}")
    def get_clip(name: str):
        try:
            cid = int(name.split(".")[0])
        except ValueError:
            raise HTTPException(404)
        f = db.clip_file(cid)
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f, media_type="video/mp4", headers={"Cache-Control": "public, max-age=86400"})

    # ------------------------------------------------------------------ crawler control
    def get_crawler() -> Crawler | None:
        return app.state.crawler

    @app.get("/api/crawl/status")
    def crawl_status():
        cr = get_crawler()
        if cr is None:
            return {"running": False, "events": [], "workers": 0, "active": [], "discovering": "",
                    "counters": {}, "queries_total": 0, "queries_exhausted": 0, **db.stats()}
        return cr.status()

    @app.post("/api/crawl/start")
    def crawl_start(req: CrawlStart):
        cr = get_crawler()
        if cr is not None and cr.running:
            return cr.status()
        srcs = default_sources(zenodo="zenodo" in req.sources, commons="commons" in req.sources)
        app.state.crawler = Crawler(engine, srcs, workers=max(1, min(req.workers, 16)),
                                    work_dir=engine.db.root / "tmp", discover=req.discover)
        app.state.crawler.start()
        return app.state.crawler.status()

    @app.post("/api/crawl/stop")
    def crawl_stop():
        cr = get_crawler()
        if cr is not None:
            cr.stop(wait=False)
            return cr.status()
        return crawl_status()

    @app.post("/api/crawl/urls")
    def crawl_urls(req: UrlList):
        added, rejected = 0, []
        for u in req.urls[:500]:
            u = u.strip()
            if not is_public_http_url(u):
                rejected.append(u)
                continue
            sid = db.add_source("url", f"url:{u}", url=u, title=u.rsplit("/", 1)[-1] or u, page_url=u)
            added += 1 if sid else 0
        cr = get_crawler()
        if cr is None or not cr.running:
            srcs = default_sources(zenodo=False, commons=False)
            app.state.crawler = Crawler(engine, srcs, workers=2, work_dir=engine.db.root / "tmp", discover=False)
            app.state.crawler.start()
        if rejected and not added:
            raise HTTPException(422, "Only public http(s) URLs are accepted.")
        return {"added": added, "rejected": len(rejected)}

    @app.post("/api/ingest")
    def api_ingest(file: UploadFile = File(...)):
        """Add a video/audio you have the rights to into the database (analysed by the workers)."""
        dest_dir = db.root / "uploads"
        dest_dir.mkdir(exist_ok=True)
        path = save_upload(file)
        final = dest_dir / f"{path.stem}{path.suffix}"
        shutil.move(path, final)
        sid = db.add_source("upload", f"upload:{final.name}", url=str(final), title=file.filename or final.name,
                            license="uploaded by user")
        cr = get_crawler()
        if cr is None or not cr.running:
            srcs = default_sources(zenodo=False, commons=False)
            app.state.crawler = Crawler(engine, srcs, workers=2, work_dir=engine.db.root / "tmp", discover=False)
            app.state.crawler.start()
        return {"source_id": sid}

    @app.get("/api/health")
    def health():
        return {"ok": True, "embedding_dim": engine.embedder.dim, "singing_head": engine.head is not None}

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
