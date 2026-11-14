"""python -m soundalike <command>"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="soundalike", description="Sound Alike Singer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--crawl", action="store_true", help="start the crawler immediately")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--sources", default="zenodo,commons")
    c = sub.add_parser("crawl", help="headless crawler (Ctrl-C to stop; resumes where it left off)")
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--sources", default="zenodo,commons")
    c.add_argument("--folder", action="append", default=[], help="also analyse a local folder (repeatable)")
    c.add_argument("--no-discover", action="store_true")
    i = sub.add_parser("ingest", help="analyse local media files and add their singing to the database")
    i.add_argument("files", nargs="+")
    m = sub.add_parser("match", help="find the closest database voices for a recording")
    m.add_argument("file")
    sub.add_parser("stats")
    sub.add_parser("download-models")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if a.cmd == "download-models":
        from .models import ensure_models
        from .pipeline import PKG
        ensure_models(PKG.parent / "models")
        print("models ready")
        return 0

    from .models import ensure_models
    from .pipeline import PKG, Engine
    ensure_models(PKG.parent / "models", quiet=True)
    eng = Engine()
    if a.cmd == "stats":
        print(json.dumps(eng.db.stats(), indent=2))
    elif a.cmd == "match":
        res = eng.match(a.file)
        for r in res["results"]:
            for mt in r["matches"]:
                print(f"voice {mt['voice_id']:>5}  cos {mt['best']:.3f}  ({mt['n_clips']} clips)  best clip #{mt['clips'][0]['id']}")
    elif a.cmd == "ingest":
        from .crawler import Crawler
        from .sources import default_sources
        cr = Crawler(eng, default_sources(zenodo=False, commons=False), workers=1, discover=False)
        for f in a.files:
            sid = eng.db.add_source("upload", f"upload:{f}:{int(time.time())}", url=f, title=f, license="local file")
            row = eng.db.claim_source()
            print(f, cr.process(row))
    elif a.cmd == "crawl":
        from .crawler import Crawler
        from .sources import default_sources
        srcs = default_sources(folders=a.folder, zenodo="zenodo" in a.sources, commons="commons" in a.sources)
        cr = Crawler(eng, srcs, workers=a.workers, work_dir=eng.db.root / "tmp", discover=not a.no_discover)
        cr.start()
        try:
            while True:
                time.sleep(15)
                st = cr.status()
                print(f"[{time.strftime('%H:%M:%S')}] sources={st['sources']} {st['sources_by_status']} voices={st['voices']} "
                      f"clips={st['clips']} singing={st['singing_seconds']/3600:.2f}h discovering={st['discovering']}", flush=True)
        except KeyboardInterrupt:
            cr.stop()
    elif a.cmd == "serve":
        import uvicorn

        from .crawler import Crawler
        from .server import create_app
        from .sources import default_sources
        app = create_app(eng)
        if a.crawl:
            srcs = default_sources(zenodo="zenodo" in a.sources, commons="commons" in a.sources)
            app.state.crawler = Crawler(eng, srcs, workers=a.workers, work_dir=eng.db.root / "tmp")
            app.state.crawler.start()
        uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
