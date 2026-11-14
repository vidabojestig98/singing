"""The pre-search crawler.

Two kinds of threads share the SQLite database as their queue, so everything survives restarts:

  discovery : walks every (source, query) page by page, inserting each media item it has not seen
              before as a 'queued' source and persisting the continuation cursor. New sub-queries
              (e.g. sub-categories) are added as it learns about them. It never stops on its own
              until every query of every source is exhausted; there is no cap on items found.
  workers   : claim queued sources, download them (temp file, deleted afterwards), run the engine
              (singing detection -> voices -> clips -> database) and record the outcome.
"""
from __future__ import annotations

import collections
import logging
import tempfile
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

from .pipeline import Engine
from .sources import Source

log = logging.getLogger("soundalike.crawler")


class Crawler:
    def __init__(self, engine: Engine, sources: list[Source], workers: int = 2, work_dir: str | Path | None = None,
                 max_queue: int = 2000, discover: bool = True):
        self.engine = engine
        self.db = engine.db
        self.sources = {s.kind: s for s in sources}
        self.n_workers = workers
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="soundalike-"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_queue = max_queue
        self.do_discover = discover
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.events: collections.deque = collections.deque(maxlen=200)
        self.active: dict[int, str] = {}          # source id -> title being processed
        self.discovering: str = ""
        self.counters = collections.Counter()

    # ---- control -----------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        requeued = self.db.requeue_stuck()
        if requeued:
            self._event(f"re-queued {requeued} interrupted sources")
        self._threads = []
        if self.do_discover:
            self._threads.append(threading.Thread(target=self._discover_loop, name="discover", daemon=True))
        for i in range(self.n_workers):
            self._threads.append(threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True))
        for t in self._threads:
            t.start()
        self._event(f"crawler started ({self.n_workers} workers, discovery={'on' if self.do_discover else 'off'})")

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            for t in self._threads:
                t.join(timeout=300)
        self._event("crawler stopped")

    def _event(self, msg: str) -> None:
        log.info(msg)
        self.events.append({"t": time.time(), "msg": msg})

    def status(self) -> dict:
        qs = self.db.conn().execute("SELECT kind, query, exhausted, n_found FROM (SELECT source_kind AS kind, query, "
                                    "exhausted, n_found FROM crawl_state)").fetchall()
        return {
            "running": self.running,
            "workers": self.n_workers,
            "discovering": self.discovering,
            "active": list(self.active.values()),
            "queries_total": len(qs),
            "queries_exhausted": sum(1 for q in qs if q["exhausted"]),
            "counters": dict(self.counters),
            "events": list(self.events)[-40:],
            **self.db.stats(),
        }

    # ---- discovery ---------------------------------------------------------------------------
    def _ensure_seed_queries(self) -> None:
        for s in self.sources.values():
            for q in s.seed_queries():
                if self.db.get_cursor(s.kind, q) is None:
                    self.db.set_cursor(s.kind, q, None, False)

    def _discover_loop(self) -> None:
        self._ensure_seed_queries()
        failures: collections.Counter = collections.Counter()
        retry_at: dict[tuple, float] = {}
        while not self._stop.is_set():
            rows = self.db.conn().execute(
                "SELECT source_kind, query, cursor FROM crawl_state WHERE exhausted=0 ORDER BY updated_at IS NOT NULL, updated_at"
            ).fetchall()
            rows = [r for r in rows if retry_at.get((r["source_kind"], r["query"]), 0) <= time.time()
                    and r["source_kind"] in self.sources]
            if not rows:
                self.discovering = ""
                self._stop.wait(5)
                continue
            if self.db.conn().execute("SELECT COUNT(*) FROM sources WHERE status='queued'").fetchone()[0] > self.max_queue:
                self.discovering = "(paused: analysis queue is full)"
                self._stop.wait(5)
                continue
            r = rows[0]
            kind, query, cursor = r["source_kind"], r["query"], r["cursor"]
            src = self.sources[kind]
            self.discovering = f"{kind}: {query}"
            try:
                page = src.discover(query, cursor)
            except Exception as e:  # network hiccup etc.: back off this query only
                failures[(kind, query)] += 1
                wait = min(3600, 30 * 2 ** failures[(kind, query)])
                retry_at[(kind, query)] = time.time() + wait
                self._event(f"discovery error on {kind} '{query}': {e} (retry in {wait}s)")
                self.db.set_cursor(kind, query, cursor, False)   # touch updated_at so others go first
                continue
            failures[(kind, query)] = 0
            new = 0
            for it in page.items:
                sid = self.db.add_source(
                    kind, it.ref, url=it.url, page_url=it.page_url, title=it.title, author=it.author,
                    license=it.license, license_url=it.license_url, duration=it.duration,
                    size_bytes=it.size_bytes, meta=it.meta or None)
                new += 1 if sid else 0
            for nq in page.new_queries:
                if self.db.get_cursor(kind, nq) is None:
                    self.db.set_cursor(kind, nq, None, False)
            self.db.set_cursor(kind, query, page.next_cursor, page.next_cursor is None, new)
            self.counters["discovered"] += new
            if new:
                self._event(f"found {new} new items via {kind} '{query}'")

    # ---- workers -----------------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            row = self.db.claim_source()
            if row is None:
                self._stop.wait(3)
                continue
            self.process(row)

    def process(self, row) -> dict:
        sid, title = row["id"], row["title"] or row["ref"]
        self.active[sid] = title
        tmp: Path | None = None
        try:
            src = self.sources.get(row["kind"])
            if src is None:
                raise RuntimeError(f"no source adapter for kind '{row['kind']}'")
            path = src.local_path(row)
            if path is None:
                ext = Path(urlparse(row["url"]).path).suffix or ".bin"
                tmp = self.work_dir / f"{sid}{ext}"
                src.download(row, tmp)
                path = tmp
            res = self.engine.ingest(path, sid)
            self.db.finish_source(sid, res["status"], None, n_segments=res["segments"], n_clips=res["clips"],
                                  n_voices=res["voices"])
            self.counters[res["status"]] += 1
            self._event(f"{res['status']}: {title[:70]} — {res['clips']} clips / {res['voices']} voices ({res['seconds']}s)")
            return res
        except Exception as e:
            log.debug(traceback.format_exc())
            self.db.finish_source(sid, "failed", f"{type(e).__name__}: {e}"[:500])
            self.counters["failed"] += 1
            self._event(f"failed: {title[:60]} — {e}")
            return {"status": "failed", "error": str(e)}
        finally:
            self.active.pop(sid, None)
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def wait_idle(self, timeout: float = 3600) -> bool:
        """Block until no source is queued/working (used by tests and the CLI one-shot mode)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            n = self.db.conn().execute("SELECT COUNT(*) FROM sources WHERE status IN ('queued','working')").fetchone()[0]
            if n == 0:
                return True
            time.sleep(1)
        return False
