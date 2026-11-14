"""The singing-voices database.

* SQLite (WAL) holds sources (videos/audio that were found), voices (distinct singers) and clips.
* Clip files live on disk under  <root>/clips/<bucket>/<id>.mp4.
* Voice embeddings live in a flat float32 file  <root>/embeddings.f32  where row == clip id, so
  an insert is a single pwrite (no rewriting, no index rebuild) and the file is memory-mapped
  for search. There is no cap on sources, voices or clips: growth is bounded by disk only.
  Search is an exact, chunked cosine scan (fast to many millions of rows); swap `search_clips`
  for an ANN index if you ever need sub-100 ms at billions of rows.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,              -- commons | zenodo | folder | url | upload
  ref TEXT NOT NULL UNIQUE,        -- stable external id, e.g. commons:File:Foo.ogv
  url TEXT,                        -- direct media URL (or local path for folder)
  page_url TEXT,                   -- human page (attribution link)
  title TEXT, author TEXT, license TEXT, license_url TEXT,
  duration REAL, size_bytes INTEGER,
  status TEXT NOT NULL DEFAULT 'queued',   -- queued|working|done|no_singing|failed|skipped
  error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
  n_segments INTEGER DEFAULT 0, n_clips INTEGER DEFAULT 0, n_voices INTEGER DEFAULT 0,
  discovered_at REAL, started_at REAL, finished_at REAL,
  meta TEXT
);
CREATE INDEX IF NOT EXISTS sources_status ON sources(status, id);
CREATE TABLE IF NOT EXISTS voices (
  id INTEGER PRIMARY KEY,
  label TEXT,
  n_clips INTEGER NOT NULL DEFAULT 0,
  total_sec REAL NOT NULL DEFAULT 0,
  emb_sum BLOB,                    -- sum of clip embeddings (float32); centroid = normalise(sum)
  created_at REAL
);
CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES sources(id),
  voice_id INTEGER NOT NULL REFERENCES voices(id),
  start REAL NOT NULL, end REAL NOT NULL,
  path TEXT,                       -- relative to <root>/clips
  kind TEXT,                       -- video | audio-viz
  score REAL,                      -- mean P(singing) over the clip
  created_at REAL,
  deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS clips_voice ON clips(voice_id);
CREATE INDEX IF NOT EXISTS clips_source ON clips(source_id);
CREATE TABLE IF NOT EXISTS crawl_state (
  source_kind TEXT NOT NULL, query TEXT NOT NULL,
  cursor TEXT, exhausted INTEGER NOT NULL DEFAULT 0,
  n_found INTEGER NOT NULL DEFAULT 0, updated_at REAL,
  PRIMARY KEY (source_kind, query)
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


class Database:
    def __init__(self, root: str | Path, dim: int):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "clips").mkdir(exist_ok=True)
        self.dim = dim
        self.db_path = self.root / "sound.db"
        self.emb_path = self.root / "embeddings.f32"
        self._local = threading.local()
        self._wlock = threading.RLock()
        c = self.conn()
        c.executescript(SCHEMA)
        c.commit()
        stored = self.kv_get("emb_dim")
        if stored and int(stored) != dim:
            raise RuntimeError(f"database was built with {stored}-d embeddings, current model gives {dim}-d")
        self.kv_set("emb_dim", str(dim))
        if not self.emb_path.exists():
            self.emb_path.touch()
        self._emb_fd = os.open(self.emb_path, os.O_RDWR)
        self._voice_cache: tuple[np.ndarray, np.ndarray] | None = None   # (ids, normalised centroids)
        self._voice_dirty = True

    # ---- plumbing ----------------------------------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=60000")
            self._local.c = c
        return c

    def kv_get(self, k: str, default: str | None = None) -> str | None:
        r = self.conn().execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r["v"] if r else default

    def kv_set(self, k: str, v: str) -> None:
        with self._wlock:
            self.conn().execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
            self.conn().commit()

    # ---- sources / crawl queue -----------------------------------------------------------------
    def add_source(self, kind: str, ref: str, **fields) -> int | None:
        """Insert a discovered media item; returns its id, or None if it was already known."""
        meta = fields.pop("meta", None)
        cols = ["kind", "ref", "discovered_at", "meta"] + list(fields)
        vals = [kind, ref, time.time(), json.dumps(meta) if meta else None] + list(fields.values())
        with self._wlock:
            cur = self.conn().execute(
                f"INSERT OR IGNORE INTO sources({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals
            )
            self.conn().commit()
            return cur.lastrowid if cur.rowcount else None

    def claim_source(self, kinds: list[str] | None = None) -> sqlite3.Row | None:
        """Atomically take the oldest queued source."""
        with self._wlock:
            c = self.conn()
            q = "SELECT * FROM sources WHERE status='queued'"
            args: list = []
            if kinds:
                q += f" AND kind IN ({','.join('?' * len(kinds))})"
                args += kinds
            q += " ORDER BY id LIMIT 1"
            r = c.execute(q, args).fetchone()
            if not r:
                return None
            c.execute(
                "UPDATE sources SET status='working', started_at=?, attempts=attempts+1 WHERE id=?",
                (time.time(), r["id"]),
            )
            c.commit()
            return r

    def finish_source(self, sid: int, status: str, error: str | None = None, **counts) -> None:
        sets = ["status=?", "error=?", "finished_at=?"] + [f"{k}=?" for k in counts]
        with self._wlock:
            self.conn().execute(
                f"UPDATE sources SET {','.join(sets)} WHERE id=?", [status, error, time.time(), *counts.values(), sid]
            )
            self.conn().commit()

    def requeue_stuck(self) -> int:
        """After a crash, sources left in 'working' go back to the queue."""
        with self._wlock:
            cur = self.conn().execute("UPDATE sources SET status='queued' WHERE status='working'")
            self.conn().commit()
            return cur.rowcount

    def retry_failed(self, max_attempts: int = 3) -> int:
        with self._wlock:
            cur = self.conn().execute(
                "UPDATE sources SET status='queued' WHERE status='failed' AND attempts<?", (max_attempts,)
            )
            self.conn().commit()
            return cur.rowcount

    def get_cursor(self, kind: str, query: str):
        return self.conn().execute(
            "SELECT * FROM crawl_state WHERE source_kind=? AND query=?", (kind, query)
        ).fetchone()

    def set_cursor(self, kind: str, query: str, cursor: str | None, exhausted: bool, found_delta: int = 0) -> None:
        with self._wlock:
            self.conn().execute(
                """INSERT INTO crawl_state(source_kind,query,cursor,exhausted,n_found,updated_at) VALUES(?,?,?,?,?,?)
                   ON CONFLICT(source_kind,query) DO UPDATE SET cursor=excluded.cursor, exhausted=excluded.exhausted,
                   n_found=n_found+?, updated_at=excluded.updated_at""",
                (kind, query, cursor, int(exhausted), found_delta, time.time(), found_delta),
            )
            self.conn().commit()

    # ---- embeddings ----------------------------------------------------------------------------
    def _write_emb(self, row: int, vec: np.ndarray) -> None:
        os.pwrite(self._emb_fd, np.asarray(vec, np.float32).tobytes(), row * self.dim * 4)

    def _emb_matrix(self) -> np.memmap | None:
        size = os.path.getsize(self.emb_path)
        n = size // (self.dim * 4)
        if n == 0:
            return None
        return np.memmap(self.emb_path, dtype=np.float32, mode="r", shape=(n, self.dim))

    # ---- voices ------------------------------------------------------------------------------
    def _load_voices(self) -> tuple[np.ndarray, np.ndarray]:
        if self._voice_cache is None or self._voice_dirty:
            rows = self.conn().execute("SELECT id, emb_sum FROM voices WHERE n_clips>0 ORDER BY id").fetchall()
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            if rows:
                M = np.stack([np.frombuffer(r["emb_sum"], np.float32) for r in rows])
                M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
            else:
                M = np.zeros((0, self.dim), np.float32)
            self._voice_cache, self._voice_dirty = (ids, M), False
        return self._voice_cache

    def nearest_voice(self, emb: np.ndarray) -> tuple[int | None, float]:
        ids, M = self._load_voices()
        if len(ids) == 0:
            return None, -1.0
        sims = M @ emb
        j = int(sims.argmax())
        return int(ids[j]), float(sims[j])

    def add_clip(self, source_id: int, voice_id: int | None, start: float, end: float, emb: np.ndarray,
                 score: float, kind: str, new_voice_threshold: float | None = None) -> tuple[int, int]:
        """Insert a clip. If voice_id is None the clip joins the nearest voice when its centroid
        similarity >= new_voice_threshold, else founds a new voice. Returns (clip_id, voice_id)."""
        emb = np.asarray(emb, np.float32)
        emb = emb / max(float(np.linalg.norm(emb)), 1e-9)
        with self._wlock:
            c = self.conn()
            if voice_id is None:
                vid, sim = self.nearest_voice(emb)
                if vid is not None and new_voice_threshold is not None and sim >= new_voice_threshold:
                    voice_id = vid
                else:
                    cur = c.execute("INSERT INTO voices(label, emb_sum, created_at) VALUES(?,?,?)",
                                    (None, np.zeros(self.dim, np.float32).tobytes(), time.time()))
                    voice_id = cur.lastrowid
                    c.execute("UPDATE voices SET label=? WHERE id=?", (f"Voice {voice_id}", voice_id))
            cur = c.execute(
                "INSERT INTO clips(source_id,voice_id,start,end,kind,score,created_at) VALUES(?,?,?,?,?,?,?)",
                (source_id, voice_id, start, end, kind, score, time.time()),
            )
            cid = cur.lastrowid
            self._write_emb(cid, emb)
            row = c.execute("SELECT emb_sum FROM voices WHERE id=?", (voice_id,)).fetchone()
            s = np.frombuffer(row["emb_sum"], np.float32) + emb
            c.execute("UPDATE voices SET emb_sum=?, n_clips=n_clips+1, total_sec=total_sec+? WHERE id=?",
                      (s.astype(np.float32).tobytes(), end - start, voice_id))
            c.commit()
            self._voice_dirty = True
            return cid, voice_id

    def set_clip_path(self, clip_id: int, rel_path: str, kind: str | None = None) -> None:
        with self._wlock:
            self.conn().execute("UPDATE clips SET path=?, kind=COALESCE(?, kind) WHERE id=?", (rel_path, kind, clip_id))
            self.conn().commit()

    def clip_file(self, clip_id: int) -> Path:
        return self.root / "clips" / f"{clip_id // 1000:05d}" / f"{clip_id}.mp4"

    def delete_clip(self, clip_id: int) -> None:
        with self._wlock:
            c = self.conn()
            r = c.execute("SELECT voice_id, start, end FROM clips WHERE id=? AND deleted=0", (clip_id,)).fetchone()
            if not r:
                return
            self._write_emb(clip_id, np.zeros(self.dim, np.float32))
            c.execute("UPDATE clips SET deleted=1 WHERE id=?", (clip_id,))
            c.execute("UPDATE voices SET n_clips=n_clips-1, total_sec=total_sec-? WHERE id=?",
                      (r["end"] - r["start"], r["voice_id"]))
            c.commit()
            self._voice_dirty = True
        f = self.clip_file(clip_id)
        if f.exists():
            f.unlink()

    # ---- search ------------------------------------------------------------------------------
    def search_clips(self, q: np.ndarray, k: int = 50, block: int = 200_000) -> list[tuple[int, float]]:
        """Exact top-k cosine over every clip embedding (rows are unit vectors; zero rows = deleted)."""
        M = self._emb_matrix()
        if M is None:
            return []
        q = np.asarray(q, np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        best_s: list[np.ndarray] = []
        best_i: list[np.ndarray] = []
        for a in range(0, M.shape[0], block):
            s = np.asarray(M[a : a + block]) @ q
            kk = min(k, s.size)
            idx = np.argpartition(-s, kk - 1)[:kk]
            best_s.append(s[idx])
            best_i.append(idx + a)
        S, I = np.concatenate(best_s), np.concatenate(best_i)
        order = np.argsort(-S)[: k * 2]
        return [(int(I[o]), float(S[o])) for o in order if S[o] > 0]

    def match(self, q: np.ndarray, top_voices: int = 5, clips_per_voice: int = 3, pool: int = 400) -> list[dict]:
        """Rank voices by similarity to the query embedding; return their best clips."""
        hits = self.search_clips(q, k=pool)
        if not hits:
            return []
        c = self.conn()
        ids = [h[0] for h in hits]
        rows = {}
        for a in range(0, len(ids), 500):
            part = ids[a : a + 500]
            for r in c.execute(
                f"""SELECT clips.*, sources.title, sources.page_url, sources.author, sources.license,
                           sources.license_url, sources.kind AS source_kind
                    FROM clips JOIN sources ON sources.id=clips.source_id
                    WHERE clips.deleted=0 AND clips.id IN ({','.join('?' * len(part))})""", part
            ):
                rows[r["id"]] = r
        by_voice: dict[int, list[dict]] = {}
        for cid, sim in hits:
            r = rows.get(cid)
            if r is None:
                continue
            d = dict(r)
            d["similarity"] = sim
            by_voice.setdefault(r["voice_id"], []).append(d)
        out = []
        for vid, clips in by_voice.items():
            clips.sort(key=lambda d: -d["similarity"])
            top = clips[:clips_per_voice]
            out.append({
                "voice_id": vid,
                "score": float(np.mean([t["similarity"] for t in clips[:2]])),   # mean of best two clips
                "best": top[0]["similarity"],
                "clips": top,
            })
        out.sort(key=lambda d: -d["score"])
        vrows = {r["id"]: r for r in c.execute(
            f"SELECT id,label,n_clips,total_sec FROM voices WHERE id IN ({','.join('?' * len(out))})",
            [o["voice_id"] for o in out[:top_voices]]) } if out else {}
        for o in out[:top_voices]:
            v = vrows.get(o["voice_id"])
            o["label"] = v["label"] if v else f"Voice {o['voice_id']}"
            o["n_clips"] = v["n_clips"] if v else 0
            o["total_sec"] = v["total_sec"] if v else 0
        return out[:top_voices]

    # ---- stats -------------------------------------------------------------------------------
    def stats(self) -> dict:
        c = self.conn()
        s = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM sources GROUP BY status")}
        one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
        return {
            "sources": sum(s.values()),
            "sources_by_status": s,
            "voices": one("SELECT COUNT(*) FROM voices WHERE n_clips>0"),
            "clips": one("SELECT COUNT(*) FROM clips WHERE deleted=0"),
            "singing_seconds": one("SELECT COALESCE(SUM(end-start),0) FROM clips WHERE deleted=0"),
        }
