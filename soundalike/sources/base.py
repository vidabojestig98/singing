"""Source adapters: how the crawler finds media and gets it onto local disk.

A source exposes *queries* (a search term, a category, a folder...). Each query is paged through
with an opaque cursor that the crawler persists, so discovery is resumable and unbounded: it just
keeps going until the source reports it has no more pages."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

MEDIA_EXT = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".ogv", ".mpg", ".mpeg", ".flv", ".wmv", ".3gp",
             ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif"}

CONTACT = os.environ.get("SOUNDALIKE_CONTACT", "set SOUNDALIKE_CONTACT=<url-or-email> to identify your crawler")
USER_AGENT = f"SoundAlikeSinger/0.1 (open-license singing crawler; {CONTACT}) python-requests"


class SkipSource(Exception):
    """This item cannot / should not be analysed (too large, no audio...). Not a failure: recorded as 'skipped'."""


@dataclass
class MediaItem:
    ref: str                       # stable unique id, e.g. "zenodo:123/file.mp4"
    url: str                       # where to download from
    title: str = ""
    page_url: str = ""
    author: str = ""
    license: str = ""
    license_url: str = ""
    duration: float | None = None
    size_bytes: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Page:
    items: list[MediaItem]
    next_cursor: str | None            # None == this query is exhausted
    new_queries: list[str] = field(default_factory=list)


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._t = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._t + self.min_interval - now
            if delay > 0:
                time.sleep(delay)
            self._t = time.monotonic()


class Source:
    kind = "base"
    # a per-file safety valve (disk protection), NOT a limit on the database; raise it with SOUNDALIKE_MAX_DOWNLOAD_GB
    max_download_bytes = int(float(os.environ.get("SOUNDALIKE_MAX_DOWNLOAD_GB", "3")) * 1024**3)

    def __init__(self):
        self.http = requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT
        self.limiter = RateLimiter(1.0)

    # --- discovery ---
    def seed_queries(self) -> list[str]:
        return []

    def discover(self, query: str, cursor: str | None) -> Page:
        raise NotImplementedError

    # --- fetching ---
    def local_path(self, row) -> Path | None:
        """Sources that already live on local disk return their path (nothing is copied or deleted)."""
        return None

    def download(self, row, dest: Path) -> None:
        url = row["url"]
        for attempt in range(4):
            try:
                self.limiter.wait()
                with self.http.get(url, stream=True, timeout=(15, 120)) as r:
                    if r.status_code in (429, 503):
                        time.sleep(min(60, 5 * (attempt + 1)))
                        continue
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length") or 0)
                    if total and total > self.max_download_bytes:
                        raise SkipSource(f"file is {total / 1e9:.1f} GB, above the {self.max_download_bytes / 1e9:.0f} GB per-file "
                                         "download cap (set SOUNDALIKE_MAX_DOWNLOAD_GB to raise it)")
                    n = 0
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            n += len(chunk)
                            if n > self.max_download_bytes:
                                raise SkipSource("file exceeds the per-file download cap")
                            f.write(chunk)
                    return
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 3:
                    raise RuntimeError(f"download failed: {e}") from e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("download failed after retries")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(5):
            self.limiter.wait()
            r = self.http.get(url, params=params, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(min(120, 10 * (attempt + 1)))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"rate limited by {url}")
