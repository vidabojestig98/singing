"""Local folders and direct media URLs: for anything you have the right to analyse."""
from __future__ import annotations

import os
from pathlib import Path

from .base import MEDIA_EXT, MediaItem, Page, Source


class FolderSource(Source):
    """Query form:  dir:<absolute directory>  (walked recursively; files are analysed in place)."""
    kind = "folder"
    PAGE = 500

    def __init__(self, dirs: list[str] | None = None):
        super().__init__()
        self.dirs = [str(Path(d).resolve()) for d in (dirs or [])]

    def seed_queries(self) -> list[str]:
        return [f"dir:{d}" for d in self.dirs]

    def discover(self, query: str, cursor: str | None) -> Page:
        root = query[len("dir:"):]
        files = sorted(
            os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if Path(f).suffix.lower() in MEDIA_EXT
        )
        start = int(cursor or 0)
        chunk = files[start : start + self.PAGE]
        items = []
        for p in chunk:
            st = os.stat(p)
            items.append(MediaItem(ref=f"folder:{p}:{st.st_size}:{int(st.st_mtime)}", url=p, title=Path(p).stem,
                                   license="local file", size_bytes=st.st_size))
        nxt = start + self.PAGE
        return Page(items, str(nxt) if nxt < len(files) else None)

    def local_path(self, row) -> Path | None:
        return Path(row["url"])


class UrlSource(Source):
    """Direct media URLs added by hand (API/CLI). Nothing to discover: items are inserted directly."""
    kind = "url"

    def discover(self, query: str, cursor: str | None) -> Page:
        return Page([], None)


class UploadSource(Source):
    """Files a user uploaded to be added to the database. Already on disk."""
    kind = "upload"

    def discover(self, query: str, cursor: str | None) -> Page:
        return Page([], None)

    def local_path(self, row) -> Path | None:
        return Path(row["url"])
