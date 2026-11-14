"""Zenodo: open-access audio/video records (CC licences) found through the public REST API."""
from __future__ import annotations

import os
import urllib.parse
from pathlib import PurePosixPath

from .base import MEDIA_EXT, MediaItem, Page, Source

LICENSE_URLS = {
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc-by-nd-4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}

SEED_TERMS = [
    "singing", "singer", "song", "songs", "choir", "chorus", "opera", "aria", "a cappella", "vocal",
    "vocals", "voice recording", "karaoke", "lullaby", "hymn", "folk song", "throat singing", "ballad",
    "concert", "music performance", "musical performance", "cover song", "lieder", "gospel", "chant",
]


class ZenodoSource(Source):
    kind = "zenodo"
    PAGE = 25            # unauthenticated Zenodo requests may not ask for more
    MAX_HITS = 10_000          # Zenodo does not page past 10k hits per query; more queries go deeper

    def __init__(self, api: str | None = None, types: tuple[str, ...] = ("video", "audio"), terms=None):
        super().__init__()
        self.api = (api or os.environ.get("ZENODO_API", "https://zenodo.org/api")).rstrip("/")
        self.types = types
        self.terms = list(terms or SEED_TERMS)
        self.limiter.min_interval = 1.1       # unauthenticated limit is ~60 requests/minute

    def seed_queries(self) -> list[str]:
        return [f"{t}|{term}" for term in self.terms for t in self.types]

    def discover(self, query: str, cursor: str | None) -> Page:
        rtype, term = query.split("|", 1)
        page = int(cursor or 1)
        data = self.get_json(f"{self.api}/records", {
            "q": term, "type": rtype, "access_right": "open", "size": self.PAGE, "page": page, "sort": "bestmatch",
        })
        hits = data.get("hits", {}).get("hits", [])
        items = []
        for h in hits:
            m = h.get("metadata", {})
            lic = (m.get("license") or {}).get("id", "") or ""
            authors = ", ".join(c.get("name", "") for c in (m.get("creators") or [])[:4])
            files = h.get("files") or []
            media = [f for f in files if PurePosixPath(f["key"]).suffix.lower() in MEDIA_EXT]
            for f in media:
                key = f["key"]
                url = f.get("links", {}).get("self") or f"{self.api}/records/{h['id']}/files/{urllib.parse.quote(key)}/content"
                title = m.get("title", "") + (f" — {key}" if len(media) > 1 else "")
                items.append(MediaItem(
                    ref=f"zenodo:{h['id']}/{key}", url=url, title=title,
                    page_url=h.get("links", {}).get("self_html") or f"https://zenodo.org/records/{h['id']}",
                    author=authors, license=lic or "open access", license_url=LICENSE_URLS.get(lic, ""),
                    size_bytes=f.get("size"), meta={"doi": h.get("doi"), "record": h["id"]},
                ))
        total = data.get("hits", {}).get("total", 0)
        more = len(hits) == self.PAGE and page * self.PAGE < min(total, self.MAX_HITS)
        return Page(items, str(page + 1) if more else None)
