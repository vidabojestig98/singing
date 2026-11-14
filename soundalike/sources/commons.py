"""Wikimedia Commons: millions of freely licensed files, with machine-readable licence + author.

Discovery uses the MediaWiki Action API (one generator query returns file titles *and* their
imageinfo), either by walking a category tree or through full-text file search. Continuation
tokens are persisted by the crawler, so a crawl can be stopped/resumed at any time.
"""
from __future__ import annotations

import html
import json
import os
import re

from .base import MediaItem, Page, Source

IMAGEINFO = {
    "prop": "imageinfo",
    "iiprop": "url|size|mime|mediatype|extmetadata",
    "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|ObjectName|ImageDescription",
}

SEED_CATEGORIES = ["Singing", "Songs", "Vocal music", "Choral music", "Choirs", "Opera", "A cappella",
                   "Folk songs", "Sacred music", "Lullabies", "Videos of singing", "Audio files of singing"]
SEED_SEARCH_TERMS = ["singing", "song", "choir", "aria", "a cappella", "vocal", "lullaby", "hymn", "folk song",
                     "karaoke", "opera", "chorus", "concert"]


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


class CommonsSource(Source):
    kind = "commons"

    def __init__(self, api: str | None = None, max_category_depth: int = 8):
        super().__init__()
        self.api = api or os.environ.get("COMMONS_API", "https://commons.wikimedia.org/w/api.php")
        self.max_depth = max_category_depth
        self.limiter.min_interval = 1.0

    def seed_queries(self) -> list[str]:
        q = [f"category:0:{c}" for c in SEED_CATEGORIES]
        for t in SEED_SEARCH_TERMS:
            q += [f"search:{t} filetype:video", f"search:{t} filetype:audio"]
        return q

    # query forms:  category:<depth>:<Category name>   |   search:<terms>
    def discover(self, query: str, cursor: str | None) -> Page:
        params = {"action": "query", "format": "json", "formatversion": "2", **IMAGEINFO}
        new_queries: list[str] = []
        if query.startswith("category:"):
            _, depth_s, name = query.split(":", 2)
            depth = int(depth_s)
            params.update({"generator": "categorymembers", "gcmtitle": f"Category:{name}", "gcmtype": "file",
                           "gcmlimit": "50"})
            if cursor is None and depth < self.max_depth:
                new_queries = [f"category:{depth + 1}:{sub}" for sub in self._subcategories(name)]
        elif query.startswith("search:"):
            params.update({"generator": "search", "gsrsearch": query[len("search:"):], "gsrnamespace": "6",
                           "gsrlimit": "50"})
        else:
            raise ValueError(query)
        if cursor:
            params.update(json.loads(cursor))
        data = self.get_json(self.api, params)
        items = []
        for page in data.get("query", {}).get("pages", []):
            ii = (page.get("imageinfo") or [None])[0]
            if not ii or ii.get("mediatype") not in ("AUDIO", "VIDEO"):
                continue
            ext = ii.get("extmetadata") or {}
            val = lambda k: (ext.get(k) or {}).get("value", "")  # noqa: E731
            title = page.get("title", "")
            items.append(MediaItem(
                ref=f"commons:{title}",
                url=ii["url"],
                title=_strip(val("ObjectName")) or title.replace("File:", ""),
                page_url=ii.get("descriptionurl", ""),
                author=_strip(val("Artist")),
                license=_strip(val("LicenseShortName")),
                license_url=_strip(val("LicenseUrl")),
                duration=ii.get("duration"),
                size_bytes=ii.get("size"),
                meta={"mime": ii.get("mime"), "mediatype": ii.get("mediatype")},
            ))
        cont = data.get("continue")
        return Page(items, json.dumps({k: v for k, v in cont.items()}) if cont else None, new_queries)

    def _subcategories(self, name: str) -> list[str]:
        out, cont = [], None
        while True:
            params = {"action": "query", "format": "json", "formatversion": "2", "list": "categorymembers",
                      "cmtitle": f"Category:{name}", "cmtype": "subcat", "cmlimit": "500"}
            if cont:
                params.update(cont)
            data = self.get_json(self.api, params)
            out += [m["title"].split(":", 1)[1] for m in data.get("query", {}).get("categorymembers", [])]
            cont = data.get("continue")
            if not cont:
                return out
