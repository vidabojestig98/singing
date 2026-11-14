"""Tiny read-only seekable file object over HTTP range requests, so `zipfile`
can list and extract individual members of a huge remote archive without
downloading all of it. Used only by the test-data scripts."""
from __future__ import annotations

import io
import time

import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache_start = 0
        self._cache = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                r = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120
                )
                if r.status_code == 206:
                    return r.content
                r.raise_for_status()
                raise IOError(f"unexpected status {r.status_code}")
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        raise IOError("unreachable")

    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        # serve from the one-block cache when possible (zipfile does many small reads)
        if self._cache_start <= self.pos and end < self._cache_start + len(self._cache):
            off = self.pos - self._cache_start
            data = self._cache[off : off + (end - self.pos + 1)]
        else:
            want_end = min(max(end, self.pos + self.block - 1), self.size - 1)
            blob = self._fetch(self.pos, want_end)
            self._cache_start, self._cache = self.pos, blob
            data = blob[: end - self.pos + 1]
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)
