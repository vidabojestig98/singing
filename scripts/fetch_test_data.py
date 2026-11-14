"""Download REAL audio used only for evaluation/tests (never committed):

  * VocalSet (CC-BY 4.0, Zenodo 1193957): 20 singers -> ground truth for singer matching.
    We pull only the song "excerpts" plus a few scales/arpeggios/long-tones per singer
    using HTTP range requests (the whole archive is 2 GB).
  * LibriSpeech dummy split (CC-BY 4.0, Hugging Face hf-internal-testing): real speech.

Usage: python scripts/fetch_test_data.py [vocalset|speech|all]
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import HTTPRangeFile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "testdata"
VOCALSET_URL = "https://zenodo.org/api/records/1193957/files/VocalSet.zip/content"


def fetch_vocalset(extra_per_singer: int = 6) -> None:
    out = ROOT / "vocalset"
    out.mkdir(parents=True, exist_ok=True)
    f = HTTPRangeFile(VOCALSET_URL)
    z = zipfile.ZipFile(f)
    wavs = [i for i in z.infolist() if i.filename.endswith(".wav") and not i.is_dir()]
    chosen = []
    per_singer_extra: dict[str, int] = {}
    for i in wavs:
        parts = i.filename.split("/")  # FULL/<singer>/<group>/...
        singer, group = parts[1], parts[2]
        if group == "excerpts":
            chosen.append(i)
        elif group in ("scales", "arpeggios", "long_tones"):
            n = per_singer_extra.get(singer, 0)
            if n < extra_per_singer and hash(i.filename) % 7 == 0:
                per_singer_extra[singer] = n + 1
                chosen.append(i)
    print(f"vocalset: {len(chosen)} files, {sum(i.compress_size for i in chosen)/1e6:.0f} MB compressed")

    def get(info):
        dest = out / info.filename.replace("FULL/", "")
        if dest.exists() and dest.stat().st_size == info.file_size:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        # one range file (and HTTP session) per thread
        zz = zipfile.ZipFile(HTTPRangeFile(VOCALSET_URL))
        with zz.open(info) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        return dest

    done = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for _ in ex.map(get, chosen):
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(chosen)}", flush=True)
    print("vocalset done ->", out)


def fetch_speech() -> None:
    out = ROOT / "speech"
    out.mkdir(parents=True, exist_ok=True)
    api = "https://huggingface.co/api/datasets/hf-internal-testing/librispeech_asr_dummy"
    info = requests.get(api, timeout=60).json()
    parquet = [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".parquet")]
    print("speech parquet files:", parquet)
    import pyarrow.parquet as pq  # optional dependency, only for this script

    n = 0
    for name in parquet:
        url = f"https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/{name}"
        data = requests.get(url, timeout=300).content
        table = pq.read_table(io.BytesIO(data)).to_pylist()
        for row in table:
            audio = row["audio"]
            (out / f"{row.get('id', n)}.flac").write_bytes(audio["bytes"])
            n += 1
    print("speech files:", n, "->", out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("vocalset", "all"):
        fetch_vocalset()
    if what in ("speech", "all"):
        fetch_speech()
