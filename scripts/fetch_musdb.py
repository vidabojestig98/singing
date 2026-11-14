"""Fetch N MUSDB18 songs (Zenodo 1117372, licence: non-commercial research) via HTTP range requests.
   split=test  -> evaluation only;  split=train -> used by train_singing_head.py --musdb"""
import sys, zipfile, pathlib, concurrent.futures as cf
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from remote_zip import HTTPRangeFile
split, n = sys.argv[1], int(sys.argv[2])
url = "https://zenodo.org/api/records/1117372/files/musdb18.zip/content"
z = zipfile.ZipFile(HTTPRangeFile(url))
infos = [i for i in z.infolist() if i.filename.startswith(split + "/") and i.filename.endswith(".mp4")]
step = max(1, len(infos) // n)
sel = infos[::step][:n]
out = pathlib.Path("data/testdata/musdb_" + split if split == "train" else "data/testdata/musdb"); out.mkdir(parents=True, exist_ok=True)
def get(i):
    d = out / pathlib.Path(i.filename).name.replace(" ", "_")
    if d.exists() and d.stat().st_size == i.file_size: return d
    d.write_bytes(zipfile.ZipFile(HTTPRangeFile(url)).read(i)); return d
with cf.ThreadPoolExecutor(6) as ex:
    for k, d in enumerate(ex.map(get, sel), 1):
        print(k, len(sel), d.name, flush=True)
