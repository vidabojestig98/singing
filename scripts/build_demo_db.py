"""Fill a database with real singers (VocalSet, CC-BY 4.0) so the app has something to match against.

  python scripts/build_demo_db.py --root data/demo_db
Ingests two songs per singer (Caro mio ben, Dona nobis pacem) -> the third song ("Row, row, row your
boat") is then free to use as a held-out query.
"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundalike.crawler import Crawler
from soundalike.pipeline import Engine
from soundalike.sources import default_sources

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/demo_db")
ap.add_argument("--songs", default="caro,dona")
a = ap.parse_args()
eng = Engine(data_dir=a.root)
cr = Crawler(eng, default_sources(zenodo=False, commons=False), workers=2, discover=False)
files = []
for song in a.songs.split(","):
    files += sorted(f for f in glob.glob(f"data/testdata/vocalset/*/excerpts/*/*{song}*.wav") if "/spoken/" not in f)
for f in files:
    parts = f.split("/")
    eng.db.add_source("upload", f"upload:{f}", url=os.path.abspath(f), title=f"VocalSet {parts[-4]} – {parts[-1][:-4]}",
                      license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                      author="VocalSet (Wilkins et al.)", page_url="https://zenodo.org/records/1193957")
cr.start(); t0 = time.time()
while not cr.wait_idle(timeout=5):
    st = eng.db.stats(); print(f"\r{st['sources_by_status']}  voices={st['voices']} clips={st['clips']}", end="", flush=True)
cr.stop(); print("\n", eng.db.stats(), f"{time.time()-t0:.0f}s")
