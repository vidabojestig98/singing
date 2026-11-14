"""More real audio for training/evaluating the singing detector (never committed).
 - LibriSpeech test-clean / test-other (CC-BY 4.0): many real speakers -> 'speech' negatives
 - FMA 'Instrumental' clips (via lewtun/music_genres_small): vocal-free music -> accompaniment / negatives
"""
import io, pathlib, re, sys, collections, requests
import pyarrow.parquet as pq
ROOT = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'testdata'
def get(url, dest):
    if dest.exists(): return
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for ch in r.iter_content(1 << 20): f.write(ch)
tmp = ROOT / '_tmp'; tmp.mkdir(parents=True, exist_ok=True)
sp = ROOT / 'speech_big'; sp.mkdir(exist_ok=True)
for sub in ('clean', 'other'):
    p = tmp / f'ls_{sub}.parquet'
    get(f'https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/{sub}/test/0000.parquet', p)
    rows = pq.read_table(p).to_pylist(); n = 0
    spk = collections.Counter()
    for r in rows:
        s = r['speaker_id']
        if spk[s] >= 12: continue          # up to 12 utterances per speaker
        spk[s] += 1
        (sp / f"{sub}_{r['id']}.flac").write_bytes(r['audio']['bytes']); n += 1
    print(sub, n, 'utterances from', len(spk), 'speakers')
    p.unlink()
mu = ROOT / 'music'; (mu / 'instrumental').mkdir(parents=True, exist_ok=True)
p = tmp / 'fma.parquet'
get('https://huggingface.co/datasets/lewtun/music_genres_small/resolve/main/data/train-00000-of-00001-63d68663287b1638.parquet', p)
n = 0
for r in pq.read_table(p).to_pylist():
    g = re.sub(r'[^a-z0-9]+', '_', r['genre'].lower()).strip('_')
    d = mu / g; d.mkdir(exist_ok=True)
    dest = d / f"{g}_{r['song_id']}.ogg"
    if not dest.exists(): dest.write_bytes(r['audio']['bytes']); n += 1
print('music clips written:', n)
p.unlink()
