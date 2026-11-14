"""Singer-identification benchmark on VocalSet (real singers).
Protocol (hard): a query recording is NEVER compared with another recording of the same
song/exercise; gallery = every other recording from every singer; 20-way singer ID.
Reports top-1 / top-3 singer accuracy; chance = 5%."""
import sys, glob, os, re, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from soundalike.audio import load_audio
from soundalike.embed import VoiceEmbedder, l2norm

models = sys.argv[1:] or ['wespeaker_resnet34_lm', 'wespeaker_campplus_lm', 'wespeaker_ecapa1024_lm']
files = sorted(glob.glob('data/testdata/vocalset/*/*/*/*.wav'))
def meta(p):
    parts = p.split('/'); singer, group, tech, fn = parts[-4], parts[-3], parts[-2], parts[-1]
    m = re.match(r'(?:[fm]\d+_)?(.*)\.wav', fn); key = m.group(1)
    task = key.split('_')[0] if group == 'excerpts' else group       # song name / exercise family
    return singer, group, tech, task
M = [meta(p) for p in files]
waves = {}
t0 = time.time()
for p in files:
    w = load_audio(p, 16000)
    waves[p] = w[: 16000 * 40]                                           # at most 40 s per file
print(f'{len(files)} files loaded in {time.time()-t0:.0f}s')
for name in models:
    emb = VoiceEmbedder(f'models/{name}.onnx'); t0 = time.time()
    E = {}
    for p in files:
        _, X = emb.embed_windows(waves[p], 4.0, 2.0)
        if len(X): E[p] = l2norm(X.mean(0))
    ps = [p for p in files if p in E]; G = np.stack([E[p] for p in ps])
    singers = np.array([meta(p)[0] for p in ps]); tasks = np.array([meta(p)[3] for p in ps]); groups = np.array([meta(p)[1] for p in ps])
    ok1 = ok3 = n = 0; by = collections.defaultdict(lambda: [0, 0])
    for i, p in enumerate(ps):
        sims = G @ G[i]
        mask = (tasks != tasks[i])                                       # different song/exercise only
        # singer score = best matching recording of that singer
        best = {}
        for j in np.where(mask)[0]:
            best[singers[j]] = max(best.get(singers[j], -9), sims[j])
        rank = sorted(best, key=best.get, reverse=True)
        n += 1; hit1 = rank[0] == singers[i]; hit3 = singers[i] in rank[:3]
        ok1 += hit1; ok3 += hit3; by[groups[i]][0] += hit1; by[groups[i]][1] += 1
    print(f'{name:26s} dim={emb.dim:3d}  top1={ok1/n:.3f}  top3={ok3/n:.3f}  n={n}  ' +
          ' '.join(f'{g}:{a/b:.2f}' for g, (a, b) in sorted(by.items())) + f'   ({time.time()-t0:.0f}s)')
