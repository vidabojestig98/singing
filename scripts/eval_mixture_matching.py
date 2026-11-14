"""How much does accompaniment hurt voice matching?  (real songs, MUSDB18 test, 10 different singers)

query   = 20 s of the DRY vocal stem from the first half of a song   ("the user sings")
gallery = 30 s clips from the second half of every song, embedded as
            (a) the full MIXTURE (what a crawled video gives us)
            (b) the isolated vocal stem (oracle: what perfect vocal separation would give)
Metric: 10-way top-1 / top-3 identification accuracy (chance = 10%).
"""
import glob, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundalike.embed import VoiceEmbedder, l2norm

def dec(p, stream):
    out = subprocess.run(["ffmpeg","-v","error","-i",p,"-map",f"0:{stream}","-f","f32le","-ac","1","-ar","16000","-"],capture_output=True,check=True).stdout
    return np.frombuffer(out, np.float32)
emb = VoiceEmbedder("models/wespeaker_campplus_lm.onnx")
SR = 16000
songs = sorted(glob.glob("data/testdata/musdb/*.mp4"))
def clip_emb(w):
    _, E = emb.embed_windows(w, 3.0, 1.5); return l2norm(E.mean(0)) if len(E) else None
def active(v, a, b):          # windows where the vocal stem is clearly audible
    seg = v[int(a*SR):int(b*SR)]
    return 20*np.log10(np.sqrt((seg**2).mean()+1e-12)) > -35
Q, G_mix, G_voc = [], [], []
for si, p in enumerate(songs):
    m, v = dec(p, 0), dec(p, 4); half = len(m) // 2
    # queries: up to 4 vocal-active 20 s excerpts from the first half
    qs = [(a, a+20) for a in range(5, int(half/SR)-20, 15) if active(v, a, a+20)][:4]
    gs = [(a, a+30) for a in range(int(half/SR), int(len(m)/SR)-30, 25) if active(v, a, a+30)][:4]
    for a, b in qs:
        e = clip_emb(v[int(a*SR):int(b*SR)]);  Q.append((si, e)) if e is not None else None
    for a, b in gs:
        G_mix.append((si, clip_emb(m[int(a*SR):int(b*SR)]))); G_voc.append((si, clip_emb(v[int(a*SR):int(b*SR)])))
print(f"{len(songs)} songs, {len(Q)} queries, {len(G_mix)} gallery clips")
def evaluate(G, name):
    t1 = t3 = 0
    for si, q in Q:
        sc = {}
        for gi, g in G: sc[gi] = max(sc.get(gi, -1), float(q @ g))
        rank = sorted(sc, key=sc.get, reverse=True)
        t1 += rank[0] == si; t3 += si in rank[:3]
    print(f"  gallery = {name:28s} top-1 {t1/len(Q):.2f}   top-3 {t3/len(Q):.2f}")
    return [float(q @ g) for si, q in Q for gi, g in G if gi == si], [float(q @ g) for si, q in Q for gi, g in G if gi != si]
s1, d1 = evaluate(G_mix, "full mixture (as crawled)")
s2, d2 = evaluate(G_voc, "isolated vocals (oracle)")
print(f"  mean cosine  same song/singer: mixture {np.mean(s1):.2f} vs oracle {np.mean(s2):.2f};  different singers: mixture {np.mean(d1):.2f} vs oracle {np.mean(d2):.2f}")
