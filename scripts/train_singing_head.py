"""Train the small singing/speech/music/other head that sits on top of YAMNet embeddings.

All components are REAL recordings; only the mixing is synthetic:
  singing     : VocalSet (CC-BY 4.0) dry, and mixed over FMA instrumental music at random SNR / reverb / noise
                (VocalSet's "spoken" technique - the same singers TALKING the lyrics - is used as hard-negative speech)
  speech      : LibriSpeech test-clean/other (CC-BY 4.0), alone, over music, over noise
  music       : FMA "Instrumental" clips (vocal-free music)
  other       : synthetic noise / hum / near-silence / multi-talker babble
  real songs  : with --musdb, MUSDB18 *train* songs (isolated vocal stem + accompaniment stems; licence is
                non-commercial research) re-mixed at random vocal levels; frame labels come from the stem.
                The MUSDB *test* songs are never used for training - they are the honest benchmark.

Held-out singers, speakers and music tracks are used to report honest accuracy; the shipped head
is then re-fit on everything with the same hyper-parameters.

    python scripts/train_singing_head.py            # writes soundalike/assets/singing_head.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soundalike.audio import load_audio  # noqa: E402
from soundalike.singing import CLASSES, PITCH_BOOST, Head, Yamnet, frame_features  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

SR = 16000
CLIP = 8 * SR
rng = np.random.default_rng(0)
ROOT = "data/testdata"


def rms(x):
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def mix(voice, bg, snr_db):
    g = rms(voice) / (rms(bg) * 10 ** (snr_db / 20))
    return voice + g * bg


def reverb(x, rt60):
    n = int(SR * min(rt60 * 1.2, 1.2))
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir /= np.linalg.norm(ir)
    return fftconvolve(x, ir)[: x.size].astype(np.float32)


def crop(x, n=CLIP, min_rms=0.004, tries=20):
    if x.size <= n:
        return np.pad(x, (0, n - x.size))
    for _ in range(tries):
        s = int(rng.integers(0, x.size - n))
        c = x[s : s + n]
        if rms(c) > min_rms:
            return c
    return c


def noise(kind):
    w = rng.standard_normal(CLIP)
    if kind == "pink":
        f = np.fft.rfft(w)
        f /= np.sqrt(np.maximum(np.arange(f.size), 1))
        w = np.fft.irfft(f, CLIP)
    elif kind == "brown":
        w = np.cumsum(w)
        w -= np.linspace(w[0], w[-1], w.size)
    elif kind == "hum":
        t = np.arange(CLIP) / SR
        w = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.02 * w
    return (w / (np.abs(w).max() + 1e-9)).astype(np.float32)


class Pools:
    def __init__(self):
        singers = sorted({p.split("/")[-4] for p in glob.glob(f"{ROOT}/vocalset/*/*/*/*.wav")})
        self.test_singers = {s for i, s in enumerate(singers) if i % 3 == 0}
        self.sing = {"train": [], "test": []}
        self.spoken = {"train": [], "test": []}     # VocalSet "spoken" technique: the singers TALKING the lyrics
        for p in sorted(glob.glob(f"{ROOT}/vocalset/*/*/*/*.wav")):
            split = "test" if p.split("/")[-4] in self.test_singers else "train"
            (self.spoken if p.split("/")[-2] == "spoken" else self.sing)[split].append(p)
        sp = sorted(glob.glob(f"{ROOT}/speech_big/*.flac")) or sorted(glob.glob(f"{ROOT}/speech/*.flac"))
        spk = lambda p: os.path.basename(p).split("_")[1].split("-")[0]  # noqa: E731
        speakers = sorted({spk(p) for p in sp})
        test_spk = {s for i, s in enumerate(speakers) if i % 4 == 0}
        self.speech = {"train": [], "test": []}
        for p in sp:
            self.speech["test" if spk(p) in test_spk else "train"].append(p)
        ins = sorted(glob.glob(f"{ROOT}/music/instrumental/*.ogg"))
        self.music = {"train": [], "test": []}
        for i, p in enumerate(ins):
            self.music["test" if i % 4 == 0 else "train"].append(p)
        with ThreadPoolExecutor(8) as ex:   # decode everything in parallel
            for d in (self.sing, self.spoken, self.speech, self.music):
                for k in d:
                    d[k] = list(ex.map(load_audio, d[k]))
        self.songs = {"train": [], "test": []}
        if USE_MUSDB:
            files = sorted(glob.glob(f"{ROOT}/musdb_train/*.mp4"))
            def load_song(p):
                st = [np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", f"0:{k}", "-f", "f32le",
                      "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout, np.float32) for k in (1, 2, 3, 4)]
                n = min(len(x) for x in st)
                return (st[0][:n] + st[1][:n] + st[2][:n]).astype(np.float32), st[3][:n].copy()
            with ThreadPoolExecutor(6) as ex:
                for i, song in enumerate(ex.map(load_song, files)):
                    self.songs["test" if i % 5 == 0 else "train"].append(song)
        print({k: {s: len(v) for s, v in d.items()} for k, d in
               dict(sing=self.sing, spoken=self.spoken, speech=self.speech, music=self.music, songs=self.songs).items()}, flush=True)


EXTRA_DB = None


def load_extra(db_dir: str):
    """Hard examples mined from a live crawl: clips cut from real sources, labelled by scripts/hard_negatives.json.
    Returns {"train": [(wave, label)], "test": [...]}; the split is by SOURCE id parity so held-out sources are unseen."""
    import json
    import sqlite3
    rules = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hard_negatives.json")))["rules"]
    con = sqlite3.connect(os.path.join(db_dir, "sound.db")); con.row_factory = sqlite3.Row
    out = {"train": [], "test": []}
    for r in con.execute("SELECT clips.id cid, clips.source_id sid, sources.title t FROM clips JOIN sources ON sources.id=clips.source_id WHERE deleted=0"):
        lab = next((x["label"] for x in rules if x["match"] in r["t"]), None)
        if lab is None:
            continue
        f = f"{db_dir}/clips/{r['cid'] // 1000:05d}/{r['cid']}.mp4"
        out["test" if r["sid"] % 2 else "train"].append((load_audio(f), lab))
    print("extra (mined) clips:", {k: len(v) for k, v in out.items()}, flush=True)
    return out


def maybe_reverb(x, p=0.35):
    return reverb(x, rng.uniform(0.2, 1.0)) if rng.random() < p else x


HOPS, WINS = 7680, 15360


def stem_frame_labels(acc, voc):
    """Per YAMNet frame: 0 singing if the vocal stem is audible, else 2 music, else 3 other."""
    n = 1 + (len(voc) - WINS) // HOPS
    idx = np.arange(WINS)[None, :] + HOPS * np.arange(n)[:, None]
    rv = 20 * np.log10(np.sqrt((voc[idx] ** 2).mean(1) + 1e-12))
    ra = 20 * np.log10(np.sqrt((acc[idx] ** 2).mean(1) + 1e-12))
    return np.where(rv > -40, 0, np.where(ra > -50, 2, 3))


def gen(pools: Pools, split: str, n_each: dict[str, int]):
    pick = lambda lst: lst[int(rng.integers(len(lst)))]  # noqa: E731
    X, y = [], []
    for kind, n in n_each.items():
        for _ in range(n):
            if kind == "sing_dry":
                v = maybe_reverb(crop(pick(pools.sing[split]))); lab = 0; w = v
            elif kind == "sing_music":
                v = maybe_reverb(crop(pick(pools.sing[split]))); m = crop(pick(pools.music[split]))
                w = mix(v, m, rng.uniform(-3, 14)); lab = 0
            elif kind == "sing_noise":
                v = crop(pick(pools.sing[split])); w = mix(v, noise(pick(["white", "pink", "brown"])), rng.uniform(5, 25)); lab = 0
            elif kind == "speech":
                v = maybe_reverb(crop(pick(pools.speech[split]), min_rms=0.01)); w = v; lab = 1
            elif kind == "spoken":
                v = maybe_reverb(crop(pick(pools.spoken[split]))); w = v; lab = 1
            elif kind == "spoken_music":
                v = maybe_reverb(crop(pick(pools.spoken[split]))); m = crop(pick(pools.music[split]))
                w = mix(v, m, rng.uniform(0, 15)); lab = 1
            elif kind == "speech_music":
                v = maybe_reverb(crop(pick(pools.speech[split]), min_rms=0.01)); m = crop(pick(pools.music[split]))
                w = mix(v, m, rng.uniform(0, 18)); lab = 1
            elif kind == "speech_noise":
                v = crop(pick(pools.speech[split]), min_rms=0.01); w = mix(v, noise(pick(["white", "pink", "brown"])), rng.uniform(3, 25)); lab = 1
            elif kind == "music":
                w = maybe_reverb(crop(pick(pools.music[split]))); lab = 2
            elif kind == "noise":
                w = noise(pick(["white", "pink", "brown", "hum"])) * rng.uniform(0.01, 0.5); lab = 3
            elif kind == "babble":
                w = sum(crop(pick(pools.speech[split]), min_rms=0.01) for _ in range(int(rng.integers(5, 9)))); lab = 3
            elif kind == "real_mix":
                acc, voc = pick(pools.songs[split])
                s0 = int(rng.integers(0, len(acc) - CLIP)); g = 10 ** (rng.uniform(-8, 4) / 20)
                a, v = acc[s0 : s0 + CLIP], voc[s0 : s0 + CLIP] * g
                w = a + v; lab = stem_frame_labels(a, v)
            elif kind == "speech_real_music":
                acc, _ = pick(pools.songs[split])
                s0 = int(rng.integers(0, len(acc) - CLIP))
                v = maybe_reverb(crop(pick(pools.speech[split]), min_rms=0.01))
                w = mix(v, acc[s0 : s0 + CLIP], rng.uniform(0, 18)); lab = 1
            elif kind == "silence":
                w = rng.standard_normal(CLIP).astype(np.float32) * rng.uniform(1e-5, 2e-3); lab = 3
            else:
                raise ValueError(kind)
            w = w / (np.abs(w).max() + 1e-9) * rng.uniform(0.15, 0.95)
            X.append(w.astype(np.float32)); y.append(lab)
    return X, y


def embed_all(yam, waves, labels):
    with ThreadPoolExecutor(4) as ex:
        feats = list(ex.map(lambda w: frame_features(yam, w)[1], waves))
    Y = [np.full(len(e), l) if np.isscalar(l) else np.asarray(l)[: len(e)] for e, l in zip(feats, labels)]
    return np.concatenate(feats), np.concatenate(Y)


def fit(X, y, C, K=4):
    mean, std = X.mean(0), X.std(0) + 1e-6
    std[1024:] /= PITCH_BOOST
    Z = (X - mean) / std
    cw = len(y) / (K * np.bincount(y, minlength=K))
    sw = cw[y]
    d = Z.shape[1]

    def f(theta):
        W = theta[: K * d].reshape(K, d); b = theta[K * d :]
        s = Z @ W.T + b; s -= s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1)); logp = s - lse[:, None]
        loss = -(sw * logp[np.arange(len(y)), y]).sum() / sw.sum() + 0.5 / C * (W**2).sum() / len(y)
        p = np.exp(logp); p[np.arange(len(y)), y] -= 1; p *= sw[:, None] / sw.sum()
        gW = p.T @ Z + W / (C * len(y)); gb = p.sum(0)
        return loss, np.concatenate([gW.ravel(), gb])

    r = minimize(f, np.zeros(K * d + K), jac=True, method="L-BFGS-B", options={"maxiter": 400})
    return Head(mean, std, r.x[: K * d].reshape(K, d), r.x[K * d :])


def report(name, head, X, y):
    p = head.proba(X); pred = p.argmax(1)
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    print(f"  {name:12s} frame acc {np.mean(pred == y):.3f}  recall per class " +
          " ".join(f"{c}={r:.2f}" for c, r in zip(CLASSES, rec)))
    return cm


USE_MUSDB = False


def main():
    global USE_MUSDB, EXTRA_DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb", action="store_true", help="also train on real songs from data/testdata/musdb_train")
    ap.add_argument("--extra-db", help="a crawl database whose clips are mined as hard examples (see hard_negatives.json)")
    a = ap.parse_args()
    USE_MUSDB, EXTRA_DB = a.musdb, a.extra_db
    t0 = time.time()
    yam = Yamnet("models/yamnet.onnx", "models/yamnet_class_map.csv")
    pools = Pools()
    mix_train = dict(sing_dry=500, sing_music=500, sing_noise=120, speech=380, speech_music=280, speech_noise=120,
                     spoken=160, spoken_music=60,
                     music=450, noise=130, babble=80, silence=40)
    if USE_MUSDB:
        mix_train.update(real_mix=2200, speech_real_music=250)
    mix_test = {k: max(10, v // 4) for k, v in mix_train.items()}
    Xtr_w, ytr_w = gen(pools, "train", mix_train)
    Xte_w, yte_w = gen(pools, "test", mix_test)
    Xtr, ytr = embed_all(yam, Xtr_w, ytr_w)
    Xte, yte = embed_all(yam, Xte_w, yte_w)
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    if EXTRA_DB:
        ex = load_extra(EXTRA_DB)
        for split in ("train", "test"):
            if not ex[split]:
                continue
            Xe, ye = embed_all(yam, [w for w, _ in ex[split]], [l for _, l in ex[split]])
            rep = 4 if split == "train" else 1          # hard examples are few: replicate for training
            if split == "train":
                Xtr, ytr = np.concatenate([Xtr] + [Xe] * rep), np.concatenate([ytr] + [ye] * rep)
            else:
                Xte, yte = np.concatenate([Xte, Xe]), np.concatenate([yte, ye])
                Xex_te, yex_te = Xe, ye
    print(f"features: train {Xtr.shape}, test {Xte.shape}  ({time.time()-t0:.0f}s)", flush=True)
    best = None
    for C in (0.01,):
        h = fit(Xtr, ytr, C)
        acc = np.mean(h.proba(Xte).argmax(1) == yte)
        print(f"C={C}: held-out frame accuracy {acc:.3f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, C, h)
    acc, C, h = best
    print(f"best C={C}; held-out detail (singers/speakers/tracks unseen in training):")
    report("held-out", h, Xte, yte)
    if EXTRA_DB and "Xex_te" in globals():
        report("mined held-out sources", h, Xex_te, yex_te)
    # final model re-fit on train+test pools
    head = fit(np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]), C)
    out = "soundalike/assets/singing_head.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    head.save(out)
    print("saved", out, f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
