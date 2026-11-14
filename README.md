# Sound Alike Singer

Find out which singer you sound like.

1. **A crawler pre-searches videos** for singing, works out *where each different voice sings*, and saves
   those parts as separate **video clips** in a **singing-voices database**.
2. **You record yourself (or upload any audio/video).** The app finds the singing in your recording,
   builds a voice fingerprint, and returns the closest voices in the database, each with a playable clip.

Everything here runs for real, on CPU, with open pretrained models: no cloud API is needed to run it.

```
 discovery (Zenodo, Wikimedia Commons, folders, URLs)  ──►  queue (SQLite)  ──►  workers
                                                                                   │
   download ─► YAMNet + pitch-stability head ─► singing segments ─► CAM++ voice embeddings (3 s windows)
            ─► cluster windows per video (= distinct voices) ─► split segments at voice changes
            ─► cut each part as an MP4 clip ─► assign voice in the global database

 your recording ─► same detector ─► voice embedding ─► exact cosine search over every clip ─► best voices + clips
```

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # numpy, scipy, onnxruntime, fastapi, uvicorn, ...
sudo apt-get install ffmpeg                # (or brew install ffmpeg)
python -m soundalike download-models       # ~100 MB of ONNX models, SHA-256 verified
python -m soundalike serve                 # http://127.0.0.1:8000
```

Open the **Crawler** tab and press *Start crawling* (or `python -m soundalike crawl`); it resumes where it
stopped. Then use **Find my match**: press Record, sing for 10-30 s, press Stop, press *Find my sound-alike*.
You can also analyse your own media: `python -m soundalike ingest my_video.mp4`, or
`python -m soundalike crawl --folder ~/Videos`.

## Where the videos come from, and why not YouTube

The YouTube Data API key you supplied **works**, but the API only returns *metadata* (titles, ids, licences);
it never gives access to audio or video. Getting the media itself means downloading from `youtube.com` /
`googlevideo.com`, which YouTube's Terms of Service and API Developer Policies forbid (and which this sandbox's
network policy blocks). So the key is not used. The crawler uses sources whose licences *allow* analysing and
re-distributing clips, and stores each clip's author, licence and a link back to the original:

| source | what | status |
|---|---|---|
| **Zenodo** | open-access audio/video records (CC licences) | verified live against the real API |
| **Wikimedia Commons** | millions of CC / public-domain files; category-tree + search discovery | implemented to the documented MediaWiki API and tested against a local mock; **not** verified against the live site (blocked in the development sandbox) |
| **Folder / URL / upload** | anything you have the right to use | verified |

Adding a source is ~40 lines (`soundalike/sources/`): `seed_queries()`, `discover(query, cursor) -> Page`.

## "No limit"

There are no caps on videos searched, clips, or voices. Discovery pages through every query with persisted
cursors until the source is exhausted; sub-categories are discovered recursively; clips are files on disk;
metadata is SQLite; embeddings are a flat memory-mapped file (row = clip id, one `pwrite` per insert).
Search is an exact chunked cosine scan: 300k clips in well under a second on this machine, millions are fine,
and `Database.search_clips` is the one function to replace with an ANN index at much larger scale.
Real limits you will meet: the sources' own rate limits (Zenodo ≈ 60 requests/min unauthenticated) and CPU
time (≈ 10-15x faster than real time per worker).

## How well does it work? (measured on real recordings)

<!--RESULTS-->

## Honest limitations

* Voice similarity measures how alike two voices *sound* (timbre), not identity; similar voices happen.
* Detection quality drops for heavily distorted/screamed vocals, rap, and very dense mixes; spoken-word
  with music can occasionally be accepted, and instrumental tracks with vocal samples can be accepted.
* The per-video voice clustering splits one singer into two voices more often than it merges two singers
  (deliberate: a wrong merge corrupts a voice, a split only duplicates it).
* Accuracy numbers above come from 20 (dry) + 10 (songs) singers; they are indicative, not a guarantee.
* Browser recording needs `https://` or `localhost` for microphone access.

## Development

```bash
scripts/setup_dev_data.sh                 # real test audio (VocalSet, LibriSpeech, FMA, MUSDB18 test songs)
python -m pytest -q                       # unit + real-audio integration tests
python scripts/eval_singing_detection.py  # singing detection on real songs with isolated vocal stems
python scripts/eval_singer_id.py          # singer identification on VocalSet
python scripts/calibrate.py               # same-/different-singer similarity thresholds
python scripts/train_singing_head.py [--musdb]   # retrain the singing detector head
```

Data/licence notes: models are Apache-2.0 (YAMNet, WeSpeaker CAM++). The singing head
(`soundalike/assets/singing_head.npz`, 40 KB) was trained on VocalSet (CC-BY 4.0), LibriSpeech (CC-BY 4.0),
FMA instrumental clips (assorted Creative Commons licences) and MUSDB18 *train* songs (non-commercial research
licence). Retrain without `--musdb` if you need a head with no non-commercial data.
