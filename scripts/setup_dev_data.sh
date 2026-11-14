#!/usr/bin/env bash
# Fetch the REAL audio used by the evaluation scripts and the integration tests (~1.5 GB, never committed).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/fetch_test_data.py all        # VocalSet singers (CC-BY 4.0) + LibriSpeech speech (CC-BY 4.0)
python scripts/fetch_more_data.py            # more speakers + FMA instrumentals
python scripts/fetch_musdb.py test 10        # real songs with isolated vocal stems (evaluation only; non-commercial licence)
# optional, only needed to re-train the singing detector with real songs:
#   python scripts/fetch_musdb.py train 40 && python scripts/train_singing_head.py --musdb
