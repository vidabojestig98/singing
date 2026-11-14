import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "testdata"
MODELS = ROOT / "models"
have_models = (MODELS / "yamnet.onnx").exists() and (MODELS / "wespeaker_campplus_lm.onnx").exists()
have_data = (DATA / "vocalset").exists() and (DATA / "speech_big").exists() and (DATA / "music" / "instrumental").exists()

needs_models = pytest.mark.skipif(not have_models, reason="run `python -m soundalike download-models`")
needs_data = pytest.mark.skipif(not (have_models and have_data),
                                reason="real test audio missing: run scripts/fetch_test_data.py and scripts/fetch_more_data.py")


@pytest.fixture(scope="session")
def engine(tmp_path_factory):
    from soundalike.pipeline import Engine
    return Engine(data_dir=tmp_path_factory.mktemp("db"))


@pytest.fixture(scope="session")
def test_video(tmp_path_factory):
    from make_test_video import build
    path, timeline = build(str(tmp_path_factory.mktemp("vid") / "video1"))
    return path, timeline


def vocalset(singer: str, song: str, technique: str = "straight") -> str:
    """A SUNG VocalSet excerpt. (Beware: sorted(glob(...))[0] picks the 'spoken' technique - singers talking the lyrics.)"""
    hits = list((DATA / "vocalset" / singer / "excerpts" / technique).glob(f"*{song}*.wav"))
    assert hits, (singer, song, technique)
    return str(hits[0])
