"""Download (once) the pretrained ONNX models the engine needs, verifying SHA-256."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

HF = "https://huggingface.co"
MODELS = [
    # name, url, sha256, licence
    ("yamnet.onnx", f"{HF}/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx",
     "1510041dce24a2e9e84ec546807ac408ae496da6d1ed41bc3ccba649623f8e19", "Apache-2.0 (Google YAMNet)"),
    ("yamnet_class_map.csv", f"{HF}/zeropointnine/yamnet-onnx/resolve/main/yamnet_class_map.csv",
     "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2", "Apache-2.0"),
    ("wespeaker_campplus_lm.onnx", f"{HF}/Wespeaker/wespeaker-voxceleb-campplus-LM/resolve/main/voxceleb_CAM++_LM.onnx",
     "1068e4ac3a76bb9c769e6816ef30bf89363f6e966f1d938210cb8ed4038f8e93", "Apache-2.0 (WeSpeaker CAM++)"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ensure_models(models_dir: str | Path, quiet: bool = False) -> Path:
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, url, digest, lic in MODELS:
        dest = d / name
        if dest.exists() and sha256(dest) == digest:
            continue
        if not quiet:
            print(f"downloading {name} ({lic}) ...", file=sys.stderr)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if sha256(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}; refusing to use it")
        tmp.replace(dest)
    return d


if __name__ == "__main__":
    ensure_models(sys.argv[1] if len(sys.argv) > 1 else "models")
    print("models ready")
