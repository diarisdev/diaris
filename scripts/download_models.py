"""Download local model assets used by Audio-process."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_SPECS = [
    {
        "name": "Faster-Whisper",
        "repo_id": "Systran/faster-whisper-small",
        "local_dir": MODELS_DIR / "whisper-small",
        "requires_token": False,
        "ignore_patterns": ["*.msgpack", "*.h5"],
    },
    {
        "name": "Pyannote Segmentation",
        "repo_id": "pyannote/segmentation-3.0",
        "local_dir": MODELS_DIR / "pyannote-segmentation",
        "requires_token": True,
        "ignore_patterns": None,
    },
    {
        "name": "Pyannote Speaker Embeddings",
        "repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "local_dir": MODELS_DIR / "pyannote-embeddings",
        "requires_token": True,
        "ignore_patterns": None,
    },
    {
        "name": "NLLB-200 Translation",
        "repo_id": "Tushe/nllb-200-600M-ct2-int8",
        "local_dir": MODELS_DIR / "ctranslate2-nllb-200-distilled-600M",
        "requires_token": False,
        "ignore_patterns": None,
    },
]


def _has_files(path: Path) -> bool:
    """Check if directory contains actual model weight files (not just metadata)."""
    if not path.exists():
        return False
    # Model dosya uzantıları — bunlardan en az biri olmalı
    model_patterns = ["*.bin", "*.safetensors", "*.ckpt", "*.pt"]
    for pattern in model_patterns:
        if list(path.glob(pattern)):
            return True
    # Whisper modeli için: model.bin veya config.json yeterli
    if (path / "config.json").exists():
        return True
    return False


def download_models(force: bool = False) -> bool:
    """Download missing model folders. Returns True when all required assets exist."""
    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.getenv("HF_TOKEN")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in MODEL_SPECS:
        name = spec["name"]
        local_dir = spec["local_dir"]

        if _has_files(local_dir) and not force:
            print(f"✅ {name} zaten mevcut: {local_dir}")
            continue

        if spec["requires_token"] and not hf_token:
            print(f"❌ {name} için HF_TOKEN gerekli.")
            print("   .env içine HF_TOKEN=... ekleyin ve Hugging Face model erişimlerini kabul edin.")
            return False

        print(f"⬇️ {name} indiriliyor...")
        try:
            snapshot_download(
                repo_id=spec["repo_id"],
                local_dir=str(local_dir),
                token=hf_token if spec["requires_token"] else None,
                ignore_patterns=spec["ignore_patterns"],
            )
        except Exception as exc:
            print(f"❌ {name} indirilemedi: {exc}")
            return False

    print(f"✅ Modeller hazır: {MODELS_DIR}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if download_models(force=False) else 1)
