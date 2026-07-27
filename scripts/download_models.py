"""Download local model assets used by Diaris."""

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


SILERO_ONNX_TARGET = MODELS_DIR / "silero_vad.onnx"
# Silero VAD ONNX. torch.hub önbelleğinde zaten varsa oradan kopyalanır (ağ
# gerekmez); yoksa resmî depodan indirilir. Bu dosya, çalışma anında GitHub'a
# çıkan torch.hub çağrısını ortadan kaldırır — paketleme (.exe) için şart.
SILERO_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_HUB_CACHE = (
    Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
    / "src" / "silero_vad" / "data" / "silero_vad.onnx"
)


def ensure_silero_onnx(force: bool = False) -> bool:
    """models/silero_vad.onnx dosyasını hazırlar (kopyala ya da indir)."""
    if SILERO_ONNX_TARGET.is_file() and not force:
        print(f"✅ Silero VAD (ONNX) zaten mevcut: {SILERO_ONNX_TARGET}")
        return True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if SILERO_HUB_CACHE.is_file():
        import shutil
        shutil.copy2(SILERO_HUB_CACHE, SILERO_ONNX_TARGET)
        print(f"✅ Silero VAD (ONNX) torch.hub önbelleğinden kopyalandı: {SILERO_ONNX_TARGET}")
        return True

    print("⬇️ Silero VAD (ONNX) indiriliyor...")
    try:
        import urllib.request
        urllib.request.urlretrieve(SILERO_ONNX_URL, SILERO_ONNX_TARGET)
        print(f"✅ Silero VAD (ONNX) indirildi: {SILERO_ONNX_TARGET}")
        return True
    except Exception as exc:
        print(f"⚠️  Silero VAD (ONNX) indirilemedi: {exc}")
        print("   Sistem torch.hub yoluna düşecek (çalışır ama daha yavaş ve ağ ister).")
        return False


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

    # Silero VAD (ONNX) — zorunlu değil; başarısız olursa torch.hub yoluna düşülür.
    ensure_silero_onnx(force=force)

    print(f"✅ Modeller hazır: {MODELS_DIR}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if download_models(force=False) else 1)
