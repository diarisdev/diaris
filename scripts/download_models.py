"""
Model İndirme Aracı.

Whisper ve Pyannote modellerini Hugging Face'den indirir
ve models/ dizinine kaydeder.

Kullanım:
    python scripts/download_models.py
"""

from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

# Proje kökündeki .env dosyasını yükle
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")

# Modelleri proje kökündeki models/ dizinine kaydet
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

print("⬇️ Downloading Faster-Whisper...")
snapshot_download(
    repo_id="Systran/faster-whisper-small",
    local_dir=os.path.join(BASE_DIR, "whisper-small"),
    ignore_patterns=["*.msgpack", "*.h5"]  # We only need the .bin files
)

print("⬇️ Downloading Pyannote Segmentation...")
snapshot_download(
    repo_id="pyannote/segmentation-3.0",
    local_dir=os.path.join(BASE_DIR, "pyannote-segmentation"),
    token=HF_TOKEN
)

print("⬇️ Downloading Pyannote Speaker Embeddings...")
snapshot_download(
    repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
    local_dir=os.path.join(BASE_DIR, "pyannote-embeddings"),
    token=HF_TOKEN
)

print(f"✅ All models downloaded to: {BASE_DIR}")
