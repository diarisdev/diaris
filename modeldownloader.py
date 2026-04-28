from huggingface_hub import snapshot_download
import os

# ⚠️ PASTE YOUR TOKEN HERE
HF_TOKEN = os.getenv("HF_TOKEN")

# Choose where you want to save the models (e.g., a folder on your Desktop)
BASE_DIR = os.path.abspath("./Local_Models")

print("⬇️ Downloading Faster-Whisper...")
snapshot_download(
    repo_id="Systran/faster-whisper-small", 
    local_dir=os.path.join(BASE_DIR, "whisper-small"),
    ignore_patterns=["*.msgpack", "*.h5"] # We only need the .bin files
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