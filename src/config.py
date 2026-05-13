"""
Merkezi konfigürasyon modülü.
Tüm sabitler, ortam değişkenleri ve cihaz ayarları burada tanımlanır.
Hardcoded değer yoktur — ayarlar ortam değişkenleri ile özelleştirilebilir.
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()

# --- Proje Kök Dizini ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Ses Formatı Sabitleri ---
# Bu değerler varsayılandır. Cihaz algılandığında gerçek değerler
# device.auto_detect_device() tarafından döndürülür.
DEFAULT_RATE = 48000
DEFAULT_CHANNELS = 2
FRAME_DURATION_MS = 30

# --- Çıktı Ayarları ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, os.getenv("OUTPUT_FILENAME", "system_recorded.wav"))

# --- Sessizlik Algılama ---
SILENCE_LIMIT = int(os.getenv("SILENCE_LIMIT", "50"))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
SILERO_THRESHOLD = float(os.getenv("SILERO_THRESHOLD", "0.25"))

# --- AI Model Ayarları ---
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
WHISPER_PATH = os.path.join(LOCAL_MODELS_DIR, "whisper-small")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# --- Whisper Ayarları ---
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
