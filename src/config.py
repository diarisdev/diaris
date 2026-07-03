"""Application settings and startup helpers."""

from __future__ import annotations

import logging
import os
import site
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    default_rate: int
    default_channels: int
    frame_duration_ms: int
    output_dir: Path
    output_filename: Path
    save_audio_file: bool
    silence_limit: int
    short_silence_limit: int
    soft_chunk_duration_ms: int
    max_chunk_duration_ms: int
    pre_roll_ms: int
    chunk_overlap_ms: int
    partial_window_ms: int
    vad_aggressiveness: int
    vad_use_webrtc: bool
    silero_threshold: float
    hf_token: str | None
    local_models_dir: Path
    whisper_path: Path
    diarization_model: str
    diarization_config_path: Path
    device: str
    compute_type: str
    whisper_language: str
    whisper_no_speech_threshold: float
    whisper_logprob_threshold: float
    whisper_compression_ratio_threshold: float
    diarization_embedding_threshold: float
    diarization_warmup_ms: int
    candidate_confirmations_needed: int
    candidate_ttl: int
    candidate_self_similarity: float
    min_new_speaker_duration: float


def _resolve_whisper_path(local_models_dir: Path, device: str) -> Path:
    """Whisper model dizinini seçer.

    WHISPER_MODEL env ile açıkça verilebilir (models/ altındaki dizin adı veya
    mutlak yol). Verilmezse: GPU'da whisper-medium (varsa) tercih edilir —
    RTF ~0.1x olduğundan bolca marj var ve medium konuşma dilinde küçük modele
    göre belirgin daha düşük WER verir. CPU'da (int8) whisper-small kalır.
    """
    explicit = os.getenv("WHISPER_MODEL", "").strip()
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = local_models_dir / explicit
        return candidate

    if device == "cuda":
        medium = local_models_dir / "whisper-medium"
        if medium.is_dir():
            return medium
    return local_models_dir / "whisper-small"


def load_settings() -> Settings:
    """Read environment variables and return immutable settings."""
    load_dotenv(PROJECT_ROOT / ".env")

    output_dir = PROJECT_ROOT / "output"
    local_models_dir = PROJECT_ROOT / "models"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return Settings(
        project_root=PROJECT_ROOT,
        default_rate=int(os.getenv("DEFAULT_RATE", "48000")),
        default_channels=int(os.getenv("DEFAULT_CHANNELS", "2")),
        frame_duration_ms=int(os.getenv("FRAME_DURATION_MS", "30")),
        output_dir=output_dir,
        output_filename=output_dir / os.getenv("OUTPUT_FILENAME", "system_recorded.wav"),
        save_audio_file=_env_bool("SAVE_AUDIO_FILE", False),
        silence_limit=int(os.getenv("SILENCE_LIMIT", "30")),
        short_silence_limit=int(os.getenv("SHORT_SILENCE_LIMIT", "15")),
        soft_chunk_duration_ms=int(os.getenv("SOFT_CHUNK_DURATION_MS", "5000")),
        max_chunk_duration_ms=int(os.getenv("MAX_CHUNK_DURATION_MS", "10000")),
        # Pre-roll: VAD konuşmayı yakalamadan ÖNCEKİ son N ms ham ses chunk'ın
        # başına eklenir — WebRTC'nin kaçırdığı kelime başlangıçları kırpılmaz.
        pre_roll_ms=int(os.getenv("PRE_ROLL_MS", "240")),
        # Overlap: MAX_CHUNK_DURATION nedeniyle konuşmanın ORTASINDAN kesilen
        # chunk'larda son N ms bir sonraki chunk'a taşınır; sonraki chunk'ta bu
        # önek transkripsiyondan kırpılır (çift sayım olmaz), ama Whisper kesim
        # noktasındaki kelimeyi tam bağlamla görür.
        chunk_overlap_ms=int(os.getenv("CHUNK_OVERLAP_MS", "480")),
        # Kısmi (partial) transkripsiyon penceresi: canlı satır için buffer'ın
        # yalnızca SON bu kadar ms'i decode edilir. Tüm buffer'ı her 300 ms'de
        # yeniden decode etmek O(n^2) GPU işiydi (9 sn'lik chunk ~135 sn ses
        # decode ettiriyordu) ve final'lerin kuyrukta beklemesine yol açıyordu.
        # FINAL transkripsiyon her zaman TAM buffer'ı görür — kalite değişmez.
        partial_window_ms=int(os.getenv("PARTIAL_WINDOW_MS", "4000")),
        # VAD varsayılanları CHiME-6 taramasının (output/vad_sweep*.csv)
        # optimum bölgesine ayarlı: aggr=2 / silero=0.70. Not: Silero artık
        # kesintisiz 512-örnek akışla beslendiği için eşik semantiği eski
        # (padding'li) kuruluma göre daha keskin; yeniden tarama önerilir.
        vad_aggressiveness=int(os.getenv("VAD_AGGRESSIVENESS", "2")),
        # WebRTC katmanı AND-kapısı olarak oy kullanır. Silero artık her frame'de
        # zaten çalıştığı için (RNN durum sürekliliği) WebRTC hesaplama tasarrufu
        # sağlamaz; tek etkisi kaçırma (miss) ekleyip yanlış alarmı azaltmaktır.
        # DER profili miss-baskın olduğundan kapatmak muhtemelen nötr/daha iyi —
        # scripts/vad_webrtc_ablation.py ile A/B ölçün.
        vad_use_webrtc=_env_bool("VAD_USE_WEBRTC", True),
        silero_threshold=float(os.getenv("SILERO_THRESHOLD", "0.70")),
        hf_token=os.getenv("HF_TOKEN"),
        local_models_dir=local_models_dir,
        whisper_path=_resolve_whisper_path(local_models_dir, device),
        diarization_model=os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"),
        diarization_config_path=local_models_dir / "diarization_config.yaml",
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en"),
        # Whisper halüsinasyon filtresi eşikleri (sessizlik/gürültüde hayalet metni eler)
        whisper_no_speech_threshold=float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6")),
        whisper_logprob_threshold=float(os.getenv("WHISPER_LOGPROB_THRESHOLD", "-1.0")),
        whisper_compression_ratio_threshold=float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")),
        diarization_embedding_threshold=float(os.getenv("DIARIZATION_EMBEDDING_THRESHOLD", "0.66")),
        diarization_warmup_ms=int(os.getenv("DIARIZATION_WARMUP_MS", "20000")),
        # Aday kapısı gevşetildi (4 gözlem × ≥3sn → 2 gözlem × ≥2sn, TTL 15):
        # eski kapı o kadar sıkıydı ki warm-up'ta konuşmamış GERÇEK konuşmacılar
        # hiç oluşturulamıyor, sesleri en yakın mevcut konuşmacıya yapışıp
        # DER confusion / cpWER hatasına dönüşüyordu (AMI: 4 konuşmacıdan 2'si).
        candidate_confirmations_needed=int(os.getenv("CANDIDATE_CONFIRMATIONS_NEEDED", "2")),
        candidate_ttl=int(os.getenv("CANDIDATE_TTL", "15")),
        candidate_self_similarity=float(os.getenv("CANDIDATE_SELF_SIMILARITY", "0.78")),
        min_new_speaker_duration=float(os.getenv("MIN_NEW_SPEAKER_DURATION", "2.0")),
    )


settings = load_settings()


def configure_cuda_dll_paths() -> None:
    """Expose CUDA DLL folders installed by Python wheels or system toolkit on Windows."""
    dll_dirs_to_add: list[str] = []

    # 1. Check Python wheel nvidia packages (e.g. nvidia-cublas-cu12)
    for site_package in site.getsitepackages():
        for library in ("cublas", "cudnn"):
            lib_path = Path(site_package) / "nvidia" / library / "bin"
            if lib_path.exists():
                dll_dirs_to_add.append(str(lib_path))

    # 2. Check project-local CUDA compatibility shims (cuda_compat/)
    compat_path = PROJECT_ROOT / "cuda_compat"
    if compat_path.exists():
        dll_dirs_to_add.append(str(compat_path))

    # 3. Check system CUDA Toolkit installation
    cuda_base = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"))
    if cuda_base.exists():
        for version_dir in sorted(cuda_base.iterdir(), reverse=True):
            for bin_subdir in (version_dir / "bin" / "x64", version_dir / "bin"):
                if bin_subdir.exists():
                    dll_dirs_to_add.append(str(bin_subdir))
                    break

    for dll_dir in dll_dirs_to_add:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        except OSError as exc:
            logger.warning("Could not add CUDA DLL path %s: %s", dll_dir, exc)


def ensure_output_dir() -> Path:
    """Create and return the configured output directory."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings.output_dir


# Backwards-compatible constants used by the current modules.
DEFAULT_RATE = settings.default_rate
DEFAULT_CHANNELS = settings.default_channels
FRAME_DURATION_MS = settings.frame_duration_ms
OUTPUT_DIR = str(settings.output_dir)
OUTPUT_FILENAME = str(settings.output_filename)
SAVE_AUDIO_FILE = settings.save_audio_file
SILENCE_LIMIT = settings.silence_limit
SHORT_SILENCE_LIMIT = settings.short_silence_limit
SOFT_CHUNK_DURATION_MS = settings.soft_chunk_duration_ms
MAX_CHUNK_DURATION_MS = settings.max_chunk_duration_ms
PRE_ROLL_MS = settings.pre_roll_ms
CHUNK_OVERLAP_MS = settings.chunk_overlap_ms
PARTIAL_WINDOW_MS = settings.partial_window_ms
VAD_AGGRESSIVENESS = settings.vad_aggressiveness
VAD_USE_WEBRTC = settings.vad_use_webrtc
SILERO_THRESHOLD = settings.silero_threshold
HF_TOKEN = settings.hf_token
LOCAL_MODELS_DIR = str(settings.local_models_dir)
WHISPER_PATH = str(settings.whisper_path)
DIARIZATION_MODEL = settings.diarization_model
DIARIZATION_CONFIG_PATH = str(settings.diarization_config_path)
DEVICE = settings.device
COMPUTE_TYPE = settings.compute_type
WHISPER_LANGUAGE = settings.whisper_language
WHISPER_NO_SPEECH_THRESHOLD = settings.whisper_no_speech_threshold
WHISPER_LOGPROB_THRESHOLD = settings.whisper_logprob_threshold
WHISPER_COMPRESSION_RATIO_THRESHOLD = settings.whisper_compression_ratio_threshold
DIARIZATION_EMBEDDING_THRESHOLD = settings.diarization_embedding_threshold
DIARIZATION_WARMUP_MS = settings.diarization_warmup_ms
CANDIDATE_CONFIRMATIONS_NEEDED = settings.candidate_confirmations_needed
CANDIDATE_TTL = settings.candidate_ttl
CANDIDATE_SELF_SIMILARITY = settings.candidate_self_similarity
MIN_NEW_SPEAKER_DURATION = settings.min_new_speaker_duration
