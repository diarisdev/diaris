"""
src - Audio-process Modüler Paket

Alt Paketler:
    audio       : Ses donanım katmanı (cihaz algılama, VAD)
    core        : AI motor katmanı (Whisper, Pyannote)

Modüller:
    config      : Sabitler, ortam değişkenleri, GPU/model ayarları
    pipeline    : Canlı kayıt döngüsü ve orkestrasyon
"""

from .pipeline import run

__all__ = ["run"]
