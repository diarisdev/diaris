"""
src - Audio-process Modüler Paket

Modüller:
    config      : Sabitler, ortam değişkenleri, GPU/model ayarları
    device      : Ses cihazı bulma
    vad         : WebRTC + Silero ikili katman ses algılama
    ai_worker   : Whisper transkripsiyon + Pyannote konuşmacı ayrıştırma
    pipeline    : Canlı kayıt döngüsü ve orkestrasyon
"""

from .pipeline import run

__all__ = ["run"]
