"""
audio — Ses Donanım Katmanı

Modüller:
    device : WASAPI loopback cihaz algılama
    vad    : WebRTC + Silero ikili katman ses algılama
"""

from .device import auto_detect_device
from .vad import VADEngine

__all__ = ["auto_detect_device", "VADEngine"]
