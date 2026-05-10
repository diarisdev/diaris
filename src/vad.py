"""
Voice Activity Detection (VAD) modülü.
WebRTC (hızlı ön filtre) + Silero (doğru sinir ağı) ikili katmanlı ses algılama.
"""

import numpy as np
import torch
import webrtcvad

from .config import VAD_AGGRESSIVENESS, SILERO_THRESHOLD


class VADEngine:
    """
    İkili katmanlı ses algılama motoru.
    
    Katman 1 - WebRTC: Çok hızlı, kaba filtre (CPU-only, C tabanlı)
    Katman 2 - Silero: Doğru sinir ağı doğrulaması (PyTorch)
    """

    def __init__(self, aggressiveness=None, threshold=None):
        """
        Args:
            aggressiveness: WebRTC agresiflik seviyesi (0-3). None ise config'den alınır.
            threshold: Silero güven eşiği. None ise config'den alınır.
        """
        self.aggressiveness = aggressiveness if aggressiveness is not None else VAD_AGGRESSIVENESS
        self.threshold = threshold if threshold is not None else SILERO_THRESHOLD

        self.webrtc_vad = webrtcvad.Vad(self.aggressiveness)
        self.silero_model, _ = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', trust_repo=True
        )

    def check_speech(self, data, rate, channels):
        """
        Ses verisinde konuşma olup olmadığını kontrol eder.

        Args:
            data: Ham ses verisi (bytes, int16 formatında)
            rate: Örnekleme hızı (Hz)
            channels: Kanal sayısı

        Returns:
            tuple[bool, float]: (konuşma_var_mı, güven_skoru)
        """
        audio_np = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
        mono_audio = audio_np[:, 0].copy()

        # WebRTC tam olarak 30ms frame bekler
        expected_samples = int(rate * 30 / 1000)
        if len(mono_audio) != expected_samples:
            return False, 0.0

        try:
            # --- KATMAN 1: WebRTC (hızlı ön filtre) ---
            if self.webrtc_vad.is_speech(mono_audio.tobytes(), rate):
                # --- KATMAN 2: Silero (sinir ağı doğrulaması) ---
                if rate == 48000:
                    silero_audio = mono_audio[::3]
                    silero_rate = 16000
                else:
                    silero_audio = mono_audio
                    silero_rate = rate

                # Silero minimum 512 örnek gerektirir
                if len(silero_audio) < 512:
                    pad_length = 512 - len(silero_audio)
                    silero_audio = np.pad(silero_audio, (0, pad_length), 'constant')

                audio_float = silero_audio.astype(np.float32) / 32768.0

                with torch.no_grad():
                    confidence = self.silero_model(
                        torch.from_numpy(audio_float), silero_rate
                    ).item()

                return confidence > self.threshold, confidence
        except Exception:
            pass

        return False, 0.0
