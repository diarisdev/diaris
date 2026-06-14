"""Ses ön-işleme yardımcıları (durumsuz).

AIWorker'ın embedding/diarization öncesi ses hazırlama adımları. Saf fonksiyonlar
— model/parametre argümanla geçilir, sınıf durumu gerektirmez. ai_worker.py bu
fonksiyonlara delege eder ve `load_silero_vad`'ı geri export eder.
"""

import logging

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_silero_vad():
    """Load Silero VAD through one mockable boundary."""
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model, utils


def to_mono_float32(audio_np_int16):
    """Stereo int16 → mono float32, RMS normalized."""
    if audio_np_int16.ndim > 1 and audio_np_int16.shape[1] > 1:
        mono = audio_np_int16[:, 0].astype(np.float32) / 32768.0
    else:
        mono = audio_np_int16.flatten().astype(np.float32) / 32768.0

    # RMS normalizasyon (peak yerine — daha kararlı)
    rms = np.sqrt(np.mean(mono ** 2))
    if rms > 0.001:
        target_rms = 0.1
        mono = mono * (target_rms / rms)
        # Clipping önle
        mono = np.clip(mono, -1.0, 1.0)
    return mono


def resample_to_16k(mono_float32, src_rate):
    """Pyannote'un beklediği 16kHz sample rate'e resample eder."""
    waveform = torch.from_numpy(mono_float32).unsqueeze(0)
    if src_rate == 16000:
        return waveform, 16000
    resampled = torchaudio.functional.resample(
        waveform, orig_freq=src_rate, new_freq=16000
    )
    return resampled, 16000


def apply_bandpass_filter(waveform_16k):
    """
    200-3500 Hz bandpass filter — embedding çıkarmadan önce.
    İnsan konuşma frekanslarını korur, gürültüyü atar.
    """
    try:
        # Highpass 200 Hz
        filtered = torchaudio.functional.highpass_biquad(
            waveform_16k, sample_rate=16000, cutoff_freq=200.0
        )
        # Lowpass 3500 Hz
        filtered = torchaudio.functional.lowpass_biquad(
            filtered, sample_rate=16000, cutoff_freq=3500.0
        )
        return filtered
    except Exception as exc:
        logger.debug("Bandpass filter failed, using original waveform: %s", exc)
        return waveform_16k


def extract_speech_only(waveform_16k_1d, silero_vad):
    """
    Silero VAD ile sadece konuşma içeren bölümleri çıkarır.
    Sessizlik ve arka plan gürültüsünü atar.

    Args:
        waveform_16k_1d: 1D tensor (samples,) at 16kHz
        silero_vad: yüklü Silero VAD modeli (None ise ses olduğu gibi döner)

    Returns:
        torch.Tensor: sadece konuşma içeren ses (1D), veya orijinal
    """
    if silero_vad is None:
        return waveform_16k_1d

    try:
        # Silero VAD ile speech segmentlerini bul
        speech_timestamps = []
        window_size = 512  # 32ms at 16kHz
        total_samples = waveform_16k_1d.shape[0]

        # Reset VAD state
        silero_vad.reset_states()

        for start in range(0, total_samples - window_size, window_size):
            chunk = waveform_16k_1d[start:start + window_size]
            with torch.no_grad():
                prob = silero_vad(chunk, 16000).item()
            if prob > 0.5:
                speech_timestamps.append((start, start + window_size))

        if not speech_timestamps:
            return waveform_16k_1d

        # Ardışık segmentleri birleştir
        merged = [speech_timestamps[0]]
        for start, end in speech_timestamps[1:]:
            if start <= merged[-1][1] + window_size:  # gap < 32ms → merge
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        # Konuşma bölümlerini birleştir
        speech_parts = []
        for start, end in merged:
            speech_parts.append(waveform_16k_1d[start:end])

        if speech_parts:
            return torch.cat(speech_parts)
        return waveform_16k_1d

    except Exception as exc:
        logger.debug("Speech-only extraction failed, using original waveform: %s", exc)
        return waveform_16k_1d
