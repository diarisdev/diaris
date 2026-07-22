"""Ses ön-işleme yardımcıları (durumsuz).

AIWorker'ın embedding/diarization öncesi ses hazırlama adımları. Saf fonksiyonlar
— model/parametre argümanla geçilir, sınıf durumu gerektirmez. ai_worker.py bu
fonksiyonlara delege eder ve `load_silero_vad`'ı geri export eder.
"""

import logging

import numpy as np
import torch
import torchaudio

from ..config import LOCAL_MODELS_DIR
from .silero import load_silero_vad as _load_silero_vad

logger = logging.getLogger(__name__)


def load_silero_vad():
    """Load Silero VAD through one mockable boundary.

    Backend seçimi src/audio/silero.py'de: ONNX (hızlı, çevrimdışı) varsa o,
    yoksa torch.hub. utils[0] her iki durumda da get_speech_timestamps'tir.
    """
    return _load_silero_vad(models_dir=LOCAL_MODELS_DIR)


# Normalizasyon sabitleri
TARGET_SPEECH_RMS = 0.1   # hedef konuşma seviyesi
MAX_NORMALIZATION_GAIN = 10.0  # sessiz/gürültü chunk'ları 100x'e kadar yükseltmeyi önler


def _estimate_speech_rms(mono: np.ndarray, rate: int | None) -> float:
    """Konuşma seviyesini ~20 ms'lik blok RMS'lerinin 95. yüzdeliğiyle kestirir.

    Chunk'lar artık ham sessizlik/hangover da içerdiğinden global RMS konuşma
    seviyesini olduğundan düşük gösterir ve kazancı şişirir; yüksek yüzdelik
    blok RMS'i, VAD gerektirmeden "konuşmalı bölgelerin" seviyesine yakınsar.
    """
    block = max(1, int((rate or 48000) * 0.02))
    n_blocks = mono.shape[0] // block
    if n_blocks >= 3:
        blocks = mono[: n_blocks * block].reshape(n_blocks, block)
        block_rms = np.sqrt(np.mean(blocks ** 2, axis=1))
        return float(np.percentile(block_rms, 95))
    return float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0


def to_mono_float32(audio_np_int16, rate: int | None = None):
    """Çok kanallı int16 → mono float32, konuşma seviyesine göre normalize.

    * Kanal ORTALAMASI alınır (tek kanal seçmek stereo-pan'lı system-audio'da
      bir konuşmacıyı neredeyse tamamen kaybettirebiliyordu).
    * Kazanç konuşma-seviyesi kestirimi üzerinden hesaplanır ve
      MAX_NORMALIZATION_GAIN ile sınırlanır — salt gürültü içeren chunk'ların
      100x yükseltilip Whisper/pyannote'a "konuşma gibi" sunulmasını önler.
    """
    if audio_np_int16.ndim > 1 and audio_np_int16.shape[1] > 1:
        mono = audio_np_int16.astype(np.float32).mean(axis=1) / 32768.0
    else:
        mono = audio_np_int16.flatten().astype(np.float32) / 32768.0

    speech_rms = _estimate_speech_rms(mono, rate)
    if speech_rms > 0.001:
        gain = min(TARGET_SPEECH_RMS / speech_rms, MAX_NORMALIZATION_GAIN)
        mono = np.clip(mono * gain, -1.0, 1.0)
    return mono


# Kaynak hız başına önceden hesaplanmış resample kernel'i.
# torchaudio.functional.resample her çağrıda sinc kernel'ini YENİDEN kurar;
# canlı yolda bu, her partial'da (300-600 ms'de bir) ve her final chunk'ta
# gereksiz maliyetti. transforms.Resample aynı matematiği kernel'i bir kez
# kurup uygular — çıktı örnekleri birebir aynıdır.
_RESAMPLERS: dict = {}


def resample_to_16k(mono_float32, src_rate):
    """Pyannote'un beklediği 16kHz sample rate'e resample eder (kernel cache'li)."""
    waveform = torch.from_numpy(mono_float32).unsqueeze(0)
    if src_rate == 16000:
        return waveform, 16000
    resampler = _RESAMPLERS.get(src_rate)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(orig_freq=src_rate, new_freq=16000)
        _RESAMPLERS[src_rate] = resampler
    return resampler(waveform), 16000


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


def extract_speech_only(waveform_16k_1d, silero_vad, get_speech_timestamps=None):
    """
    Silero VAD ile sadece konuşma içeren bölümleri çıkarır.
    Sessizlik ve arka plan gürültüsünü atar.

    Args:
        waveform_16k_1d: 1D tensor (samples,) at 16kHz
        silero_vad: yüklü Silero VAD modeli (None ise ses olduğu gibi döner)
        get_speech_timestamps: Silero VAD'ın kendi get_speech_timestamps utility fonksiyonu.
                               Sağlanırsa hızlı ve optimize vectorized path kullanılır.

    Returns:
        torch.Tensor: sadece konuşma içeren ses (1D), veya orijinal
    """
    if silero_vad is None:
        return waveform_16k_1d

    try:
        if get_speech_timestamps is not None:
            # Hızlı, vectorized ve optimize Silero VAD path'i
            speech_dicts = get_speech_timestamps(waveform_16k_1d, silero_vad, threshold=0.5, sampling_rate=16000)
            speech_parts = [waveform_16k_1d[ts["start"]:ts["end"]] for ts in speech_dicts]
            if speech_parts:
                return torch.cat(speech_parts)
            return torch.tensor([], dtype=waveform_16k_1d.dtype, device=waveform_16k_1d.device)

        # Fallback: Yavaş ama güvenli manuel döngü (get_speech_timestamps yoksa)
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
