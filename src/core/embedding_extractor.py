"""Konuşmacı embedding çıkarma.

Pyannote diarization turn'lerinden konuşmacı-başına ses toplar, kalite kapılarından
(min süre/enerji), speech-only ve bandpass adımlarından geçirip embedding üretir.
ai_worker.py'den ayrıldı (saf taşıma; davranış birebir). ai_worker bunu ve
sabitlerini geri export eder.
"""

import numpy as np
import torch

from ..audio.preprocessing import apply_bandpass_filter, extract_speech_only

# Minimum embedding requirements
MIN_SPEECH_DURATION_FOR_EMBEDDING = 1.5  # saniye — embedding için min konuşma süresi
MIN_AUDIO_RMS_FOR_EMBEDDING = 0.01       # min RMS enerji — sessizliği filtrele
# Perf: embedding için konuşmacı başına kullanılacak max ses süresi. Daha uzun
# ses embedding kalitesini kayda değer artırmaz ama Silero VAD'in Python döngüsü
# ve embedding inference maliyetini (özellikle uzun chunk'larda) şişirir.
MAX_EMBED_AUDIO_SEC = 6.0


def extract_speaker_embeddings(embedding_model, silero_vad, waveform_16k, turns):
    """
    Her konuşmacı için ses bölümlerini ayırıp embedding çıkarır.

    İyileştirmeler:
    - Min süre kontrolü (< 1.5s konuşma → atla)
    - Min enerji kontrolü (sessizlik → atla)
    - Bandpass filter (200-3500 Hz)
    - Speech-only extraction (Silero VAD)

    Args:
        embedding_model: pyannote Inference modeli (None ise boş döner)
        silero_vad: speech-only için Silero VAD (None olabilir)
        waveform_16k: (1, samples) tensor @16kHz
        turns: [{"start","end","speaker"}, ...]

    Returns:
        tuple: (embeddings_dict, quality_dict)
            embeddings_dict: {local_speaker_label: embedding_tensor}
            quality_dict:    {local_speaker_label: temiz_konuşma_süresi_sn}
    """
    if embedding_model is None:
        return {}, {}

    embeddings_dict = {}
    quality_dict = {}
    unique_speakers = set(t["speaker"] for t in turns)

    for spk in unique_speakers:
        spk_turns = [t for t in turns if t["speaker"] == spk]
        spk_audio_parts = []

        for t in spk_turns:
            start_sample = int(t["start"] * 16000)
            end_sample = min(int(t["end"] * 16000), waveform_16k.shape[1])
            if start_sample < end_sample:
                spk_audio_parts.append(waveform_16k[0, start_sample:end_sample])

        if not spk_audio_parts:
            continue

        spk_audio_raw = torch.cat(spk_audio_parts)

        # --- Kalite kontrolleri ---
        # 1. Min süre kontrolü
        duration_sec = spk_audio_raw.shape[0] / 16000
        if duration_sec < MIN_SPEECH_DURATION_FOR_EMBEDDING:
            continue

        # 2. Min enerji kontrolü
        rms_energy = torch.sqrt(torch.mean(spk_audio_raw ** 2)).item()
        if rms_energy < MIN_AUDIO_RMS_FOR_EMBEDDING:
            continue

        # Perf cap: pahalı adımlardan (Silero VAD döngüsü + embedding) önce
        # konuşmacı başına sesi MAX_EMBED_AUDIO_SEC ile sınırla.
        max_samples = int(MAX_EMBED_AUDIO_SEC * 16000)
        if spk_audio_raw.shape[0] > max_samples:
            spk_audio_raw = spk_audio_raw[:max_samples]

        # 3. Speech-only extraction (sessizliği temizle)
        spk_audio_clean = extract_speech_only(spk_audio_raw, silero_vad)

        # Temizlenmiş ses hâlâ yeterli mi?
        if spk_audio_clean.shape[0] < int(MIN_SPEECH_DURATION_FOR_EMBEDDING * 16000):
            continue

        clean_duration_sec = spk_audio_clean.shape[0] / 16000

        # 4. Bandpass filter (konuşma frekansları)
        spk_waveform = apply_bandpass_filter(spk_audio_clean.unsqueeze(0))

        try:
            emb_output = embedding_model({
                "waveform": spk_waveform,
                "sample_rate": 16000,
            })
            if isinstance(emb_output, np.ndarray):
                emb_tensor = torch.from_numpy(emb_output).float()
            elif isinstance(emb_output, torch.Tensor):
                emb_tensor = emb_output.float()
            else:
                emb_tensor = torch.tensor(emb_output).float()

            emb_tensor = emb_tensor.squeeze().cpu()

            # 5. L2 Normalization (Prevents scale issues in clustering/similarity)
            emb_norm = torch.norm(emb_tensor)
            if emb_norm > 0:
                emb_tensor = emb_tensor / emb_norm

            embeddings_dict[spk] = emb_tensor
            quality_dict[spk] = clean_duration_sec

        except Exception as e:
            print(f"  [Embedding] {spk}: {e}")

    return embeddings_dict, quality_dict
