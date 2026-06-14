"""
AI Worker modülü.
Arka planda çalışan transkripsiyon + konuşmacı ayrıştırma iş parçacığı.
Whisper (transkripsiyon) ve Pyannote (diarization) modellerini yükler ve
kuyruktan gelen ses parçalarını işler.

Warm-up mekanizması:
- İlk N saniye boyunca embedding toplanır (konuşmacı etiketi atanmaz)
- Warm-up bitince toplanan embedding'ler kümelenir → başlangıç konuşmacıları oluşur
- Sonrasında yeni chunk'lar bu baseline'lara göre eşlenir

İyileştirmeler:
- RMS normalizasyon (peak yerine)
- Bandpass filter (200-3500 Hz) — embedding çıkarmadan önce
- Speech-only embedding (Silero VAD ile sessizlik temizleme)
- İki-aşamalı warm-up clustering (pairwise similarity + küçük küme filtreleme)
- Noisy embedding filtreleme (min süre, min enerji)
- Confidence-weighted baseline güncelleme
- Speaker label smoothing (< 1s segmentleri birleştir)
"""

import logging
import os
import numpy as np
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model, Inference

from ..config import (
    DEVICE, COMPUTE_TYPE, HF_TOKEN,
    WHISPER_PATH, DIARIZATION_CONFIG_PATH,
    LOCAL_MODELS_DIR,
    DEFAULT_RATE, DEFAULT_CHANNELS, WHISPER_LANGUAGE,
    DIARIZATION_WARMUP_MS,
)
from ..audio.utils import calculate_chunk_duration_ms
from ..audio.preprocessing import (
    to_mono_float32, resample_to_16k, apply_bandpass_filter,
    extract_speech_only, load_silero_vad,
)
from .diarization_config import prepare_runtime_config, get_short_path
from .diarization_utils import assign_words_to_speakers
# SpeakerTracker ayrı modüle taşındı; buradan geri export edilir (eski importlar +
# EvalSpeakerTracker subclass'ı aynen çalışsın diye).
from .speaker_tracker import SpeakerTracker
# Embedding çıkarma + sabitleri ayrı modüle taşındı; geri export edilir.
from .embedding_extractor import (
    extract_speaker_embeddings,
    MIN_SPEECH_DURATION_FOR_EMBEDDING,
    MIN_AUDIO_RMS_FOR_EMBEDDING,
    MAX_EMBED_AUDIO_SEC,
)

logger = logging.getLogger(__name__)

# (assign_words_to_speakers -> diarization_utils; DSP + load_silero_vad ->
#  ..audio.preprocessing; config plumbing -> diarization_config; SpeakerTracker ->
#  speaker_tracker; embedding çıkarma -> embedding_extractor. Hepsi buradan geri
#  export edilir, böylece eski import yolları + subclass'lar aynen çalışır.)


class AIWorker:
    """
    Transkripsiyon ve konuşmacı ayrıştırma motoru.
    
    Warm-up fazında embedding toplar, sonra canlı diarization yapar.
    """

    def __init__(self, rate=None, channels=None):
        self.rate = rate if rate is not None else DEFAULT_RATE
        self.channels = channels if channels is not None else DEFAULT_CHANNELS
        self.diarizer = None
        self.transcriber = None
        self.embedding_model = None
        self.silero_vad = None  # Speech-only embedding için
        self.vad_utils = None
        self._loaded = False
        self.speaker_tracker = SpeakerTracker()
        self._runtime_config_path = None  # Geçici config dosyası yolu

    def _prepare_runtime_config(self, config_path):
        """Diarization config'i runtime'da hazırlar (diarization_config'e delege)."""
        self._runtime_config_path = prepare_runtime_config(config_path)
        return self._runtime_config_path

    @staticmethod
    def _get_short_path(long_path):
        """Windows 8.3 kısa yol (diarization_config'e delege)."""
        return get_short_path(long_path)

    def load_models(self):
        """Modelleri yükler. İlk çağrıda bir kez çalışır."""
        if self._loaded:
            return True

        print(f"\n[AI Worker] {DEVICE.upper()} üzerinde başlatılıyor...")
        try:
            if not os.path.isdir(WHISPER_PATH):
                print(f"[AI Worker Error] Whisper modeli bulunamadı: {WHISPER_PATH}")
                print("   Modelleri indirmek için: python scripts/download_models.py")
                return False

            # Diarizer: Yerel config dosyasından yükle
            if os.path.exists(DIARIZATION_CONFIG_PATH):
                print(f"[AI Worker] Local diarization config: {DIARIZATION_CONFIG_PATH}")

                # Config'deki model yollarını runtime'da doğru mutlak yollarla güncelle.
                # Türkçe karakter (ü) ve boşluk içeren yollar pyannote'un
                # HuggingFace repo ID validator'ını kırdığı için config'i
                # ASCII-safe bir temp dizine yazıyoruz.
                runtime_config_path = self._prepare_runtime_config(DIARIZATION_CONFIG_PATH)
                self.diarizer = Pipeline.from_pretrained(
                    runtime_config_path,
                )
            else:
                if not HF_TOKEN:
                    print("[AI Worker Error] HF_TOKEN yok ve yerel diarization config bulunamadı.")
                    print(f"   Beklenen yerel config: {DIARIZATION_CONFIG_PATH}")
                    return False
                print("[AI Worker] Local config not found, loading from HuggingFace...")
                self.diarizer = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HF_TOKEN,
                )

            if DEVICE == "cuda":
                self.diarizer.to(torch.device("cuda"))

            # Embedding modeli: Cross-chunk konuşmacı tanıma için
            embedding_path = os.path.join(LOCAL_MODELS_DIR, "pyannote-embeddings")
            embedding_path = self._get_short_path(embedding_path)
            if not os.path.isdir(embedding_path):
                print(f"[AI Worker] Embedding model not found: {embedding_path}")
                print("   Modeli indirmek için: python scripts/download_models.py")
                self.embedding_model = None
            else:
                try:
                    emb_model = Model.from_pretrained(embedding_path)
                    self.embedding_model = Inference(
                        emb_model,
                        window="whole",
                        device=torch.device(DEVICE),
                    )
                    print("[AI Worker] Embedding model loaded (cross-chunk tracking enabled)")
                except Exception as emb_err:
                    print(f"[AI Worker] Embedding model failed: {emb_err}")
                    self.embedding_model = None

            # Silero VAD: Speech-only embedding extraction için
            try:
                self.silero_vad, self.vad_utils = load_silero_vad()
                print("[AI Worker] Silero VAD loaded (speech-only embedding enabled)")
            except Exception as vad_err:
                logger.warning("Silero VAD could not be loaded: %s", vad_err)
                self.silero_vad = None
                self.vad_utils = None

            self.transcriber = WhisperModel(
                WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE
            )
            self._loaded = True
            warmup_sec = DIARIZATION_WARMUP_MS / 1000
            print(f"[AI Worker] Models loaded. Warm-up: {warmup_sec:.0f}s\n")
            return True
        except Exception as e:
            print(f"[AI Worker Error] Models failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _to_mono_float32(self, audio_np_int16):
        """Stereo int16 → mono float32, RMS normalized (preprocessing'e delege)."""
        return to_mono_float32(audio_np_int16)

    def _resample_for_pyannote(self, mono_float32):
        """16kHz'e resample (preprocessing'e delege)."""
        return resample_to_16k(mono_float32, self.rate)

    def _apply_bandpass_filter(self, waveform_16k):
        """200-3500 Hz bandpass (preprocessing'e delege)."""
        return apply_bandpass_filter(waveform_16k)

    def _extract_speech_only(self, waveform_16k_1d):
        """Silero VAD ile speech-only (preprocessing'e delege)."""
        get_speech_timestamps = self.vad_utils[0] if (self.vad_utils and len(self.vad_utils) > 0) else None
        return extract_speech_only(waveform_16k_1d, self.silero_vad, get_speech_timestamps)

    def _extract_speaker_embeddings(self, waveform_16k, turns):
        """Konuşmacı embedding çıkarma (embedding_extractor'a delege)."""
        get_speech_timestamps = self.vad_utils[0] if (self.vad_utils and len(self.vad_utils) > 0) else None
        return extract_speaker_embeddings(
            self.embedding_model, self.silero_vad, get_speech_timestamps, waveform_16k, turns
        )

    def _get_chunk_duration_ms(self, chunk_bytes):
        """Chunk süresini ms cinsinden hesaplar."""
        return calculate_chunk_duration_ms(chunk_bytes, self.rate, self.channels)

    def _smooth_speaker_labels(self, results):
        """
        Speaker label smoothing: < 1s segmentleri komşu konuşmacıyla birleştirir.
        Hızlı speaker switching'i önler.
        """
        if len(results) < 3:
            return results

        smoothed = list(results)

        for i in range(1, len(smoothed) - 1):
            seg = smoothed[i]
            duration = seg["end"] - seg["start"]

            # Kısa segment (< 1s) ve komşuları aynı konuşmacıysa → birleştir
            if duration < 1.0:
                prev_speaker = smoothed[i - 1]["speaker"]
                next_speaker = smoothed[i + 1]["speaker"]

                if prev_speaker == next_speaker and seg["speaker"] != prev_speaker:
                    smoothed[i] = dict(seg)
                    smoothed[i]["speaker"] = prev_speaker

        return smoothed

    def process_chunk(self, chunk_bytes, is_final=True, language=None):
        """
        Bir ses parçasını işler: transkripsiyon yapar.

        Args:
            chunk_bytes: Ham ses verisi (bytes, int16)
            is_final: Eğer False ise sadece hızlı transkripsiyon yapılır (diarization pas geçilir)
            language: Transkripsiyon için dil kodu (örn. 'en', 'tr'). Belirtilmezse varsayılan WHISPER_LANGUAGE kullanılır.

        Returns:
            dict: Sonuçları ve diarization için gerekli waveform bilgilerini içeren dict.
        """
        if not self._loaded:
            return None

        if len(chunk_bytes) == 0:
            return None

        audio_np_int16 = np.frombuffer(chunk_bytes, dtype=np.int16).reshape(-1, self.channels)
        if audio_np_int16.size == 0:
            return None

        chunk_duration_ms = self._get_chunk_duration_ms(chunk_bytes)

        try:
            # --- Pyannote/Whisper için ses hazırla ---
            mono_float32 = self._to_mono_float32(audio_np_int16)
            waveform_16k, sample_rate = self._resample_for_pyannote(mono_float32)
            audio_np_16k = waveform_16k.squeeze(0).numpy()

            if not is_final:
                # Hızlı Kısmi Transkripsiyon (No Diarization, No Disk write)
                segments, _ = self.transcriber.transcribe(
                    audio_np_16k,
                    beam_size=1,
                    condition_on_previous_text=False,
                    language=language or WHISPER_LANGUAGE,
                )
                segments = list(segments)
                if len(segments) == 0:
                    return None

                results = []
                for segment in segments:
                    results.append({
                        "speaker": "Kısmi",
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    })
                return {"results": results}

            # EĞER FİNAL İSE:
            # Sadece Whisper ile yüksek kaliteli transkripsiyon yap
            segments, _ = self.transcriber.transcribe(
                audio_np_16k,
                beam_size=3,
                word_timestamps=True,
                language=language or WHISPER_LANGUAGE,
            )
            segments = list(segments)

            if len(segments) == 0:
                return None

            results = []
            for segment in segments:
                # Kelime-seviyesi zaman damgalarını sakla — diarization aşamasında
                # konuşmacı sınırından bölme için kullanılır.
                words = []
                for w in (segment.words or []):
                    words.append({
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                    })
                results.append({
                    "speaker": "Çözümleniyor...",
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": words,
                })

            return {
                "results": results,
                "waveform_16k": waveform_16k,
                "sample_rate": sample_rate,
                "chunk_duration_ms": chunk_duration_ms
            }

        except Exception as e:
            print(f"\n[Transcription Error]: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_diarization(self, waveform_16k, sample_rate, chunk_duration_ms, transcribed_segments):
        """
        Finalleşmiş bir ses parçası üzerinde pyannote diarization, konuşmacı eşleme
        ve etiket yumuşatma işlemlerini çalıştırır. Arka plan thread'inde çalışacak şekilde tasarlanmıştır.
        """
        try:
            # --- Diarization ---
            pyannote_input = {"waveform": waveform_16k, "sample_rate": sample_rate}
            pipeline_output = self.diarizer(pyannote_input)
            if hasattr(pipeline_output, "speaker_diarization"):
                diarization = pipeline_output.speaker_diarization
            else:
                diarization = pipeline_output

            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            # --- Embedding + kalite (süre) çıkar --
            if turns:
                embeddings_dict, quality_dict = self._extract_speaker_embeddings(waveform_16k, turns)
            else:
                embeddings_dict, quality_dict = {}, {}

            # --- Warm-up / Aktif faz ---
            if self.speaker_tracker.is_warming_up:
                # Warm-up: embedding topla, konuşmacı etiketi atama
                for emb in embeddings_dict.values():
                    warmup_done = self.speaker_tracker.add_warmup_embedding(emb, chunk_duration_ms)
                    if warmup_done:
                        break

                remaining_ms = max(0, self.speaker_tracker.warmup_ms - self.speaker_tracker._warmup_audio_ms)
                warmup_label = f"[Calibrating... {remaining_ms / 1000:.0f}s]"
                speaker_mapping = {t["speaker"]: warmup_label for t in turns}
            else:
                # Aktif faz: embedding-based konuşmacı eşleme
                if turns and embeddings_dict:
                    speaker_mapping = self.speaker_tracker.map_speakers(embeddings_dict, quality_dict)
                    for t in turns:
                        if t["speaker"] not in speaker_mapping:
                            speaker_mapping[t["speaker"]] = "Unknown"
                elif turns:
                    local_labels = list(set(t["speaker"] for t in turns))
                    speaker_mapping = self.speaker_tracker.map_speakers_fallback(local_labels)
                else:
                    speaker_mapping = {}

            for turn in turns:
                turn["speaker"] = speaker_mapping.get(turn["speaker"], turn["speaker"])

            # Kelime-seviyesi atama + konuşmacı sınırında bölme.
            # (Whisper segmenti A+B'yi birleştirmişse burada ayrılır.)
            results = assign_words_to_speakers(transcribed_segments, turns)

            # Speaker label smoothing
            results = self._smooth_speaker_labels(results)
            return results

        except Exception as e:
            print(f"\n[Diarization Error]: {e}")
            import traceback
            traceback.print_exc()
            return transcribed_segments
