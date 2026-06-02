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
import tempfile
import numpy as np
import torch
import torchaudio
import soundfile as sf
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model, Inference

from ..config import (
    DEVICE, COMPUTE_TYPE, HF_TOKEN,
    WHISPER_PATH, DIARIZATION_CONFIG_PATH,
    LOCAL_MODELS_DIR,
    DEFAULT_RATE, DEFAULT_CHANNELS, WHISPER_LANGUAGE,
    DIARIZATION_EMBEDDING_THRESHOLD, DIARIZATION_WARMUP_MS,
)
from ..audio.utils import calculate_chunk_duration_ms

logger = logging.getLogger(__name__)

# Minimum embedding requirements
MIN_SPEECH_DURATION_FOR_EMBEDDING = 1.5  # saniye — embedding için min konuşma süresi
MIN_AUDIO_RMS_FOR_EMBEDDING = 0.01       # min RMS enerji — sessizliği filtrele


def load_silero_vad():
    """Load Silero VAD through one mockable boundary."""
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model, utils


class SpeakerTracker:
    """
    Embedding-based konuşmacı takip sistemi (warm-up destekli).
    
    Faz 1 (Warm-up): Kaliteli embedding'ler toplanır, kümelenir.
    Faz 2 (Aktif):   Yeni embedding'ler baseline'larla karşılaştırılır.
                      Güncelleme confidence-weighted yapılır.
    """

    def __init__(self, threshold=None, warmup_ms=None):
        self.threshold = threshold if threshold is not None else DIARIZATION_EMBEDDING_THRESHOLD
        self.warmup_ms = warmup_ms if warmup_ms is not None else DIARIZATION_WARMUP_MS

        # Bilinen konuşmacılar (warm-up sonrası dolu olur)
        self.known_speakers = {}  # {global_label: embedding_tensor}
        self._next_id = 0

        # Warm-up state
        self._warmup_buffer = []  # list of embedding tensors
        self._warmup_audio_ms = 0  # toplam işlenen ses süresi
        self._warmup_complete = False

    def reset(self):
        """Tracker durumunu sıfırlayarak yeni bir dosya için hazır hale getirir."""
        self.known_speakers = {}
        self._next_id = 0
        self._warmup_buffer = []
        self._warmup_audio_ms = 0
        self._warmup_complete = False

    def _next_label(self):
        label = f"SPEAKER_{self._next_id:02d}"
        self._next_id += 1
        return label

    @property
    def is_warming_up(self):
        return not self._warmup_complete

    def add_warmup_embedding(self, embedding, chunk_duration_ms):
        """
        Warm-up fazında embedding toplar.
        Yeterli ses birikince warm-up'ı sonlandırır.

        Returns:
            bool: True ise warm-up bitti (baseline hazır)
        """
        self._warmup_buffer.append(embedding.cpu())
        self._warmup_audio_ms += chunk_duration_ms

        if self._warmup_audio_ms >= self.warmup_ms:
            self._finalize_warmup()
            return True
        return False

    def _finalize_warmup(self):
        """
        İki-aşamalı warm-up clustering:
        1. Pairwise similarity matrix ile agglomerative clustering
        2. Küçük kümeleri (< 2 embedding) filtrele (gürültü)
        """
        if not self._warmup_buffer:
            self._warmup_complete = True
            return

        n = len(self._warmup_buffer)
        print(f"\n🔬 [Warm-up] Clustering {n} embeddings...")

        if n == 1:
            # Tek embedding varsa direkt konuşmacı oluştur
            label = self._next_label()
            self.known_speakers[label] = self._warmup_buffer[0]
            self._warmup_complete = True
            self._warmup_buffer = []
            print(f"✅ [Warm-up Complete] 1 speaker detected: {label}")
            print(f"   ({self._warmup_audio_ms / 1000:.1f}s audio)\n")
            return

        # Pairwise similarity matrix
        emb_stack = torch.stack(self._warmup_buffer)  # (n, dim)
        sim_matrix = torch.nn.functional.cosine_similarity(
            emb_stack.unsqueeze(0), emb_stack.unsqueeze(1), dim=2
        )  # (n, n)

        # Agglomerative clustering — her embedding kendi kümesi olarak başlar
        cluster_ids = list(range(n))
        clusters = {i: [i] for i in range(n)}

        # Merge: en yüksek similarity'den başla
        while True:
            best_i, best_j, best_sim = -1, -1, -1.0

            active_clusters = list(clusters.keys())
            for ci_idx in range(len(active_clusters)):
                for cj_idx in range(ci_idx + 1, len(active_clusters)):
                    ci = active_clusters[ci_idx]
                    cj = active_clusters[cj_idx]

                    # Average linkage: kümelerdeki tüm çiftlerin ortalama similarity'si
                    total_sim = 0.0
                    count = 0
                    for mi in clusters[ci]:
                        for mj in clusters[cj]:
                            total_sim += sim_matrix[mi, mj].item()
                            count += 1
                    avg_sim = total_sim / count if count > 0 else 0.0

                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_i = ci
                        best_j = cj

            # Threshold'un altındaysa dur
            if best_sim < self.threshold or best_i < 0:
                break

            # Merge clusters
            clusters[best_i].extend(clusters[best_j])
            del clusters[best_j]

        # Küçük kümeleri filtrele (gürültü olma ihtimali yüksek)
        min_cluster_size = 2 if n >= 6 else 1
        valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

        # Eğer filtreleme sonrası hiçbir küme kalmadıysa, en büyük kümeyi al
        if not valid_clusters:
            largest = max(clusters.items(), key=lambda x: len(x[1]))
            valid_clusters = {largest[0]: largest[1]}

        # Her kümeden konuşmacı oluştur
        for cluster_id, member_indices in valid_clusters.items():
            member_embs = [self._warmup_buffer[i] for i in member_indices]
            centroid = torch.stack(member_embs).mean(dim=0)
            label = self._next_label()
            self.known_speakers[label] = centroid

        self._warmup_complete = True

        # Filtrelenen embedding sayısı
        total_used = sum(len(v) for v in valid_clusters.values())
        filtered_count = n - total_used

        speaker_list = ", ".join(self.known_speakers.keys())
        print(f"✅ [Warm-up Complete] {len(self.known_speakers)} speaker(s) detected: {speaker_list}")
        if filtered_count > 0:
            print(f"   (filtered {filtered_count} noisy embedding(s))")
        print(f"   ({self._warmup_audio_ms / 1000:.1f}s audio processed)\n")

        self._warmup_buffer = []

    def map_speakers(self, embeddings_dict):
        """
        Embedding'lere göre konuşmacıları eşler (warm-up sonrası).
        Confidence-weighted baseline güncelleme yapar.

        Args:
            embeddings_dict: {local_label: embedding_tensor}

        Returns:
            dict: {local_label: global_label}
        """
        mapping = {}

        for local_label, emb in embeddings_dict.items():
            emb = emb.cpu()

            # Bilinen konuşmacılarla karşılaştır
            best_match = None
            best_score = -1.0

            for global_label, known_emb in self.known_speakers.items():
                score = torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), known_emb.unsqueeze(0)
                ).item()
                if score > best_score:
                    best_score = score
                    best_match = global_label

            margin = 0.15  # Uncertainty Zone margin

            if best_match and best_score >= self.threshold:
                mapping[local_label] = best_match

                # Confidence-weighted güncelleme
                if best_score > 0.85:
                    alpha = 0.6
                elif best_score > 0.70:
                    alpha = 0.8
                else:
                    alpha = 0.95

                self.known_speakers[best_match] = (
                    alpha * self.known_speakers[best_match] + (1 - alpha) * emb
                )
                
                # Re-normalize baseline to prevent magnitude decay
                norm = torch.norm(self.known_speakers[best_match])
                if norm > 0:
                    self.known_speakers[best_match] /= norm

            elif best_match and best_score >= (self.threshold - margin):
                # Uncertainty Zone: Belirsiz ses. Yeni kişi uydurma, en yakın kişiye ata.
                # Ancak baseline'ı kirletmemek için GÜNCELLEME YAPMA.
                mapping[local_label] = best_match
                print(f"  ⚠️ Uncertain match: mapped to {best_match} (score: {best_score:.3f} < {self.threshold})")

            else:
                # Yeni konuşmacı tespit edildi
                new_label = self._next_label()
                mapping[local_label] = new_label
                self.known_speakers[new_label] = emb.clone()
                if best_match:
                    print(f"  🆕 New speaker: {new_label} (closest: {best_match}, score: {best_score:.3f})")
                else:
                    print(f"  🆕 New speaker: {new_label}")

        return mapping

    def map_speakers_fallback(self, local_labels):
        """Embedding yoksa fallback — her label'a yeni isim atar."""
        mapping = {}
        for label in local_labels:
            mapping[label] = self._next_label()
        return mapping


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
        self._loaded = False
        self.speaker_tracker = SpeakerTracker()

    def load_models(self):
        """Modelleri yükler. İlk çağrıda bir kez çalışır."""
        if self._loaded:
            return True

        print(f"\n🧠 [AI Worker] {DEVICE.upper()} üzerinde başlatılıyor...")
        try:
            if not os.path.isdir(WHISPER_PATH):
                print(f"❌ [AI Worker Error] Whisper modeli bulunamadı: {WHISPER_PATH}")
                print("   Modelleri indirmek için: python scripts/download_models.py")
                return False

            # Diarizer: Yerel config dosyasından yükle
            if os.path.exists(DIARIZATION_CONFIG_PATH):
                print(f"📂 [AI Worker] Local diarization config: {DIARIZATION_CONFIG_PATH}")
                self.diarizer = Pipeline.from_pretrained(
                    DIARIZATION_CONFIG_PATH,
                )
            else:
                if not HF_TOKEN:
                    print("❌ [AI Worker Error] HF_TOKEN yok ve yerel diarization config bulunamadı.")
                    print(f"   Beklenen yerel config: {DIARIZATION_CONFIG_PATH}")
                    return False
                print("⚠️ [AI Worker] Local config not found, loading from HuggingFace...")
                self.diarizer = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HF_TOKEN,
                )

            if DEVICE == "cuda":
                self.diarizer.to(torch.device("cuda"))

            # Embedding modeli: Cross-chunk konuşmacı tanıma için
            embedding_path = os.path.join(LOCAL_MODELS_DIR, "pyannote-embeddings")
            try:
                emb_model = Model.from_pretrained(embedding_path)
                self.embedding_model = Inference(
                    emb_model,
                    window="whole",
                    device=torch.device(DEVICE),
                )
                print("✅ [AI Worker] Embedding model loaded (cross-chunk tracking enabled)")
            except Exception as emb_err:
                print(f"⚠️ [AI Worker] Embedding model failed: {emb_err}")
                self.embedding_model = None

            # Silero VAD: Speech-only embedding extraction için
            try:
                self.silero_vad, _ = load_silero_vad()
                print("✅ [AI Worker] Silero VAD loaded (speech-only embedding enabled)")
            except Exception as vad_err:
                logger.warning("Silero VAD could not be loaded: %s", vad_err)
                self.silero_vad = None

            self.transcriber = WhisperModel(
                WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE
            )
            self._loaded = True
            warmup_sec = DIARIZATION_WARMUP_MS / 1000
            print(f"✅ [AI Worker] Models loaded. Warm-up: {warmup_sec:.0f}s\n")
            return True
        except Exception as e:
            print(f"❌ [AI Worker Error] Models failed to load: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _to_mono_float32(self, audio_np_int16):
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

    def _resample_for_pyannote(self, mono_float32):
        """Pyannote'un beklediği 16kHz sample rate'e resample eder."""
        waveform = torch.from_numpy(mono_float32).unsqueeze(0)
        if self.rate == 16000:
            return waveform, 16000
        resampled = torchaudio.functional.resample(
            waveform, orig_freq=self.rate, new_freq=16000
        )
        return resampled, 16000

    def _apply_bandpass_filter(self, waveform_16k):
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

    def _extract_speech_only(self, waveform_16k_1d):
        """
        Silero VAD ile sadece konuşma içeren bölümleri çıkarır.
        Sessizlik ve arka plan gürültüsünü atar.

        Args:
            waveform_16k_1d: 1D tensor (samples,) at 16kHz

        Returns:
            torch.Tensor: sadece konuşma içeren ses (1D), veya orijinal
        """
        if self.silero_vad is None:
            return waveform_16k_1d

        try:
            # Silero VAD ile speech segmentlerini bul
            speech_timestamps = []
            window_size = 512  # 32ms at 16kHz
            total_samples = waveform_16k_1d.shape[0]

            # Reset VAD state
            self.silero_vad.reset_states()

            for start in range(0, total_samples - window_size, window_size):
                chunk = waveform_16k_1d[start:start + window_size]
                with torch.no_grad():
                    prob = self.silero_vad(chunk, 16000).item()
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

    def _extract_speaker_embeddings(self, waveform_16k, turns):
        """
        Her konuşmacı için ses bölümlerini ayırıp embedding çıkarır.
        
        İyileştirmeler:
        - Min süre kontrolü (< 1.5s konuşma → atla)
        - Min enerji kontrolü (sessizlik → atla)
        - Bandpass filter (200-3500 Hz)
        - Speech-only extraction (Silero VAD)

        Returns:
            dict: {local_speaker_label: embedding_tensor}
        """
        if self.embedding_model is None:
            return {}

        embeddings_dict = {}
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

            # 3. Speech-only extraction (sessizliği temizle)
            spk_audio_clean = self._extract_speech_only(spk_audio_raw)

            # Temizlenmiş ses hâlâ yeterli mi?
            if spk_audio_clean.shape[0] < int(MIN_SPEECH_DURATION_FOR_EMBEDDING * 16000):
                continue

            # 4. Bandpass filter (konuşma frekansları)
            spk_waveform = self._apply_bandpass_filter(spk_audio_clean.unsqueeze(0))

            try:
                emb_output = self.embedding_model({
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

            except Exception as e:
                print(f"  ⚠️ [Embedding] {spk}: {e}")

        return embeddings_dict

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

    def process_chunk(self, chunk_bytes):
        """
        Bir ses parçasını işler: transkripsiyon + konuşmacı ayrıştırma.

        Warm-up fazında:  Transkripsiyon yapar, embedding toplar, konuşmacı etiket atanmaz.
        Aktif fazda:      Transkripsiyon + diarization + embedding eşleme + smoothing.

        Args:
            chunk_bytes: Ham ses verisi (bytes, int16)

        Returns:
            list[dict] veya None: Her segment için {speaker, start, end, text}.
        """
        if not self._loaded:
            return None

        if len(chunk_bytes) == 0:
            return None

        audio_np_int16 = np.frombuffer(chunk_bytes, dtype=np.int16).reshape(-1, self.channels)
        if audio_np_int16.size == 0:
            return None

        chunk_duration_ms = self._get_chunk_duration_ms(chunk_bytes)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_np_int16, self.rate)
                tmp_path = tmp_file.name

            # --- Pyannote için ses hazırla ---
            mono_float32 = self._to_mono_float32(audio_np_int16)
            waveform_16k, sample_rate = self._resample_for_pyannote(mono_float32)
            pyannote_input = {"waveform": waveform_16k, "sample_rate": sample_rate}

            # --- Diarization ---
            pipeline_output = self.diarizer(pyannote_input)
            if hasattr(pipeline_output, "speaker_diarization"):
                diarization = pipeline_output.speaker_diarization
            else:
                diarization = pipeline_output

            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            # --- Embedding çıkar --
            embeddings_dict = self._extract_speaker_embeddings(waveform_16k, turns) if turns else {}

            # --- Warm-up / Aktif faz ---
            if self.speaker_tracker.is_warming_up:
                # Warm-up: embedding topla, konuşmacı etiketi atama
                for emb in embeddings_dict.values():
                    warmup_done = self.speaker_tracker.add_warmup_embedding(emb, chunk_duration_ms)
                    if warmup_done:
                        break

                remaining_ms = max(0, self.speaker_tracker.warmup_ms - self.speaker_tracker._warmup_audio_ms)
                warmup_label = f"[Calibrating... {remaining_ms / 1000:.0f}s]"

                # Transkripsiyon yap ama konuşmacı olarak warm-up durumu göster
                segments, _ = self.transcriber.transcribe(
                    tmp_path, word_timestamps=True, language=WHISPER_LANGUAGE
                )
                segments = list(segments)

                if len(segments) == 0:
                    return None

                results = []
                for segment in segments:
                    results.append({
                        "speaker": warmup_label,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    })
                return results

            else:
                # Aktif faz: embedding-based konuşmacı eşleme
                if turns and embeddings_dict:
                    speaker_mapping = self.speaker_tracker.map_speakers(embeddings_dict)
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

                # Transkripsiyon
                segments, _ = self.transcriber.transcribe(
                    tmp_path, word_timestamps=True, language=WHISPER_LANGUAGE
                )
                segments = list(segments)

                if len(segments) == 0:
                    return None

                # Segmentleri konuşmacılarla eşle (overlap-based)
                results = []
                for segment in segments:
                    best_speaker = "Unknown"
                    best_overlap = 0.0

                    for turn in turns:
                        overlap_start = max(segment.start, turn["start"])
                        overlap_end = min(segment.end, turn["end"])
                        overlap = max(0.0, overlap_end - overlap_start)

                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_speaker = turn["speaker"]

                    results.append({
                        "speaker": best_speaker,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    })

                # Speaker label smoothing
                results = self._smooth_speaker_labels(results)

                return results

        except Exception as e:
            print(f"\n⚠️ [Transcription Error]: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    logger.warning("Could not delete temporary file %s: %s", tmp_path, exc)
