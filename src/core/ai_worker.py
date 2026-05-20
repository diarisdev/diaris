"""
AI Worker modülü.
Arka planda çalışan transkripsiyon + konuşmacı ayrıştırma iş parçacığı.
Whisper (transkripsiyon) ve Pyannote (diarization) modellerini yükler ve
kuyruktan gelen ses parçalarını işler.

Warm-up mekanizması:
- İlk N saniye boyunca embedding toplanır (konuşmacı etiketi atanmaz)
- Warm-up bitince toplanan embedding'ler kümelenir → başlangıç konuşmacıları oluşur
- Sonrasında yeni chunk'lar bu baseline'lara göre eşlenir
"""

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


class SpeakerTracker:
    """
    Embedding-based konuşmacı takip sistemi (warm-up destekli).
    
    Faz 1 (Warm-up): Embedding'ler toplanır, konuşmacı etiketi atanmaz.
    Faz 2 (Aktif):   Toplanan embedding'ler kümelenir → baseline oluşur.
                      Yeni embedding'ler baseline'larla karşılaştırılır.
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

        Args:
            embedding: torch.Tensor — konuşmacı embedding'i
            chunk_duration_ms: Bu chunk'ın süresi (ms)

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
        Toplanan embedding'leri kümeleyerek başlangıç konuşmacılarını oluşturur.
        Basit greedy clustering: threshold'u geçen embedding'ler aynı kümeye gider.
        """
        if not self._warmup_buffer:
            self._warmup_complete = True
            return

        clusters = []  # list of lists of embeddings

        for emb in self._warmup_buffer:
            matched = False
            for cluster in clusters:
                # Küme merkeziyle karşılaştır
                centroid = torch.stack(cluster).mean(dim=0)
                score = torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), centroid.unsqueeze(0)
                ).item()
                if score >= self.threshold:
                    cluster.append(emb)
                    matched = True
                    break

            if not matched:
                clusters.append([emb])

        # Her kümeden bir konuşmacı oluştur
        for cluster_embs in clusters:
            centroid = torch.stack(cluster_embs).mean(dim=0)
            label = self._next_label()
            self.known_speakers[label] = centroid

        self._warmup_complete = True
        self._warmup_buffer = []  # Belleği serbest bırak

        speaker_list = ", ".join(self.known_speakers.keys())
        print(f"\n✅ [Warm-up Complete] {len(self.known_speakers)} speaker(s) detected: {speaker_list}")
        print(f"   ({self._warmup_audio_ms / 1000:.1f}s audio processed during warm-up)\n")

    def map_speakers(self, embeddings_dict):
        """
        Embedding'lere göre konuşmacıları eşler (warm-up sonrası).

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

            if best_match and best_score >= self.threshold:
                mapping[local_label] = best_match
                # Embedding'i güncelle (running average)
                self.known_speakers[best_match] = (
                    0.7 * self.known_speakers[best_match] + 0.3 * emb
                )
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
        self._loaded = False
        self.speaker_tracker = SpeakerTracker()

    def load_models(self):
        """Modelleri yükler. İlk çağrıda bir kez çalışır."""
        if self._loaded:
            return True

        print(f"\n🧠 [AI Worker] {DEVICE.upper()} üzerinde başlatılıyor...")
        try:
            # Diarizer: Yerel config dosyasından yükle
            if os.path.exists(DIARIZATION_CONFIG_PATH):
                print(f"📂 [AI Worker] Local diarization config: {DIARIZATION_CONFIG_PATH}")
                self.diarizer = Pipeline.from_pretrained(
                    DIARIZATION_CONFIG_PATH,
                )
            else:
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
        """Stereo int16 → mono float32, normalized."""
        if audio_np_int16.ndim > 1 and audio_np_int16.shape[1] > 1:
            mono = audio_np_int16[:, 0].astype(np.float32) / 32768.0
        else:
            mono = audio_np_int16.flatten().astype(np.float32) / 32768.0

        peak = np.abs(mono).max()
        if peak > 0.01:
            mono = mono / peak * 0.95
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

    def _extract_speaker_embeddings(self, waveform_16k, turns):
        """
        Her konuşmacı için ses bölümlerini ayırıp embedding çıkarır.

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

            spk_waveform = torch.cat(spk_audio_parts).unsqueeze(0)

            # En az 0.5s ses gerekli
            if spk_waveform.shape[1] < 8000:
                continue

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

                embeddings_dict[spk] = emb_tensor.squeeze().cpu()

            except Exception as e:
                print(f"  ⚠️ [Embedding] {spk}: {e}")

        return embeddings_dict

    def _get_chunk_duration_ms(self, chunk_bytes):
        """Chunk süresini ms cinsinden hesaplar."""
        num_samples = len(chunk_bytes) // (2 * self.channels)  # int16 = 2 bytes
        return (num_samples / self.rate) * 1000

    def process_chunk(self, chunk_bytes):
        """
        Bir ses parçasını işler: transkripsiyon + konuşmacı ayrıştırma.

        Warm-up fazında:  Transkripsiyon yapar, embedding toplar, konuşmacı "..." olarak gösterilir.
        Aktif fazda:      Transkripsiyon + diarization + embedding eşleme ile gerçek konuşmacı atanır.

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

            # --- Embedding çıkar ---
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
                except Exception:
                    pass


def format_results(results, return_str=False):
    """
    AI sonuçlarını okunabilir formatta yazdırır veya döndürür.

    Args:
        results: process_chunk'tan dönen sonuç listesi
        return_str: True ise sonucu metin olarak döndürür, False ise terminale basar.
    """
    if not results:
        return "" if return_str else None

    lines = ["-" * 50]
    for r in results:
        lines.append(f"[{r['speaker']}] {r['start']:.1f}s - {r['end']:.1f}s: {r['text']}")
    lines.append("-" * 50)
    
    out_str = "\n".join(lines)
    
    if return_str:
        return out_str
        
    print("\n" + out_str + "\n")
