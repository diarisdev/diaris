"""
AI Worker modülü.
Arka planda çalışan transkripsiyon + konuşmacı ayrıştırma iş parçacığı.
Whisper (transkripsiyon) ve Pyannote (diarization) modellerini yükler ve
kuyruktan gelen ses parçalarını işler.

İyileştirmeler:
- 16kHz resample (Pyannote'un beklediği format)
- Ses normalizasyonu
- Overlap-based konuşmacı-segment eşleme
- Embedding-based cross-chunk konuşmacı takibi
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
    DIARIZATION_EMBEDDING_THRESHOLD,
)


class SpeakerTracker:
    """
    Embedding-based konuşmacı takip sistemi.
    
    Her chunk'taki konuşmacının ses parmak izini (embedding) çıkarır ve
    önceki konuşmacılarla cosine similarity ile karşılaştırır.
    Böylece farklı chunk'larda bile aynı/farklı konuşmacıyı ayırt eder.
    """

    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else DIARIZATION_EMBEDDING_THRESHOLD
        self.known_speakers = {}  # {global_label: embedding_tensor}
        self._next_id = 0

    def _next_label(self):
        label = f"Konuşmacı {self._next_id + 1}"
        self._next_id += 1
        return label

    def map_speakers(self, embeddings_dict):
        """
        Embedding'lere göre konuşmacıları eşler.

        Args:
            embeddings_dict: {local_label: embedding_tensor}

        Returns:
            dict: {local_label: global_label}
        """
        mapping = {}

        for local_label, emb in embeddings_dict.items():
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
                print(f"  🔗 {local_label} → {best_match} (benzerlik: {best_score:.3f})")
            else:
                new_label = self._next_label()
                mapping[local_label] = new_label
                self.known_speakers[new_label] = emb.clone()
                if best_match:
                    print(f"  🆕 {local_label} → {new_label} (en yakın: {best_match}, skor: {best_score:.3f})")
                else:
                    print(f"  🆕 {local_label} → {new_label} (ilk konuşmacı)")

        return mapping

    def map_speakers_fallback(self, local_labels):
        """
        Embedding yoksa basit sıralı eşleme (fallback).
        Her yeni label'a yeni isim atar.
        """
        mapping = {}
        for label in local_labels:
            mapping[label] = self._next_label()
        return mapping


class AIWorker:
    """
    Transkripsiyon ve konuşmacı ayrıştırma motoru.
    
    Whisper ile metne çevirir, Pyannote ile konuşmacıları ayırır,
    embedding ile konuşmacıları chunk'lar arası takip eder.
    """

    def __init__(self, rate=None, channels=None):
        """
        Args:
            rate: Örnekleme hızı (Hz). None ise config varsayılanı kullanılır.
            channels: Kanal sayısı. None ise config varsayılanı kullanılır.
        """
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
            # Diarizer: Yerel config dosyasından yükle (tuned parametreler)
            if os.path.exists(DIARIZATION_CONFIG_PATH):
                print(f"📂 [AI Worker] Yerel diarization config kullanılıyor: {DIARIZATION_CONFIG_PATH}")
                self.diarizer = Pipeline.from_pretrained(
                    DIARIZATION_CONFIG_PATH,
                )
            else:
                print("⚠️ [AI Worker] Yerel config bulunamadı, HuggingFace'den yükleniyor...")
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
                print("✅ [AI Worker] Embedding modeli yüklendi (cross-chunk takip aktif)")
            except Exception as emb_err:
                print(f"⚠️ [AI Worker] Embedding modeli yüklenemedi: {emb_err}")
                print("   Cross-chunk konuşmacı takibi devre dışı.")
                self.embedding_model = None

            self.transcriber = WhisperModel(
                WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE
            )
            self._loaded = True
            print("✅ [AI Worker] Modeller yüklendi, ses bekleniyor...\n")
            return True
        except Exception as e:
            print(f"❌ [AI Worker Error] Modeller yüklenemedi. Hata: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _to_mono_float32(self, audio_np_int16):
        """
        Stereo int16 → mono float32 dönüşümü.
        İlk kanalı alır (mean yerine — stereo karışma riski yok).
        Ses seviyesini normalize eder.
        """
        if audio_np_int16.ndim > 1 and audio_np_int16.shape[1] > 1:
            mono = audio_np_int16[:, 0].astype(np.float32) / 32768.0
        else:
            mono = audio_np_int16.flatten().astype(np.float32) / 32768.0

        # Normalize: peak seviyeyi -1..1 aralığına getir
        peak = np.abs(mono).max()
        if peak > 0.01:
            mono = mono / peak * 0.95

        return mono

    def _resample_for_pyannote(self, mono_float32):
        """
        Pyannote'un beklediği 16kHz sample rate'e resample eder.
        """
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

        Args:
            waveform_16k: 16kHz mono waveform tensor (1, samples)
            turns: Diarization turn'leri [{start, end, speaker}, ...]

        Returns:
            dict: {local_speaker_label: embedding_tensor}
        """
        if self.embedding_model is None:
            return {}

        embeddings_dict = {}
        unique_speakers = set(t["speaker"] for t in turns)

        for spk in unique_speakers:
            # Bu konuşmacının tüm turn'lerindeki sesi topla
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

            # En az 0.5 saniye ses gerekli (güvenilir embedding için)
            if spk_waveform.shape[1] < 8000:
                print(f"  ⚠️ {spk}: Ses çok kısa ({spk_waveform.shape[1]/16000:.1f}s), embedding atlanıyor")
                continue

            try:
                emb_output = self.embedding_model({
                    "waveform": spk_waveform,
                    "sample_rate": 16000,
                })
                # numpy → tensor dönüşümü
                if isinstance(emb_output, np.ndarray):
                    emb_tensor = torch.from_numpy(emb_output).float()
                elif isinstance(emb_output, torch.Tensor):
                    emb_tensor = emb_output.float()
                else:
                    emb_tensor = torch.tensor(emb_output).float()

                emb_tensor = emb_tensor.squeeze()
                # CPU'da tut (cosine similarity için)
                embeddings_dict[spk] = emb_tensor.cpu()

            except Exception as e:
                print(f"  ⚠️ [Embedding] {spk}: {e}")

        return embeddings_dict

    def process_chunk(self, chunk_bytes):
        """
        Bir ses parçasını işler: transkripsiyon + konuşmacı ayrıştırma.

        Args:
            chunk_bytes: Ham ses verisi (bytes, int16)

        Returns:
            list[dict] veya None: Her segment için {speaker, start, end, text}.
            Sonuç yoksa None.
        """
        if not self._loaded:
            return None

        # Boş veri kontrolü
        if len(chunk_bytes) == 0:
            return None

        audio_np_int16 = np.frombuffer(chunk_bytes, dtype=np.int16).reshape(-1, self.channels)

        if audio_np_int16.size == 0:
            return None

        # Geçici WAV dosyası oluştur (Whisper için)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_np_int16, self.rate)
                tmp_path = tmp_file.name

            # --- Pyannote için ses hazırla ---
            mono_float32 = self._to_mono_float32(audio_np_int16)
            waveform_16k, sample_rate = self._resample_for_pyannote(mono_float32)
            pyannote_input = {"waveform": waveform_16k, "sample_rate": sample_rate}

            # Diarization
            pipeline_output = self.diarizer(pyannote_input)
            if hasattr(pipeline_output, "speaker_diarization"):
                diarization = pipeline_output.speaker_diarization
            else:
                diarization = pipeline_output

            # Diarization turn'lerini listele
            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })

            # --- Embedding-based konuşmacı eşleme ---
            if turns:
                embeddings_dict = self._extract_speaker_embeddings(waveform_16k, turns)

                if embeddings_dict:
                    # Embedding ile eşle (doğru yöntem)
                    speaker_mapping = self.speaker_tracker.map_speakers(embeddings_dict)
                    # Embedding'i olmayan konuşmacılar için fallback
                    for t in turns:
                        if t["speaker"] not in speaker_mapping:
                            speaker_mapping[t["speaker"]] = "Bilinmeyen"
                else:
                    # Embedding yoksa fallback
                    local_labels = list(set(t["speaker"] for t in turns))
                    speaker_mapping = self.speaker_tracker.map_speakers_fallback(local_labels)

                for turn in turns:
                    turn["speaker"] = speaker_mapping.get(turn["speaker"], turn["speaker"])

            # Transkripsiyon
            segments, _ = self.transcriber.transcribe(
                tmp_path, word_timestamps=True, language=WHISPER_LANGUAGE
            )
            segments = list(segments)

            if len(segments) == 0:
                return None

            # --- Segmentleri konuşmacılarla eşle (overlap-based) ---
            results = []
            for segment in segments:
                best_speaker = "Bilinmeyen"
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
