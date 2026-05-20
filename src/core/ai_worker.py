"""
AI Worker modülü.
Arka planda çalışan transkripsiyon + konuşmacı ayrıştırma iş parçacığı.
Whisper (transkripsiyon) ve Pyannote (diarization) modellerini yükler ve
kuyruktan gelen ses parçalarını işler.
"""

import os
import tempfile
import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from ..config import (
    DEVICE, COMPUTE_TYPE, HF_TOKEN,
    WHISPER_PATH, DIARIZATION_MODEL,
    DEFAULT_RATE, DEFAULT_CHANNELS, WHISPER_LANGUAGE,
)


class AIWorker:
    """
    Transkripsiyon ve konuşmacı ayrıştırma motoru.
    
    Whisper ile metne çevirir, Pyannote ile konuşmacıları ayırır,
    ardından her metin segmentini en yakın konuşmacıya eşler.
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
        self._loaded = False

    def load_models(self):
        """Modelleri yükler. İlk çağrıda bir kez çalışır."""
        if self._loaded:
            return True

        print(f"\n🧠 [AI Worker] {DEVICE.upper()} üzerinde başlatılıyor...")
        try:
            self.diarizer = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                token=HF_TOKEN
            )
            if DEVICE == "cuda":
                self.diarizer.to(torch.device("cuda"))

            self.transcriber = WhisperModel(
                WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE
            )
            self._loaded = True
            print("✅ [AI Worker] Modeller yüklendi, ses bekleniyor...\n")
            return True
        except Exception as e:
            print(f"❌ [AI Worker Error] Modeller yüklenemedi. Hata: {e}")
            return False

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

            # Pyannote için mono float32 waveform hazırla
            mono_float32 = audio_np_int16.mean(axis=1).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(mono_float32).unsqueeze(0)
            pyannote_input = {"waveform": waveform, "sample_rate": self.rate}

            # Diarization
            pipeline_output = self.diarizer(pyannote_input)
            if hasattr(pipeline_output, "speaker_diarization"):
                diarization = pipeline_output.speaker_diarization
            else:
                diarization = pipeline_output

            # Transkripsiyon
            segments, _ = self.transcriber.transcribe(
                tmp_path, word_timestamps=True, language=WHISPER_LANGUAGE
            )
            segments = list(segments)

            if len(segments) == 0:
                return None

            # Segmentleri konuşmacılarla eşle
            results = []
            for segment in segments:
                segment_center = segment.start + (segment.end - segment.start) / 2
                current_speaker = "Unknown"

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment_center <= turn.end:
                        current_speaker = speaker
                        break

                results.append({
                    "speaker": current_speaker,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                })

            return results

        except Exception as e:
            print(f"\n⚠️ [Transcription Error]: {e}")
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
