"""Live recording and transcription pipeline."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import wave
from collections import deque
from concurrent import futures
from dataclasses import dataclass, field

import keyboard

# pyrefly: ignore [missing-import]
import pyaudiowpatch as pyaudio

from .audio.device import auto_detect_device
from .audio.vad import VADEngine
from .config import (
    CHUNK_OVERLAP_MS,
    FRAME_DURATION_MS,
    MAX_CHUNK_DURATION_MS,
    OUTPUT_FILENAME,
    PARTIAL_WINDOW_MS,
    PRE_ROLL_MS,
    SAVE_AUDIO_FILE,
    SHORT_SILENCE_LIMIT,
    SILENCE_LIMIT,
    SOFT_CHUNK_DURATION_MS,
    ensure_output_dir,
    WHISPER_LANGUAGE,
    LOCAL_MODELS_DIR,
)
from .core.ai_worker import AIWorker
from .core.formatting import format_results
from .translation import CachingTranslationEngine, get_translation_engine

logger = logging.getLogger(__name__)
FORMAT = pyaudio.paInt16
STOP_SENTINEL = None

# Frame sayısına çevrilmiş pre-roll / overlap / partial-pencere büyüklükleri.
PRE_ROLL_FRAMES = max(0, PRE_ROLL_MS // FRAME_DURATION_MS)
OVERLAP_FRAMES = max(0, CHUNK_OVERLAP_MS // FRAME_DURATION_MS)
PARTIAL_WINDOW_FRAMES = max(1, PARTIAL_WINDOW_MS // FRAME_DURATION_MS)

# UI durum güncellemeleri en fazla bu aralıkta bir gönderilir (durum METNİ
# değişirse anında). Her 30 ms frame'de sinyal + stil yenilemek Qt tarafında
# saniyede 33 gereksiz relayout üretiyordu.
STATUS_EMIT_INTERVAL_S = 0.2


def _make_preroll() -> deque:
    return deque(maxlen=PRE_ROLL_FRAMES)


@dataclass
class RecordingState:
    frames: list[bytes] = field(default_factory=list)
    chunk_buffer: list[bytes] = field(default_factory=list)
    silence_counter: int = 0
    has_spoken: bool = False
    # Pre-roll halkası: konuşma başlamadan önceki son N frame'in HAM sesi.
    # Konuşma algılandığında chunk'ın başına eklenir (kelime başı kırpılmaz).
    preroll: deque = field(default_factory=_make_preroll)
    # Overlap taşıması: max-süre kesmesinde bir önceki chunk'tan taşınan ve
    # ZATEN transkribe edilmiş önek (ms). Sonraki transkripsiyonda kırpılır.
    trim_ms: int = 0
    # Global frame sayacı + aktif chunk'ın global başlangıç frame'i.
    # Offline replay/benchmark sürücüleri mutlak zaman damgası için kullanır.
    total_frames: int = 0
    chunk_start_frame: int = 0

    @property
    def chunk_duration_ms(self) -> int:
        return len(self.chunk_buffer) * FRAME_DURATION_MS

    @property
    def chunk_start_ms(self) -> int:
        return self.chunk_start_frame * FRAME_DURATION_MS

    def reset_chunk(self) -> None:
        self.chunk_buffer = []
        self.silence_counter = 0
        self.has_spoken = False
        self.trim_ms = 0


def _emit_status(message: str, on_status_change=None) -> None:
    print(message)
    if on_status_change:
        on_status_change(message)


def _submit_final_translation(executor, engine, on_transcription, results,
                              segment_index, source_lang, target_lang) -> None:
    """Final metnin çevirisini ARKA PLANDA yapar; kritik yolu bloklamaz.

    Orijinal final anında basılmıştır; bu iş bitince aynı segment_index'e
    'provisional' işaretli bir güncelleme gönderilir. Diarization güncellemesi
    (konuşmacı bazlı çeviri) her zaman nihai otoritedir — UI, diarize edilmiş
    bir segmente geç gelen provisional çeviriyi uygulamaz. Perf alanları
    (captured_at vb.) bilerek YOKTUR: benchmark toplayıcısı aynı segmenti iki
    kez saymasın.
    """
    def job():
        try:
            for r in results:
                if r.get("text"):
                    r["text"] = engine.translate(r["text"], source_lang, target_lang)
            formatted = format_results(results, return_str=True)
            if formatted:
                on_transcription({
                    "type": "final",
                    "segment_index": segment_index,
                    "text": formatted,
                    "provisional": True,
                })
        except Exception:
            logger.exception("Async final translation failed")

    executor.submit(job)


def _submit_partial_translation(executor, engine, on_transcription, text,
                                gen_holder, gen, source_lang, target_lang) -> None:
    """Partial metni arka planda çevirir; 'en yenisi kazanır'.

    Partial'lar saniyede birkaç kez yenilenir; kuyrukta bekleyen bayat bir
    çeviri hem işten önce hem de emit'ten önce nesil kontrolüyle düşürülür.
    """
    def job():
        try:
            if gen != gen_holder["value"]:
                return  # daha yeni bir partial var — bayat işi hiç yapma
            translated = engine.translate(text, source_lang, target_lang)
            if gen != gen_holder["value"]:
                return  # çeviri sürerken yenisi geldi — bayat sonucu basma
            on_transcription({"type": "partial", "text": translated})
        except Exception:
            logger.exception("Async partial translation failed")

    executor.submit(job)


def _diarization_loop(diarization_queue, ai_worker, on_speaker_update=None, translation_engine=None):
    """Process queued diarization tasks in the background."""
    while True:
        task = diarization_queue.get()
        try:
            if task is STOP_SENTINEL:
                return

            segment_index = task["segment_index"]
            waveform_16k = task["waveform_16k"]
            sample_rate = task["sample_rate"]
            chunk_duration_ms = task["chunk_duration_ms"]
            transcribed_segments = task["transcribed_segments"]
            source_lang = task.get("source_lang")
            target_lang = task.get("target_lang")
            is_translation_needed = task.get("is_translation_needed", False)
            captured_at = task.get("captured_at")   # perf: chunk kapanış damgası
            stt_ms = task.get("stt_ms")

            # Kuyruk bekleme telemetrisi: diarization gerçek-zamandan yavaş
            # kalırsa konuşmacı etiketleri sessizce gecikmeye başlar — görünür kıl.
            enqueued_at = task.get("enqueued_at")
            diar_queue_ms = (time.time() - enqueued_at) * 1000.0 if enqueued_at else None
            if diar_queue_ms is not None and diar_queue_ms > 5000:
                logger.warning("Diarization queue lagging: %.0f ms wait", diar_queue_ms)

            # Run Pyannote diarization in background.
            # Sonuç: konuşmacıya göre bölünmüş, ÇEVRİLMEMİŞ segmentler.
            _t_diar = time.perf_counter()
            results = ai_worker.run_diarization(
                waveform_16k, sample_rate, chunk_duration_ms, transcribed_segments
            )
            diar_ms = (time.perf_counter() - _t_diar) * 1000.0

            # Çeviri: konuşmacı sınırından bölme SONRASI. TÜM segmentler TEK batch
            # çağrısında çevrilir — segment başına ayrı çağrı (özellikle yerel CPU
            # NLLB'de) diarization kuyruğunu birden çok inference ile dondururuyordu.
            if is_translation_needed and translation_engine and results:
                texts = [r.get("text") or "" for r in results]
                translated = translation_engine.translate_many(
                    texts, source_lang, target_lang
                )
                for r, t in zip(results, translated):
                    r["text"] = t

            # Format and send update
            formatted_str = format_results(results, return_str=True)
            if formatted_str:
                if on_speaker_update:
                    on_speaker_update({
                        "segment_index": segment_index,
                        "text": formatted_str,
                        # perf ölçümü (varlığı consumer'ı etkilemez)
                        "captured_at": captured_at,
                        "stt_ms": stt_ms,
                        "diar_ms": diar_ms,
                        "diar_queue_ms": diar_queue_ms,
                        "chunk_duration_ms": chunk_duration_ms,
                        "emitted_at": time.time(),
                    })
                else:
                    print(f"\n[Diarization Güncellemesi] Segment {segment_index}:\n{formatted_str}\n")
        except Exception:
            logger.exception("Diarization worker failed in background loop")
        finally:
            diarization_queue.task_done()


def _worker_loop(audio_queue, diarization_queue, ai_worker, translation_engine,
                 get_lang_pair, on_transcription=None, translation_executor=None):
    """Process queued audio chunks until a sentinel is received."""
    if not ai_worker.load_models():
        return

    segment_index = 0
    # Partial çevirileri için "en yenisi kazanır" nesil sayacı (closure'larla
    # paylaşılan mutable holder).
    partial_gen = {"value": 0}

    while True:
        task = audio_queue.get()
        try:
            if task is STOP_SENTINEL:
                return

            task_type = task.get("type", "final")
            chunk_bytes = task.get("data", b"")

            # Coalesce partial tasks to reduce latency
            if task_type == "partial":
                try:
                    while True:
                        next_task = audio_queue.get_nowait()
                        if next_task is STOP_SENTINEL:
                            # Put sentinel back so we pick it up on the
                            # next iteration and exit cleanly.
                            audio_queue.put(next_task)
                            # Mark the consumed next_task slot as done.
                            audio_queue.task_done()
                            break
                        if next_task.get("type") == "partial":
                            audio_queue.task_done()
                            task = next_task
                            chunk_bytes = task.get("data", b"")
                        else:
                            # A final task is waiting! Skip this partial and process the final task
                            audio_queue.task_done()
                            task = next_task
                            task_type = task.get("type", "final")
                            chunk_bytes = task.get("data", b"")
                            break
                except queue.Empty:
                    pass

            is_final = (task_type == "final")
            
            # Query active languages dynamically
            source_lang, target_lang = get_lang_pair()
            is_translation_needed = (source_lang.split("-")[0].lower() != target_lang.split("-")[0].lower())
            
            captured_at = task.get("captured_at")  # perf: chunk kapanış zaman damgası
            _t_stt = time.perf_counter()
            output = ai_worker.process_chunk(
                chunk_bytes,
                is_final=is_final,
                language=source_lang,
                trim_prefix_ms=task.get("trim_prefix_ms", 0),
            )
            stt_ms = (time.perf_counter() - _t_stt) * 1000.0
            if not output:
                continue

            results = output.get("results", [])

            # Orijinal (çevrilmemiş) per-segment sonuçları kelime damgalarıyla
            # birlikte sakla — diarization aşamasında konuşmacı sınırından bölme
            # ve doğru konuşmacıya çeviri için kullanılır.
            original_segments = [dict(r) for r in results]

            # Combine all results into a single segment for better translation context and cleaner UI
            if results:
                combined_text = " ".join(r["text"].strip() for r in results).strip()
                start_time = results[0]["start"]
                end_time = results[-1]["end"]
                speaker = results[0]["speaker"]
                results = [{
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": combined_text
                }]

            # Çeviri KRİTİK YOLDAN ÇIKARILDI. Eski davranış burada senkron
            # translate() çağırıyordu; online motorlarda (Google/DeepL) bu, her
            # final'de bir ağ turu demekti — final altyazıyı VE kuyruktaki bir
            # sonraki chunk'ı bloklıyordu. Artık orijinal metin ANINDA basılır,
            # çeviri translation_executor'da yapılıp güncelleme olarak gelir.

            if is_final:
                formatted_str = format_results(results, return_str=True)
                # Send the final transcript immediately (orijinal dil)
                if formatted_str:
                    if on_transcription:
                        on_transcription({
                            "type": "final",
                            "segment_index": segment_index,
                            "text": formatted_str,
                            # perf ölçümü (varlığı consumer'ı etkilemez)
                            "captured_at": captured_at,
                            "stt_ms": stt_ms,
                            "emitted_at": time.time(),
                        })
                    else:
                        print("\n" + formatted_str + "\n")

                # Çeviri gerekli ise arka planda çevir, 'provisional' güncelleme
                # olarak aynı segment_index'e gönder. (CLI modunda diarization
                # güncellemesi zaten çevrilmiş metni basar; ikinci bir baskı yok.)
                if (is_translation_needed and translation_engine and results
                        and on_transcription and translation_executor):
                    _submit_final_translation(
                        translation_executor, translation_engine, on_transcription,
                        [dict(r) for r in results], segment_index,
                        source_lang, target_lang,
                    )

                # Queue for background diarization.
                # Çevrilmemiş per-segment (kelime damgalı) sonuçları gönderiyoruz;
                # çeviri, konuşmacıya göre bölme sonrası diarization thread'inde
                # yapılır. Dil çifti enqueue anında sabitlenir (yarış önleme).
                diarization_queue.put({
                    "segment_index": segment_index,
                    "waveform_16k": output["waveform_16k"],
                    "sample_rate": output["sample_rate"],
                    "chunk_duration_ms": output["chunk_duration_ms"],
                    "transcribed_segments": original_segments,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "is_translation_needed": is_translation_needed,
                    # perf: gecikme/RTF için zaman damgaları
                    "captured_at": captured_at,
                    "stt_ms": stt_ms,
                    "enqueued_at": time.time(),  # kuyruk bekleme telemetrisi
                })
                segment_index += 1
            else:
                if results:
                    formatted_str = " ".join(r["text"] for r in results).strip()
                else:
                    formatted_str = ""

                if formatted_str:
                    if (is_translation_needed and translation_engine
                            and on_transcription and translation_executor):
                        # Çevrilmiş partial'ı arka planda üret; bayat olanlar
                        # nesil sayacıyla düşer (worker asla bloklanmaz).
                        partial_gen["value"] += 1
                        _submit_partial_translation(
                            translation_executor, translation_engine,
                            on_transcription, formatted_str,
                            partial_gen, partial_gen["value"],
                            source_lang, target_lang,
                        )
                    elif on_transcription:
                        on_transcription({"type": "partial", "text": formatted_str})
                    else:
                        print(f"\r\033[K[Canlı] {formatted_str}", end="", flush=True)

        except Exception:
            logger.exception("AI worker failed while processing a chunk")
        finally:
            audio_queue.task_done()


def _should_stop(stop_event) -> bool:
    if stop_event and stop_event.is_set():
        return True
    if stop_event:
        return False
    return keyboard.is_pressed("ctrl+q")


def _update_recording_state(state: RecordingState, data: bytes, is_speech: bool) -> str:
    """Bir frame'i kayıt durumuna işler.

    HAM SES KORUNUR: VAD kararı yalnızca chunk SINIRLARINI belirler, sesi asla
    yeniden yazmaz. (Eski davranış konuşma-dışı frame'leri sıfırla değiştiriyordu;
    VAD'in ortada kaçırdığı her frame sert bir süreksizliğe dönüşüp Whisper'da
    silme/halüsinasyon üretiyordu.)
    """
    frame_idx = state.total_frames
    state.total_frames += 1
    # Tüm oturum sesi yalnızca kaydetme AÇIKKEN biriktirilir. Koşulsuz
    # biriktirme, SAVE_AUDIO_FILE=false iken bile ~190 KB/s (~700 MB/saat)
    # ölü RAM + büyüyen GC duraksamaları demekti.
    if SAVE_AUDIO_FILE:
        state.frames.append(data)

    if is_speech:
        if not state.has_spoken:
            # Yeni chunk: pre-roll'daki ham frame'leri başa ekle — VAD konuşmayı
            # geç yakaladıysa kelimenin başı yine de chunk'ın içindedir.
            carried = list(state.preroll)
            state.preroll.clear()
            state.chunk_buffer.extend(carried)
            state.chunk_start_frame = frame_idx - len(carried)
            state.has_spoken = True
        state.chunk_buffer.append(data)
        state.silence_counter = 0
        return "[ KONUŞULUYOR ]"

    if state.has_spoken:
        # Hangover: konuşma sonrası bekleme frame'leri de HAM olarak eklenir.
        state.chunk_buffer.append(data)
        state.silence_counter += 1
        return "[ BEKLENİYOR ] "

    state.preroll.append(data)
    return "[ SESSİZLİK ]  "


def _active_silence_limit(chunk_duration_ms: int) -> int:
    if chunk_duration_ms > SOFT_CHUNK_DURATION_MS:
        return SHORT_SILENCE_LIMIT
    return SILENCE_LIMIT


def _flush_chunk_if_ready(state: RecordingState, audio_queue) -> str | None:
    """Chunk hazırsa kuyruğa koyar.

    Returns:
        None  → flush yok.
        "silence" → konuşma sessizlikle bitti (VAD akış durumu sıfırlanabilir).
        "max"     → konuşmanın ORTASINDAN max-süre kesmesi; son CHUNK_OVERLAP_MS
                    bir sonraki chunk'a taşındı ve orada transkripsiyondan
                    kırpılacak (trim_prefix_ms). Böylece kesim noktasındaki
                    kelime tam bağlamla yeniden görülür ama çift sayılmaz.
    """
    duration_ms = state.chunk_duration_ms
    silence_limit = _active_silence_limit(duration_ms)
    if not state.has_spoken:
        return None

    silence_cut = state.silence_counter > silence_limit
    max_cut = duration_ms >= MAX_CHUNK_DURATION_MS
    if not (silence_cut or max_cut):
        return None

    if state.chunk_buffer:
        audio_queue.put({
            "type": "final",
            "data": b"".join(state.chunk_buffer),
            # Performans ölçümü: chunk'ın KAPANDIĞI (son frame yakalandığı) an.
            # Gecikme metriklerinin referans noktası.
            "captured_at": time.time(),
            # Mutlak zaman: chunk verisinin ilk frame'inin global konumu (ms).
            "start_ms": state.chunk_start_ms,
            # Önceki chunk'tan taşınan, zaten transkribe edilmiş önek (ms).
            "trim_prefix_ms": state.trim_ms,
        })

    if max_cut and not silence_cut and OVERLAP_FRAMES > 0:
        # Konuşma sürüyor: kuyruğu taşı, durumu 'konuşuyor' bırak.
        carry = state.chunk_buffer[-OVERLAP_FRAMES:]
        state.chunk_buffer = list(carry)
        state.silence_counter = 0
        state.has_spoken = True
        state.trim_ms = len(carry) * FRAME_DURATION_MS
        state.chunk_start_frame = state.total_frames - len(carry)
        return "max"

    state.reset_chunk()
    return "silence"


def _save_recording(frames, channels, rate, sample_width, on_status_change=None) -> None:
    if not frames or not SAVE_AUDIO_FILE:
        return

    ensure_output_dir()
    _emit_status("\nAna ses dosyası kaydediliyor...", on_status_change)

    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    _emit_status(f"Dosya kaydedildi: {os.path.abspath(OUTPUT_FILENAME)}", on_status_change)


def run(stop_event=None, on_status_change=None, on_transcription=None, on_speaker_update=None, allow_interactive_device=False, device_index=None, get_lang_pair=None):
    """
    Run the live recording and transcription loop.

    GUI callers should keep allow_interactive_device=False so failed auto-detection
    reports a status instead of blocking on input(). CLI callers can set it to True.

    Args:
        device_index: Specific PyAudio device index to use. When provided,
                      auto-detection is skipped entirely.
        get_lang_pair: A callable returning (source_lang, target_lang) dynamically.
    """
    p = pyaudio.PyAudio()
    stream = None
    audio_queue = queue.Queue()
    diarization_queue = queue.Queue()
    ai_thread = None
    diarization_thread = None
    translation_executor = None
    state = RecordingState()
    channels = None
    rate = None

    try:
        if device_index is not None:
            device_info = p.get_device_info_by_index(device_index)
            channels = max(int(device_info["maxInputChannels"]), 1)
            rate = int(device_info["defaultSampleRate"])
            print(f"Seçilen cihaz: {device_info['name']}")
            print(f"   Kanal: {channels} | Hız: {rate} Hz")
        else:
            result = auto_detect_device(p, allow_interactive=allow_interactive_device)
            if result is None:
                _emit_status("Uygun ses cihazı bulunamadı.", on_status_change)
                return

            device_info, channels, rate = result
            device_index = device_info["index"]
        frame_size = int(rate * FRAME_DURATION_MS / 1000)

        vad_engine = VADEngine()
        ai_worker = AIWorker(rate=rate, channels=channels)
        
        # Setup translation config
        if get_lang_pair is None:
            # Fallback for CLI and tests
            def get_lang_pair():
                return WHISPER_LANGUAGE, "tr"

        # Always initialize translation engine for dynamic switching capability
        deepl_key = os.getenv("DEEPL_API_KEY")
        nllb_path = os.path.join(LOCAL_MODELS_DIR, "ctranslate2-nllb-200-distilled-600M")
        # TRANSLATION_ENGINE açıkça verildiyse otomatik zinciri ATLA ve o motoru
        # zorla (örn. "google"). Boş bırakılırsa eski otomatik öncelik korunur:
        # DeepL anahtarı > yerel NLLB > Google (yedek).
        forced_engine = os.getenv("TRANSLATION_ENGINE", "").strip().lower()
        translation_engine = None
        engine_name = "Çeviri Devre Dışı"

        if forced_engine == "google":
            translation_engine = get_translation_engine("google")
            engine_name = "Google Translate (Online - forced)"
        elif forced_engine == "deepl" or (not forced_engine and deepl_key):
            translation_engine = get_translation_engine("deepl", api_key=deepl_key)
            engine_name = "DeepL API (Online)"
        elif forced_engine == "ctranslate2" or (not forced_engine and os.path.exists(nllb_path)):
            translation_engine = get_translation_engine("ctranslate2", model_path=nllb_path)
            if hasattr(translation_engine, 'translator') and translation_engine.translator is not None:
                engine_name = "CTranslate2 (NLLB-200 Local)"
            else:
                engine_name = "Google Translate (Online - Fallback)"
        else:
            translation_engine = get_translation_engine("google")
            engine_name = "Google Translate (Online)"
            
        print(f"[Çeviri Motoru Yüklendi] Motor: {engine_name}")

        # LRU çeviri cache'i: aynı metni (anlık final + diarization geçişi)
        # iki kez çevirmeyi önler — online motorlarda ağ turu tasarrufu.
        if translation_engine is not None:
            translation_engine = CachingTranslationEngine(translation_engine)

        # Çeviri yürütücüsü: ağ/CPU çevirilerini ASR ve capture thread'lerinden
        # tamamen çıkarır (tek işçi; partial'larda "en yenisi kazanır").
        translation_executor = futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="translate"
        )

        ai_thread = threading.Thread(
            target=_worker_loop,
            args=(audio_queue, diarization_queue, ai_worker, translation_engine,
                  get_lang_pair, on_transcription, translation_executor),
            daemon=True,
        )
        ai_thread.start()

        diarization_thread = threading.Thread(
            target=_diarization_loop,
            args=(diarization_queue, ai_worker, on_speaker_update, translation_engine),
            daemon=True,
        )
        diarization_thread.start()

        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frame_size,
        )

        msg = "CANLI DİNLENİYOR VE ÇEVRİLİYOR..."
        print("\n" + "=" * 40 + "\n" + msg + "\n" + "=" * 40 + "\n")
        if on_status_change:
            on_status_change(msg)

        last_status = None
        last_status_emit = 0.0

        while not _should_stop(stop_event):
            data = stream.read(frame_size, exception_on_overflow=False)
            is_speech, confidence = vad_engine.check_speech(data, rate, channels)
            status = _update_recording_state(state, data, is_speech)

            # Partial güncelleme: buffer kısayken ~300 ms'de, uzadıkça ~600 ms'de
            # bir. Backpressure: worker hâlâ bir görevi işliyorsa YENİ partial
            # enqueue edilmez — GPU'yu bayat partial'larla doldurup final'leri
            # bekletmenin anlamı yok (partial cadansı decode hızına uyarlanır).
            # Pencere: yalnızca buffer'ın SON PARTIAL_WINDOW_MS kısmı decode
            # edilir; final her zaman TAM buffer'ı görür (kalite değişmez).
            n_buf = len(state.chunk_buffer)
            partial_interval = 10 if n_buf <= 100 else 20
            if (state.has_spoken and n_buf > 0
                    and n_buf % partial_interval == 0
                    and audio_queue.unfinished_tasks == 0):
                window = state.chunk_buffer[-PARTIAL_WINDOW_FRAMES:]
                audio_queue.put({
                    "type": "partial",
                    "data": b"".join(window),
                    # Pencere buffer başını dışarıda bıraktıysa overlap öneki
                    # zaten pencere dışındadır — trim gereksiz.
                    "trim_prefix_ms": state.trim_ms if n_buf <= PARTIAL_WINDOW_FRAMES else 0,
                })

            flush_reason = _flush_chunk_if_ready(state, audio_queue)
            if flush_reason:
                status = "[ YAPAY ZEKAYA İLETİLDİ ]"
                if flush_reason == "silence":
                    # Konuşma arası: Silero RNN durumu sonsuza dek taşınmasın.
                    vad_engine.reset_stream()

            # Durum güncellemesi: metin değiştiğinde anında, aksi halde en çok
            # 5 Hz. (Eski hali her 30 ms frame'de print + Qt sinyali + stil
            # yenileme tetikliyordu.)
            now = time.time()
            if status != last_status or (now - last_status_emit) >= STATUS_EMIT_INTERVAL_S:
                print(f"Durum: {status} | AI: {confidence:.2f}       ", end="\r")
                if on_status_change:
                    on_status_change(f"{status} (AI: {confidence:.2f})")
                last_status = status
                last_status_emit = now

    except Exception as exc:
        logger.exception("Main loop failed")
        _emit_status(f"\nMain Loop Error: {exc}", on_status_change)
    finally:
        if stream:
            stream.stop_stream()
            stream.close()

        if ai_thread and ai_thread.is_alive():
            _emit_status("\nAI kapatılıyor, lütfen bekleyin...", on_status_change)
            audio_queue.put(STOP_SENTINEL)
            ai_thread.join(timeout=30)
            if ai_thread.is_alive():
                logger.warning("AI worker did not shut down within 30s")
        elif ai_thread:
            logger.warning("AI worker was not running during shutdown")

        if diarization_thread and diarization_thread.is_alive():
            diarization_queue.put(STOP_SENTINEL)
            diarization_thread.join(timeout=30)
            if diarization_thread.is_alive():
                logger.warning("Diarization worker did not shut down within 30s")

        if translation_executor is not None:
            translation_executor.shutdown(wait=False)

        if channels is not None and rate is not None:
            sample_width = p.get_sample_size(FORMAT)
            _save_recording(state.frames, channels, rate, sample_width, on_status_change)

        p.terminate()
        if on_status_change:
            on_status_change("Hazır.")
