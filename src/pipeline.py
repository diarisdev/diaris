"""
Canlı kayıt ve ana iş akışı modülü.
Ses cihazından canlı okuma, VAD ile konuşma algılama,
ve AI worker thread'e ses parçası gönderme mantığını içerir.
"""

import wave
import os
import queue
import threading
# pyrefly: ignore [missing-import]
import pyaudiowpatch as pyaudio
import keyboard

from .config import DEFAULT_RATE, DEFAULT_CHANNELS, FRAME_DURATION_MS, OUTPUT_FILENAME, SILENCE_LIMIT, SHORT_SILENCE_LIMIT, SOFT_CHUNK_DURATION_MS, MAX_CHUNK_DURATION_MS, SAVE_AUDIO_FILE
from .audio.device import auto_detect_device
from .audio.vad import VADEngine
from .core.ai_worker import AIWorker, format_results

FORMAT = pyaudio.paInt16


def _worker_loop(audio_queue, ai_worker):
    """
    Arka planda çalışan AI iş parçacığı döngüsü.
    Kuyruktan ses parçaları alır, işler ve sonuçları yazdırır.

    Args:
        audio_queue: Ses parçalarını içeren thread-safe kuyruk
        ai_worker: AIWorker instance
    """
    if not ai_worker.load_models():
        return

    while True:
        chunk_bytes = audio_queue.get()
        if chunk_bytes is None:
            break

        results = ai_worker.process_chunk(chunk_bytes)
        format_results(results)
        audio_queue.task_done()


def run():
    """
    Ana canlı kayıt ve transkripsiyon döngüsü.
    
    1. Ses cihazını otomatik algılar (loopback veya kullanıcı seçimi)
    2. Cihazın kanal sayısı ve örnekleme hızını otomatik alır
    3. VAD motorunu başlatır
    4. AI worker thread'i arka planda başlatır
    5. Canlı ses okur, VAD ile filtreler, konuşma bitince chunk'ı AI'a gönderir
    6. CTRL+Q tuşlarıyla durur, ses dosyasını kaydeder
    """
    p = pyaudio.PyAudio()

    # Otomatik cihaz algılama — hardcoded değer yok
    result = auto_detect_device(p)
    if result is None:
        print("❌ Uygun ses cihazı bulunamadı.")
        p.terminate()
        return

    device_info, channels, rate = result
    device_index = device_info["index"]
    frame_size = int(rate * FRAME_DURATION_MS / 1000)

    # VAD motoru
    vad_engine = VADEngine()

    # AI worker + kuyruk — rate ve channels dinamik geçirilir
    audio_queue = queue.Queue()
    ai_worker = AIWorker(rate=rate, channels=channels)
    ai_thread = threading.Thread(
        target=_worker_loop, args=(audio_queue, ai_worker), daemon=True
    )
    ai_thread.start()

    # Kayıt durumu
    frames = []
    chunk_buffer = []
    silence_counter = 0
    has_spoken = False
    stream = None

    try:
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frame_size,
        )

        print("\n" + "=" * 40)
        print("🔴 CANLI DİNLENİYOR VE ÇEVRİLİYOR...")
        print("Durdurmak için 'CTRL + Q' tuşlarına bas.")
        print("=" * 40 + "\n")

        while not keyboard.is_pressed("ctrl+q"):
            data = stream.read(frame_size, exception_on_overflow=False)
            is_speech, conf = vad_engine.check_speech(data, rate, channels)

            if is_speech:
                frames.append(data)
                chunk_buffer.append(data)
                silence_counter = 0
                has_spoken = True
                status = "🎙️  [ KONUŞULUYOR ]"
            else:
                silence_bytes = b'\x00' * len(data)
                frames.append(silence_bytes)

                if has_spoken:
                    chunk_buffer.append(silence_bytes)
                    silence_counter += 1
                    status = "⏱️  [ BEKLENİYOR ] "
                else:
                    status = "😶 [ SESSİZLİK ]  "

            # Chunk süresini hesapla (milisaniye cinsinden)
            current_duration_ms = len(chunk_buffer) * FRAME_DURATION_MS
            
            # Dinamik sessizlik limiti belirleme
            if current_duration_ms > SOFT_CHUNK_DURATION_MS:
                active_silence_limit = SHORT_SILENCE_LIMIT
            else:
                active_silence_limit = SILENCE_LIMIT
                
            # Yapay zekaya gönderme şartları:
            # 1. Dinamik sessizlik süresine ulaşıldıysa (normal veya virgül es'i)
            # 2. VEYA chunk zorunlu maksimum süreye ulaştıysa (MAX_CHUNK_DURATION_MS)
            if has_spoken and (silence_counter > active_silence_limit or current_duration_ms >= MAX_CHUNK_DURATION_MS):
                if len(chunk_buffer) > 0:
                    audio_queue.put(b''.join(chunk_buffer))
                chunk_buffer = []
                silence_counter = 0
                has_spoken = False
                status = "📦 [ YAPAY ZEKAYA İLETİLDİ ]"

            print(f"Durum: {status} | AI: {conf:.2f}       ", end='\r')

    except Exception as e:
        print(f"\n⚠️ Main Loop Error: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()

        print("\n🛑 AI Kapatılıyor, lütfen bekleyin...")
        audio_queue.put(None)
        ai_thread.join()

        if frames and SAVE_AUDIO_FILE:
            print("\n💾 Ana ses dosyası kaydediliyor...")
            with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
            print(f"✅ Dosya kaydedildi: {os.path.abspath(OUTPUT_FILENAME)}")

        p.terminate()
