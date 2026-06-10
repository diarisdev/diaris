"""Live recording and transcription pipeline."""

from __future__ import annotations

import logging
import os
import queue
import threading
import wave
from dataclasses import dataclass, field

import keyboard

# pyrefly: ignore [missing-import]
import pyaudiowpatch as pyaudio

from .audio.device import auto_detect_device
from .audio.vad import VADEngine
from .config import (
    FRAME_DURATION_MS,
    MAX_CHUNK_DURATION_MS,
    OUTPUT_FILENAME,
    SAVE_AUDIO_FILE,
    SHORT_SILENCE_LIMIT,
    SILENCE_LIMIT,
    SOFT_CHUNK_DURATION_MS,
    ensure_output_dir,
)
from .core.ai_worker import AIWorker
from .core.formatting import format_results

logger = logging.getLogger(__name__)
FORMAT = pyaudio.paInt16
STOP_SENTINEL = None


@dataclass
class RecordingState:
    frames: list[bytes] = field(default_factory=list)
    chunk_buffer: list[bytes] = field(default_factory=list)
    silence_counter: int = 0
    has_spoken: bool = False

    @property
    def chunk_duration_ms(self) -> int:
        return len(self.chunk_buffer) * FRAME_DURATION_MS

    def reset_chunk(self) -> None:
        self.chunk_buffer = []
        self.silence_counter = 0
        self.has_spoken = False


def _emit_status(message: str, on_status_change=None) -> None:
    print(message)
    if on_status_change:
        on_status_change(message)


def _diarization_loop(diarization_queue, ai_worker, on_speaker_update=None):
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

            # Run Pyannote diarization in background
            results = ai_worker.run_diarization(
                waveform_16k, sample_rate, chunk_duration_ms, transcribed_segments
            )

            # Format and send update
            formatted_str = format_results(results, return_str=True)
            if formatted_str:
                if on_speaker_update:
                    on_speaker_update({"segment_index": segment_index, "text": formatted_str})
                else:
                    print(f"\n[Diarization Güncellemesi] Segment {segment_index}:\n{formatted_str}\n")
        except Exception:
            logger.exception("Diarization worker failed in background loop")
        finally:
            diarization_queue.task_done()


def _worker_loop(audio_queue, diarization_queue, ai_worker, on_transcription=None):
    """Process queued audio chunks until a sentinel is received."""
    if not ai_worker.load_models():
        return

    segment_index = 0

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
                            audio_queue.put(next_task)
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
            output = ai_worker.process_chunk(chunk_bytes, is_final=is_final)
            if not output:
                continue

            results = output.get("results", [])

            if is_final:
                formatted_str = format_results(results, return_str=True)
                # Send the final transcript immediately with "Çözümleniyor..." speaker tag
                if formatted_str:
                    if on_transcription:
                        on_transcription({
                            "type": "final",
                            "segment_index": segment_index,
                            "text": formatted_str
                        })
                    else:
                        print("\n" + formatted_str + "\n")

                # Queue for background diarization
                diarization_queue.put({
                    "segment_index": segment_index,
                    "waveform_16k": output["waveform_16k"],
                    "sample_rate": output["sample_rate"],
                    "chunk_duration_ms": output["chunk_duration_ms"],
                    "transcribed_segments": results
                })
                segment_index += 1
            else:
                if results:
                    formatted_str = " ".join(r["text"] for r in results).strip()
                else:
                    formatted_str = ""

                if formatted_str:
                    if on_transcription:
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
    if is_speech:
        state.frames.append(data)
        state.chunk_buffer.append(data)
        state.silence_counter = 0
        state.has_spoken = True
        return "🎙️  [ KONUŞULUYOR ]"

    silence_bytes = b"\x00" * len(data)
    state.frames.append(silence_bytes)

    if state.has_spoken:
        state.chunk_buffer.append(silence_bytes)
        state.silence_counter += 1
        return "⏱️  [ BEKLENİYOR ] "

    return "😶 [ SESSİZLİK ]  "


def _active_silence_limit(chunk_duration_ms: int) -> int:
    if chunk_duration_ms > SOFT_CHUNK_DURATION_MS:
        return SHORT_SILENCE_LIMIT
    return SILENCE_LIMIT


def _flush_chunk_if_ready(state: RecordingState, audio_queue) -> bool:
    duration_ms = state.chunk_duration_ms
    silence_limit = _active_silence_limit(duration_ms)
    should_flush = (
        state.has_spoken
        and (state.silence_counter > silence_limit or duration_ms >= MAX_CHUNK_DURATION_MS)
    )
    if not should_flush:
        return False

    if state.chunk_buffer:
        audio_queue.put({
            "type": "final",
            "data": b"".join(state.chunk_buffer)
        })
    state.reset_chunk()
    return True


def _save_recording(frames, channels, rate, sample_width, on_status_change=None) -> None:
    if not frames or not SAVE_AUDIO_FILE:
        return

    ensure_output_dir()
    _emit_status("\n💾 Ana ses dosyası kaydediliyor...", on_status_change)

    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    _emit_status(f"✅ Dosya kaydedildi: {os.path.abspath(OUTPUT_FILENAME)}", on_status_change)


def run(stop_event=None, on_status_change=None, on_transcription=None, on_speaker_update=None, allow_interactive_device=False, device_index=None):
    """
    Run the live recording and transcription loop.

    GUI callers should keep allow_interactive_device=False so failed auto-detection
    reports a status instead of blocking on input(). CLI callers can set it to True.

    Args:
        device_index: Specific PyAudio device index to use. When provided,
                      auto-detection is skipped entirely.
    """
    p = pyaudio.PyAudio()
    stream = None
    audio_queue = queue.Queue()
    diarization_queue = queue.Queue()
    ai_thread = None
    diarization_thread = None
    state = RecordingState()
    channels = None
    rate = None

    try:
        if device_index is not None:
            device_info = p.get_device_info_by_index(device_index)
            channels = max(int(device_info["maxInputChannels"]), 1)
            rate = int(device_info["defaultSampleRate"])
            print(f"✅ Seçilen cihaz: {device_info['name']}")
            print(f"   Kanal: {channels} | Hız: {rate} Hz")
        else:
            result = auto_detect_device(p, allow_interactive=allow_interactive_device)
            if result is None:
                _emit_status("❌ Uygun ses cihazı bulunamadı.", on_status_change)
                return

            device_info, channels, rate = result
            device_index = device_info["index"]
        frame_size = int(rate * FRAME_DURATION_MS / 1000)

        vad_engine = VADEngine()
        ai_worker = AIWorker(rate=rate, channels=channels)
        
        ai_thread = threading.Thread(
            target=_worker_loop,
            args=(audio_queue, diarization_queue, ai_worker, on_transcription),
            daemon=True,
        )
        ai_thread.start()

        diarization_thread = threading.Thread(
            target=_diarization_loop,
            args=(diarization_queue, ai_worker, on_speaker_update),
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

        msg = "🔴 CANLI DİNLENİYOR VE ÇEVRİLİYOR..."
        print("\n" + "=" * 40 + "\n" + msg + "\n" + "=" * 40 + "\n")
        if on_status_change:
            on_status_change(msg)

        while not _should_stop(stop_event):
            data = stream.read(frame_size, exception_on_overflow=False)
            is_speech, confidence = vad_engine.check_speech(data, rate, channels)
            status = _update_recording_state(state, data, is_speech)

            # Send partial update if speech is active and we've gathered enough frames (every ~300ms)
            if state.has_spoken and len(state.chunk_buffer) > 0 and len(state.chunk_buffer) % 10 == 0:
                audio_queue.put({
                    "type": "partial",
                    "data": b"".join(state.chunk_buffer)
                })

            if _flush_chunk_if_ready(state, audio_queue):
                status = "📦 [ YAPAY ZEKAYA İLETİLDİ ]"

            print(f"Durum: {status} | AI: {confidence:.2f}       ", end="\r")
            if on_status_change:
                on_status_change(f"{status} (AI: {confidence:.2f})")

    except Exception as exc:
        logger.exception("Main loop failed")
        _emit_status(f"\n⚠️ Main Loop Error: {exc}", on_status_change)
    finally:
        if stream:
            stream.stop_stream()
            stream.close()

        if ai_thread and ai_thread.is_alive():
            _emit_status("\n🛑 AI kapatılıyor, lütfen bekleyin...", on_status_change)
            audio_queue.put(STOP_SENTINEL)
            audio_queue.join()
            ai_thread.join(timeout=30)
        elif ai_thread:
            logger.warning("AI worker was not running during shutdown")

        if diarization_thread and diarization_thread.is_alive():
            diarization_queue.put(STOP_SENTINEL)
            diarization_queue.join()
            diarization_thread.join(timeout=30)

        if channels is not None and rate is not None:
            sample_width = p.get_sample_size(FORMAT)
            _save_recording(state.frames, channels, rate, sample_width, on_status_change)

        p.terminate()
        if on_status_change:
            on_status_change("Hazır.")
