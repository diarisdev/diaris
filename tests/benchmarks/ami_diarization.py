"""AMI diarization benchmark — DER (Diarization Error Rate).

pyannote modelinin (raw) ya da AIWorker canlı-stream simülasyonunun (aiworker)
konuşmacı ayrımı doğruluğunu AMI alt kümesinde ölçer.

Çalıştırma (proje kökünden):
    python -m tests.benchmarks.ami_diarization --mode raw
    python -m tests.benchmarks.ami_diarization --mode aiworker --max-minutes 5
"""

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*std\\(\\).*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")

import os
import sys
import time
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# CUDA DLL'lerini (cublas/cudnn) model yüklemeden ÖNCE bulunur kıl (main.py gibi)
from src.config import configure_cuda_dll_paths
configure_cuda_dll_paths()

from tests.dataset_managers import AmiDiarizationManager
from tests.metrics import DiarizationEvaluator


def run_diarization_benchmark_raw(max_minutes=None):
    """Saf pyannote modeli ile DER testi."""
    print("\n" + "=" * 70)
    print("🧪  BENCHMARK — Diarization Doğruluk Testi (Saf Model Modu)")
    print("=" * 70)

    print("\n📦 [Adım 1/3] Veri seti hazırlanıyor...")
    dm = AmiDiarizationManager()
    if not dm.download():
        print("❌ Veri seti indirilemedi. Benchmark iptal.")
        return None
    samples = dm.get_samples()
    if not samples:
        print("❌ Test örnekleri bulunamadı.")
        return None
    print(f"   📊 {len(samples)} meeting seçildi")

    print("\n🧠 [Adım 2/3] Diarization modeli yükleniyor...")
    from pyannote.audio import Pipeline
    from src.config import DIARIZATION_CONFIG_PATH, HF_TOKEN, DEVICE
    import torch
    try:
        if os.path.exists(DIARIZATION_CONFIG_PATH):
            # Türkçe karakter ('ü') ve boşluk ('GitHub Desktop') içeren Windows
            # yolları pyannote'un HuggingFace repo-id doğrulayıcısını kırar. Canlı
            # sistemle aynı çözümü kullan: config'i Windows 8.3 kısa yola çevirip
            # ASCII-güvenli geçici dosyadan yükle.
            from src.core.diarization_config import prepare_runtime_config
            runtime_config = prepare_runtime_config(DIARIZATION_CONFIG_PATH)
            diarizer = Pipeline.from_pretrained(runtime_config)
        else:
            diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
        if DEVICE == "cuda":
            diarizer.to(torch.device("cuda"))
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"\n🔄 [Adım 3/3] {len(samples)} meeting test ediliyor...\n")
    evaluator = DiarizationEvaluator()

    for i, sample in enumerate(samples, 1):
        meeting_id = sample["meeting_id"]
        print(f"   [{i}/{len(samples)}] {meeting_id} ({sample['duration'] / 60:.1f} dk)...", end=" ", flush=True)
        start_time = time.time()
        try:
            import soundfile as sf
            import torch
            if max_minutes:
                info = sf.info(sample["audio_path"])
                sr = info.samplerate
                frames_to_read = int(sr * max_minutes * 60)
                audio_data, sample_rate = sf.read(sample["audio_path"], dtype="float32", frames=frames_to_read)
                actual_duration = len(audio_data) / sample_rate
            else:
                audio_data, sample_rate = sf.read(sample["audio_path"], dtype="float32")
                actual_duration = sample["duration"]

            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            waveform = torch.from_numpy(audio_data).unsqueeze(0)

            pipeline_output = diarizer({"waveform": waveform, "sample_rate": sample_rate})
            diarization = getattr(pipeline_output, "speaker_diarization", pipeline_output)

            hyp_intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                hyp_intervals.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            ref_intervals = sample["annotations"]
            if max_minutes:
                filtered_refs = []
                for ann in ref_intervals:
                    if ann["start"] >= actual_duration:
                        continue
                    new_ann = dict(ann)
                    if new_ann["end"] > actual_duration:
                        new_ann["end"] = actual_duration
                    filtered_refs.append(new_ann)
                ref_intervals = filtered_refs

            elapsed = time.time() - start_time
            res = evaluator.evaluate(meeting_id=meeting_id, reference_intervals=ref_intervals,
                                     hypothesis_intervals=hyp_intervals, duration=actual_duration)
            if res:
                print(f"DER: {res.der * 100:.1f}% ({elapsed:.1f}s)")
            else:
                print(f"⚠️ Değerlendirme yapılamadı ({elapsed:.1f}s)")
        except Exception as e:
            print(f"❌ Hata: {e}")

    evaluator.print_report()
    return evaluator.report


def run_diarization_benchmark_aiworker(max_minutes=None):
    """AIWorker canlı-stream simülasyonu ile DER testi."""
    print("\n" + "=" * 70)
    print("🧪  BENCHMARK — Diarization Doğruluk Testi (AIWorker Modu)")
    print("=" * 70)

    print("\n📦 [Adım 1/3] Veri seti hazırlanıyor...")
    dm = AmiDiarizationManager()
    if not dm.download():
        print("❌ Veri seti indirilemedi. Benchmark iptal.")
        return None
    samples = dm.get_samples()
    if not samples:
        print("❌ Test örnekleri bulunamadı.")
        return None
    print(f"   📊 {len(samples)} meeting seçildi")

    print("\n🧠 [Adım 2/3] AIWorker ve VADEngine yükleniyor...")
    from src.audio.vad import VADEngine
    from src.core.ai_worker import AIWorker
    from src.config import (FRAME_DURATION_MS, SILENCE_LIMIT, SHORT_SILENCE_LIMIT,
                            SOFT_CHUNK_DURATION_MS, MAX_CHUNK_DURATION_MS)
    import soundfile as sf

    vad_engine = VADEngine()
    ai_worker = AIWorker(rate=16000, channels=1)
    if not ai_worker.load_models():
        print("❌ AIWorker yüklenemedi. Benchmark iptal.")
        return None
    warmup_s = ai_worker.speaker_tracker.warmup_ms / 1000.0

    print(f"\n🔄 [Adım 3/3] {len(samples)} meeting test ediliyor...\n")
    evaluator = DiarizationEvaluator()

    for i, sample in enumerate(samples, 1):
        meeting_id = sample["meeting_id"]
        print(f"   [{i}/{len(samples)}] {meeting_id} işleniyor...", flush=True)
        start_time = time.time()
        try:
            if max_minutes:
                info = sf.info(sample["audio_path"])
                frames_to_read = int(info.samplerate * max_minutes * 60)
                audio_data, sample_rate = sf.read(sample["audio_path"], dtype="int16", frames=frames_to_read)
                actual_duration = len(audio_data) / sample_rate
            else:
                audio_data, sample_rate = sf.read(sample["audio_path"], dtype="int16")
                actual_duration = sample["duration"]

            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1).astype(np.int16)
            audio_bytes = audio_data.tobytes()

            ai_worker.speaker_tracker.reset()
            bytes_per_frame = int(sample_rate * (FRAME_DURATION_MS / 1000.0) * 2)
            chunk_buffer = []
            silence_counter = 0
            has_spoken = False
            hyp_intervals = []
            global_time_s = 0.0
            chunk_start_s = 0.0

            for offset in range(0, len(audio_bytes), bytes_per_frame):
                frame_bytes = audio_bytes[offset:offset + bytes_per_frame]
                if len(frame_bytes) < bytes_per_frame:
                    break
                is_speech, conf = vad_engine.check_speech(frame_bytes, sample_rate, 1)
                if is_speech:
                    chunk_buffer.append(frame_bytes)
                    silence_counter = 0
                    if not has_spoken:
                        has_spoken = True
                        chunk_start_s = global_time_s
                else:
                    silence_bytes = b'\x00' * len(frame_bytes)
                    if has_spoken:
                        chunk_buffer.append(silence_bytes)
                        silence_counter += 1
                global_time_s += (FRAME_DURATION_MS / 1000.0)
                current_duration_ms = len(chunk_buffer) * FRAME_DURATION_MS
                active_silence_limit = SHORT_SILENCE_LIMIT if current_duration_ms > SOFT_CHUNK_DURATION_MS else SILENCE_LIMIT
                if has_spoken and (silence_counter > active_silence_limit or current_duration_ms >= MAX_CHUNK_DURATION_MS):
                    chunk_bytes_to_send = b''.join(chunk_buffer)
                    output = ai_worker.process_chunk(chunk_bytes_to_send)
                    results = output.get("results") if isinstance(output, dict) else output
                    if results:
                        for r in results:
                            speaker = r["speaker"]
                            if "Calibrating" not in speaker:
                                global_start = chunk_start_s + r["start"]
                                global_end = chunk_start_s + r["end"]
                                if global_end > warmup_s:
                                    hyp_intervals.append({
                                        "start": max(warmup_s, global_start),
                                        "end": global_end, "speaker": speaker,
                                    })
                    chunk_buffer = []
                    silence_counter = 0
                    has_spoken = False
                    # Silero VAD durumunu chunk'lar arası sıfırla (state sızıntısını önle)
                    if hasattr(vad_engine.silero_model, "reset_states"):
                        vad_engine.silero_model.reset_states()

            ref_intervals = sample["annotations"]
            filtered_refs = []
            for ann in ref_intervals:
                if max_minutes and ann["start"] >= actual_duration:
                    continue
                if ann["end"] <= warmup_s:
                    continue
                new_ann = dict(ann)
                new_ann["start"] = max(warmup_s, new_ann["start"])
                if max_minutes:
                    new_ann["end"] = min(actual_duration, new_ann["end"])
                if new_ann["start"] < new_ann["end"]:
                    filtered_refs.append(new_ann)

            elapsed = time.time() - start_time
            res = evaluator.evaluate(meeting_id=meeting_id, reference_intervals=filtered_refs,
                                     hypothesis_intervals=hyp_intervals,
                                     duration=max(0.1, actual_duration - warmup_s))
            if res:
                print(f"      DER: {res.der * 100:.1f}% ({elapsed:.1f}s)")
            else:
                print(f"      ⚠️ Değerlendirme yapılamadı ({elapsed:.1f}s)")
        except Exception as e:
            print(f"      ❌ Hata: {e}")

    evaluator.print_report()
    return evaluator.report


def main():
    parser = argparse.ArgumentParser(description="AMI Diarization (DER) Benchmark Aracı")
    parser.add_argument("--mode", type=str, choices=["raw", "aiworker"], default="raw",
                        help="raw (saf pyannote) veya aiworker (canlı simülasyon) (varsayılan: raw)")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="İşlenecek maksimum ses süresi (dakika) (None: tamamı)")
    args = parser.parse_args()

    if args.mode == "aiworker":
        run_diarization_benchmark_aiworker(max_minutes=args.max_minutes)
    else:
        run_diarization_benchmark_raw(max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
