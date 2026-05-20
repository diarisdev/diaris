"""
Benchmark Çalıştırıcısı.

Veri setindeki ses dosyalarını AIWorker üzerinden geçirir,
gerçek transkripsiyon ile karşılaştırır ve doğruluk raporu üretir.

Kullanım:
    python -m tests.benchmark --limit 20

Veya Python'dan:
    from tests.benchmark import run_benchmark
    report = run_benchmark(limit=20)
"""

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*std\\(\\).*degrees of freedom.*")

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.dataset_manager import DatasetManager, AmiDiarizationManager
from tests.evaluator import TranscriptionEvaluator, DiarizationEvaluator


def run_benchmark(limit=20, min_duration=2.0, max_duration=15.0, use_jiwer=False, csv_path=None, mode="aiworker"):
    """
    Ana benchmark fonksiyonu.

    1. Veri setini kontrol eder / indirir
    2. AIWorker'ı yükler
    3. Her ses dosyasını işler
    4. Sonuçları değerlendirir ve rapor üretir

    Args:
        limit: Test edilecek örnek sayısı
        min_duration: Minimum ses süresi (saniye)
        max_duration: Maksimum ses süresi (saniye)
        use_jiwer: jiwer kütüphanesi kullanılsın mı
        csv_path: CSV rapor dosyası yolu (isteğe bağlı)

    Returns:
        BenchmarkReport: Toplu sonuçlar
    """
    print("\n" + "=" * 70)
    print("🧪  BENCHMARK BAŞLATILIYOR — Transkripsiyon Doğruluk Testi")
    print("=" * 70)

    # --- 1. Veri seti ---
    print("\n📦 [Adım 1/4] Veri seti hazırlanıyor...")
    dm = DatasetManager()

    if not dm.is_downloaded():
        print("   Veri seti bulunamadı, indiriliyor...")
        if not dm.download():
            print("❌ Veri seti indirilemedi. Benchmark iptal.")
            return None
    else:
        print("   ✅ Veri seti mevcut.")

    samples = dm.get_samples(limit=limit, min_duration=min_duration, max_duration=max_duration)
    if not samples:
        print("❌ Test örnekleri bulunamadı.")
        return None

    total_duration = sum(s["duration"] for s in samples)
    print(f"   📊 {len(samples)} örnek seçildi (toplam {total_duration / 60:.1f} dakika ses)")

    # --- 2. AI Modelleri ---
    print(f"\n🧠 [Adım 2/4] Transkripsiyon modeli yükleniyor ({mode} modu)...")
    if mode == "aiworker":
        from src.core.ai_worker import AIWorker
        # LibriSpeech 16kHz mono
        ai_worker = AIWorker(rate=16000, channels=1)
        if not ai_worker.load_models():
            print("❌ AI modelleri yüklenemedi. Benchmark iptal.")
            return None
    else:
        from faster_whisper import WhisperModel
        from src.config import WHISPER_PATH, DEVICE, COMPUTE_TYPE
        ai_worker = None
        transcriber = WhisperModel(WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)

    # --- 3. Test döngüsü ---
    print(f"\n🔄 [Adım 3/4] {len(samples)} örnek test ediliyor...\n")
    evaluator = TranscriptionEvaluator(use_jiwer=use_jiwer)

    for i, sample in enumerate(samples, 1):
        fname = os.path.basename(sample["audio_path"])
        print(f"   [{i}/{len(samples)}] {fname} ({sample['duration']:.1f}s)...", end=" ", flush=True)

        start_time = time.time()

        try:
            if mode == "aiworker":
                # Ses dosyasını oku ve int16 bytes'a çevir
                audio_data, sr = sf.read(sample["audio_path"], dtype="int16")
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]
                chunk_bytes = audio_data.tobytes()
                
                # AIWorker ile işle
                results = ai_worker.process_chunk(chunk_bytes)
                if results:
                    hypothesis = " ".join(r["text"].strip() for r in results)
                else:
                    hypothesis = None
            else:
                # Raw model ile işle
                segments, _ = transcriber.transcribe(sample["audio_path"], language="en")
                hypothesis = " ".join(s.text.strip() for s in segments)

            elapsed = time.time() - start_time

            if hypothesis:
                eval_result = evaluator.evaluate(
                    reference=sample["transcript"],
                    hypothesis=hypothesis,
                    audio_path=sample["audio_path"],
                    speaker_id=sample["speaker_id"],
                    duration=sample["duration"],
                )
                print(f"WER: {eval_result.wer * 100:.1f}% ({elapsed:.1f}s)")
            else:
                evaluator.record_failure(
                    audio_path=sample["audio_path"],
                    reason="Transkripsiyon üretilemedi"
                )
                print(f"⚠️  Sonuç yok ({elapsed:.1f}s)")

        except Exception as e:
            evaluator.record_failure(
                audio_path=sample["audio_path"],
                reason=str(e)
            )
            print(f"❌ Hata: {e}")

    # --- 4. Rapor ---
    print("\n📊 [Adım 4/4] Rapor oluşturuluyor...")
    evaluator.print_report()

    if csv_path:
        if not os.path.isabs(csv_path):
            from src.config import OUTPUT_DIR
            csv_path = os.path.join(OUTPUT_DIR, csv_path)
        evaluator.export_csv(csv_path)

    return evaluator.report


def run_diarization_benchmark_raw(max_minutes=None):
    """
    Diarization benchmark'ı (AMI Corpus ile).
    Pyannote modelinin doğruluğunu DER (Diarization Error Rate) üzerinden test eder.
    """
    print("\n" + "=" * 70)
    print("🧪  BENCHMARK BAŞLATILIYOR — Diarization Doğruluk Testi (Saf Model Modu)")
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
            diarizer = Pipeline.from_pretrained(DIARIZATION_CONFIG_PATH)
        else:
            diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
        
        if DEVICE == "cuda":
            diarizer.to(torch.device("cuda"))
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return None

    print(f"\n🔄 [Adım 3/3] {len(samples)} meeting test ediliyor...\n")
    evaluator = DiarizationEvaluator()

    for i, sample in enumerate(samples, 1):
        meeting_id = sample["meeting_id"]
        print(f"   [{i}/{len(samples)}] {meeting_id} ({sample['duration'] / 60:.1f} dk)...", end=" ", flush=True)
        start_time = time.time()

        try:
            # Model inference (tüm dosya)
            import soundfile as sf
            import torch
            
            # Read all or max_minutes
            if max_minutes:
                frames_to_read = int(16000 * max_minutes * 60) # Assume 16kHz but sf will tell us
                # First get sample rate
                info = sf.info(sample["audio_path"])
                sr = info.samplerate
                frames_to_read = int(sr * max_minutes * 60)
                audio_data, sample_rate = sf.read(sample["audio_path"], dtype="float32", frames=frames_to_read)
                actual_duration = len(audio_data) / sample_rate
            else:
                audio_data, sample_rate = sf.read(sample["audio_path"], dtype="float32")
                actual_duration = sample["duration"]
                
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1) # Mono'ya çevir
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
            
            pipeline_output = diarizer({"waveform": waveform, "sample_rate": sample_rate})
            if hasattr(pipeline_output, "speaker_diarization"):
                diarization = pipeline_output.speaker_diarization
            else:
                diarization = pipeline_output
            
            # Hypothesis interval'lara çevir
            hyp_intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                hyp_intervals.append({"start": turn.start, "end": turn.end, "speaker": speaker})

            # Reference interval'ları filtrele (truncate)
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
            
            # Değerlendir
            res = evaluator.evaluate(
                meeting_id=meeting_id,
                reference_intervals=ref_intervals,
                hypothesis_intervals=hyp_intervals,
                duration=actual_duration
            )
            
            if res:
                print(f"DER: {res.der * 100:.1f}% ({elapsed:.1f}s)")
            else:
                print(f"⚠️ Değerlendirme yapılamadı ({elapsed:.1f}s)")

        except Exception as e:
            print(f"❌ Hata: {e}")

    # Rapor
    evaluator.print_report()
    return evaluator.report


def run_diarization_benchmark_aiworker(max_minutes=None):
    """
    Diarization benchmark'ı (AMI Corpus ile).
    AIWorker'in canlı stream simülasyonunu yapar ve doğruluğunu test eder.
    """
    print("\n" + "=" * 70)
    print("🧪  BENCHMARK BAŞLATILIYOR — Diarization Doğruluk Testi (AIWorker Modu)")
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
    from src.config import FRAME_DURATION_MS, SILENCE_LIMIT, SHORT_SILENCE_LIMIT, SOFT_CHUNK_DURATION_MS, MAX_CHUNK_DURATION_MS
    import soundfile as sf
    import numpy as np
    
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
            # Sesi yükle ve int16'ya çevir (VAD ve AIWorker için)
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
            
            # Tracker durumunu sıfırla
            ai_worker.speaker_tracker.reset()
            
            # Stream simülasyonu değişkenleri
            bytes_per_frame = int(sample_rate * (FRAME_DURATION_MS / 1000.0) * 2)
            
            chunk_buffer = []
            silence_counter = 0
            has_spoken = False
            
            hyp_intervals = []
            global_time_s = 0.0
            chunk_start_s = 0.0
            
            # Ses dosyasını frame frame işle
            for offset in range(0, len(audio_bytes), bytes_per_frame):
                frame_bytes = audio_bytes[offset:offset+bytes_per_frame]
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
                    # AIWorker'a gönder
                    chunk_bytes_to_send = b''.join(chunk_buffer)
                    results = ai_worker.process_chunk(chunk_bytes_to_send)
                    
                    if results:
                        for r in results:
                            speaker = r["speaker"]
                            # Kalibrasyon uyarısını yoksay
                            if "Calibrating" not in speaker:
                                global_start = chunk_start_s + r["start"]
                                global_end = chunk_start_s + r["end"]
                                
                                # Warm-up süresi öncesini yoksay
                                if global_end > warmup_s:
                                    hyp_intervals.append({
                                        "start": max(warmup_s, global_start),
                                        "end": global_end,
                                        "speaker": speaker
                                    })
                    
                    # Bufferı sıfırla
                    chunk_buffer = []
                    silence_counter = 0
                    has_spoken = False

            # Referansları warm-up'ı yoksayacak şekilde filtrele
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
            
            # Değerlendir
            res = evaluator.evaluate(
                meeting_id=meeting_id,
                reference_intervals=filtered_refs,
                hypothesis_intervals=hyp_intervals,
                duration=max(0.1, actual_duration - warmup_s)
            )
            
            if res:
                print(f"      DER: {res.der * 100:.1f}% ({elapsed:.1f}s)")
            else:
                print(f"      ⚠️ Değerlendirme yapılamadı ({elapsed:.1f}s)")

        except Exception as e:
            print(f"      ❌ Hata: {e}")

    # Rapor
    evaluator.print_report()
    return evaluator.report


def main():
    """CLI giriş noktası."""
    parser = argparse.ArgumentParser(
        description="Audio-process Transkripsiyon Benchmark Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python -m tests.benchmark --limit 10
  python -m tests.benchmark --limit 50 --csv results.csv
  python -m tests.benchmark --min-duration 5 --max-duration 10 --jiwer
        """
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Test edilecek örnek sayısı (varsayılan: 20)"
    )
    parser.add_argument(
        "--min-duration", type=float, default=2.0,
        help="Minimum ses süresi, saniye (varsayılan: 2.0)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=15.0,
        help="Maksimum ses süresi, saniye (varsayılan: 15.0)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Sonuçları CSV dosyasına kaydet"
    )
    parser.add_argument(
        "--jiwer", action="store_true",
        help="jiwer kütüphanesi kullan (daha hassas WER hesaplama)"
    )
    parser.add_argument(
        "--task", type=str, choices=["transcription", "diarization"], default="transcription",
        help="Çalıştırılacak benchmark görevi (varsayılan: transcription)"
    )
    parser.add_argument(
        "--transcription-mode", type=str, choices=["raw", "aiworker"], default="aiworker",
        help="Transcription testinde kullanılacak mod (varsayılan: aiworker)"
    )
    parser.add_argument(
        "--diarization-mode", type=str, choices=["raw", "aiworker"], default="raw",
        help="Diarization testinde kullanılacak mod (varsayılan: raw)"
    )
    parser.add_argument(
        "--diarization-max-minutes", type=float, default=None,
        help="Diarization testinde işlenecek maksimum ses süresi (dakika) (None: tamamı)"
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Sadece veri setini indir, test çalıştırma"
    )

    args = parser.parse_args()

    if args.download_only:
        dm = DatasetManager()
        dm.download()
        summary = dm.get_summary()
        print(f"\n📊 Veri Seti Özeti:")
        for key, val in summary.items():
            print(f"   {key}: {val}")
        return

    if args.task == "diarization":
        if args.diarization_mode == "aiworker":
            run_diarization_benchmark_aiworker(max_minutes=args.diarization_max_minutes)
        else:
            run_diarization_benchmark_raw(max_minutes=args.diarization_max_minutes)
    else:
        run_benchmark(
            limit=args.limit,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            use_jiwer=args.jiwer,
            csv_path=args.csv,
            mode=args.transcription_mode,
        )


if __name__ == "__main__":
    main()
