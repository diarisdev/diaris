"""LibriSpeech ASR benchmark — transkripsiyon doğruluğu (WER/CER).

Veri setindeki ses dosyalarını AIWorker (veya ham Whisper) üzerinden geçirir,
gerçek transkripsiyonla karşılaştırır ve doğruluk raporu üretir.

Çalıştırma (proje kökünden):
    python -m tests.benchmarks.librispeech_asr --limit 20
    python -m tests.benchmarks.librispeech_asr --limit 50 --csv results.csv
    python -m tests.benchmarks.librispeech_asr --min-duration 5 --max-duration 10 --jiwer
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
import soundfile as sf

# Proje kökünü sys.path'e ekle (tests/benchmarks/librispeech_asr.py -> 3 yukarı)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# CUDA DLL'lerini (cublas/cudnn) model yüklemeden ÖNCE bulunur kıl (main.py gibi)
from src.config import configure_cuda_dll_paths
configure_cuda_dll_paths()

from tests.dataset_managers import DatasetManager
from tests.metrics import TranscriptionEvaluator


def run_benchmark(limit=20, min_duration=2.0, max_duration=15.0, use_jiwer=False,
                  csv_path=None, mode="aiworker"):
    """LibriSpeech transkripsiyon benchmark'ı."""
    print("\n" + "=" * 70)
    print("🧪  BENCHMARK BAŞLATILIYOR — Transkripsiyon Doğruluk Testi")
    print("=" * 70)

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

    print(f"\n🧠 [Adım 2/4] Transkripsiyon modeli yükleniyor ({mode} modu)...")
    if mode == "aiworker":
        from src.core.ai_worker import AIWorker
        ai_worker = AIWorker(rate=16000, channels=1)
        if not ai_worker.load_models():
            print("❌ AI modelleri yüklenemedi. Benchmark iptal.")
            return None
    else:
        from faster_whisper import WhisperModel
        from src.config import WHISPER_PATH, DEVICE, COMPUTE_TYPE
        ai_worker = None
        transcriber = WhisperModel(WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)

    print(f"\n🔄 [Adım 3/4] {len(samples)} örnek test ediliyor...\n")
    evaluator = TranscriptionEvaluator(use_jiwer=use_jiwer)

    # RTF (Real-Time Factor) için: toplam inference süresi / toplam ses süresi.
    # Model YÜKLEME süresi dahil DEĞİL — sadece process_chunk/transcribe ölçülür.
    total_compute = 0.0
    timed_audio = 0.0

    for i, sample in enumerate(samples, 1):
        fname = os.path.basename(sample["audio_path"])
        print(f"   [{i}/{len(samples)}] {fname} ({sample['duration']:.1f}s)...", end=" ", flush=True)
        start_time = time.time()
        try:
            if mode == "aiworker":
                audio_data, sr = sf.read(sample["audio_path"], dtype="int16")
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]
                output = ai_worker.process_chunk(audio_data.tobytes())
                results = output.get("results") if isinstance(output, dict) else output
                hypothesis = " ".join(r["text"].strip() for r in results) if results else None
            else:
                segments, _ = transcriber.transcribe(sample["audio_path"], language="en")
                hypothesis = " ".join(s.text.strip() for s in segments)

            elapsed = time.time() - start_time
            total_compute += elapsed
            timed_audio += sample["duration"]
            if hypothesis:
                eval_result = evaluator.evaluate(
                    reference=sample["transcript"], hypothesis=hypothesis,
                    audio_path=sample["audio_path"], speaker_id=sample["speaker_id"],
                    duration=sample["duration"],
                )
                print(f"WER: {eval_result.wer * 100:.1f}% ({elapsed:.1f}s)")
            else:
                evaluator.record_failure(audio_path=sample["audio_path"], reason="Transkripsiyon üretilemedi")
                print(f"⚠️  Sonuç yok ({elapsed:.1f}s)")
        except Exception as e:
            evaluator.record_failure(audio_path=sample["audio_path"], reason=str(e))
            print(f"❌ Hata: {e}")

    print("\n📊 [Adım 4/4] Rapor oluşturuluyor...")
    evaluator.print_report()

    # --- Performans / RTF ---
    try:
        from src.config import DEVICE
        device = DEVICE
    except Exception:
        device = "?"
    rtf = (total_compute / timed_audio) if timed_audio > 0 else 0.0
    print("\n⏱️  Performans (RTF):")
    print(f"   Cihaz:                {device}")
    print(f"   Toplam İşlem Süresi:  {total_compute:.1f}s")
    print(f"   Toplam Ses Süresi:    {timed_audio:.1f}s")
    print(f"   RTF:                  {rtf:.3f}  (1.0 = gerçek zamanlı, <1 daha hızlı)")
    if rtf > 0:
        print(f"   Hız:                  gerçek zamanın {1.0 / rtf:.1f}× katı")
    # Rapora yazılabilmesi için report nesnesine de iliştir
    try:
        evaluator.report.total_compute_time = total_compute
        evaluator.report.rtf = rtf
        evaluator.report.device = device
    except Exception:
        pass

    if csv_path:
        if not os.path.isabs(csv_path):
            from src.config import OUTPUT_DIR
            csv_path = os.path.join(OUTPUT_DIR, csv_path)
        evaluator.export_csv(csv_path)
    return evaluator.report


def main():
    parser = argparse.ArgumentParser(
        description="LibriSpeech Transkripsiyon Benchmark Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python -m tests.benchmarks.librispeech_asr --limit 10
  python -m tests.benchmarks.librispeech_asr --limit 50 --csv results.csv
  python -m tests.benchmarks.librispeech_asr --min-duration 5 --max-duration 10 --jiwer
        """,
    )
    parser.add_argument("--limit", type=int, default=20, help="Test edilecek örnek sayısı (varsayılan: 20)")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum ses süresi, sn (varsayılan: 2.0)")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Maksimum ses süresi, sn (varsayılan: 15.0)")
    parser.add_argument("--csv", type=str, default=None, help="Sonuçları CSV dosyasına kaydet")
    parser.add_argument("--jiwer", action="store_true", help="jiwer ile hassas WER hesabı")
    parser.add_argument("--transcription-mode", type=str, choices=["raw", "aiworker"], default="aiworker",
                        help="Kullanılacak mod (varsayılan: aiworker)")
    parser.add_argument("--download-only", action="store_true", help="Sadece veri setini indir, test çalıştırma")
    args = parser.parse_args()

    if args.download_only:
        dm = DatasetManager()
        dm.download()
        print(f"\n📊 Veri Seti Özeti:")
        for key, val in dm.get_summary().items():
            print(f"   {key}: {val}")
        return

    run_benchmark(
        limit=args.limit, min_duration=args.min_duration, max_duration=args.max_duration,
        use_jiwer=args.jiwer, csv_path=args.csv, mode=args.transcription_mode,
    )


if __name__ == "__main__":
    main()
