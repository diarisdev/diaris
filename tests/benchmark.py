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

from tests.dataset_manager import DatasetManager
from tests.evaluator import TranscriptionEvaluator


def run_benchmark(limit=20, min_duration=2.0, max_duration=15.0, use_jiwer=False, csv_path=None):
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

    # --- 2. AI Worker ---
    print("\n🧠 [Adım 2/4] AI modelleri yükleniyor...")
    from src.core.ai_worker import AIWorker

    # LibriSpeech 16kHz mono
    ai_worker = AIWorker(rate=16000, channels=1)
    if not ai_worker.load_models():
        print("❌ AI modelleri yüklenemedi. Benchmark iptal.")
        return None

    # --- 3. Test döngüsü ---
    print(f"\n🔄 [Adım 3/4] {len(samples)} örnek test ediliyor...\n")
    evaluator = TranscriptionEvaluator(use_jiwer=use_jiwer)

    for i, sample in enumerate(samples, 1):
        fname = os.path.basename(sample["audio_path"])
        print(f"   [{i}/{len(samples)}] {fname} ({sample['duration']:.1f}s)...", end=" ", flush=True)

        start_time = time.time()

        try:
            # Ses dosyasını oku ve int16 bytes'a çevir
            audio_data, sr = sf.read(sample["audio_path"], dtype="int16")

            # Mono olduğundan emin ol
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            chunk_bytes = audio_data.tobytes()

            # AIWorker ile işle
            results = ai_worker.process_chunk(chunk_bytes)

            elapsed = time.time() - start_time

            if results:
                # Tüm segmentleri birleştir
                hypothesis = " ".join(r["text"].strip() for r in results)
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

    run_benchmark(
        limit=args.limit,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        use_jiwer=args.jiwer,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
