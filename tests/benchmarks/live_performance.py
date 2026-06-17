"""Canlı performans benchmark'ı — gecikme + RTF + tepe CPU/RAM/VRAM.

GERÇEK mikrofon/loopback yolunu (src.pipeline.run) çalıştırır; ses oynatırken
(ör. bir YouTube videosu) belirtilen süre boyunca metrikleri toplar ve tez
tablosunu basar.

Ölçülen metrikler:
  - Ortalama uçtan uca gecikme : chunk KAPANIŞINDAN final (diarize+çeviri)
                                 altyazının hazır olduğu ana kadar.
  - Transkripsiyon gecikmesi   : chunk kapanışından İLK (anlık) transkripsiyona.
  - RTF                        : toplam işlem süresi / toplam ses süresi.
  - Tepe CPU / RAM / VRAM      : ResourceMonitor (psutil + pynvml).

Çalıştırma (proje kökünden, ses çalarken):
    python -m tests.benchmarks.live_performance --seconds 60
    python -m tests.benchmarks.live_performance --seconds 90 --source en --target tr
    python -m tests.benchmarks.live_performance --seconds 60 --target en   # çeviri yok
    python -m tests.benchmarks.live_performance --device 7 --seconds 60
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import argparse
import threading
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import configure_cuda_dll_paths
configure_cuda_dll_paths()

from src.pipeline import run
from tests.benchmarks.resource_monitor import ResourceMonitor


class _Collector:
    """Pipeline callback'lerinden gecikme/RTF verisi toplar (thread-güvenli)."""

    def __init__(self, skip_first=1):
        self.lock = threading.Lock()
        self.skip_first = skip_first
        self.transcription_latencies = []   # sn (chunk kapanışı → anlık transkripsiyon)
        self.end_to_end_latencies = []      # sn (chunk kapanışı → final diarize+çeviri)
        self.chunk_durations = []           # sn (işlenen ses)
        self.total_stt_ms = 0.0
        self.total_diar_ms = 0.0
        self._trans_seen = 0
        self._spk_seen = 0

    def on_transcription(self, event):
        if not isinstance(event, dict) or event.get("type") != "final":
            return
        cap = event.get("captured_at")
        emitted = event.get("emitted_at") or time.time()
        with self.lock:
            self._trans_seen += 1
            if event.get("stt_ms") is not None:
                self.total_stt_ms += event["stt_ms"]
            if cap and self._trans_seen > self.skip_first:
                self.transcription_latencies.append(emitted - cap)

    def on_speaker_update(self, event):
        if not isinstance(event, dict):
            return
        cap = event.get("captured_at")
        emitted = event.get("emitted_at") or time.time()
        with self.lock:
            self._spk_seen += 1
            if event.get("diar_ms") is not None:
                self.total_diar_ms += event["diar_ms"]
            if event.get("chunk_duration_ms"):
                self.chunk_durations.append(event["chunk_duration_ms"] / 1000.0)
            if cap and self._spk_seen > self.skip_first:
                self.end_to_end_latencies.append(emitted - cap)


def _fmt(v, unit, scale=1.0, nd=2):
    return f"{v * scale:.{nd}f} {unit}" if v is not None else "—"


def main():
    parser = argparse.ArgumentParser(description="Canlı performans benchmark'ı")
    parser.add_argument("--seconds", type=float, default=60.0, help="Ölçüm süresi (varsayılan: 60)")
    parser.add_argument("--device", type=int, default=None, help="PyAudio cihaz index'i (varsayılan: otomatik loopback)")
    parser.add_argument("--source", type=str, default="en", help="Kaynak dil (varsayılan: en)")
    parser.add_argument("--target", type=str, default="tr", help="Hedef dil; kaynakla aynıysa çeviri yok (varsayılan: tr)")
    parser.add_argument("--skip-first", type=int, default=1, help="Gecikme istatistiğinde atlanacak ilk chunk sayısı (model ısınması)")
    parser.add_argument("--interval", type=float, default=0.2, help="Kaynak örnekleme aralığı, sn")
    args = parser.parse_args()

    print("=" * 70)
    print("⏱️  CANLI PERFORMANS BENCHMARK'I")
    print("=" * 70)
    print(f"   Süre: {args.seconds:.0f}s | Dil: {args.source}->{args.target} | "
          f"Cihaz: {'oto' if args.device is None else args.device}")
    print("   NOT: Ölçüm boyunca sesi (ör. YouTube videosu) ÇALIYOR durumda tut.\n")

    collector = _Collector(skip_first=args.skip_first)
    stop_event = threading.Event()

    def get_lang_pair():
        return args.source, args.target

    monitor = ResourceMonitor(interval=args.interval)
    monitor.start()

    pipeline_thread = threading.Thread(
        target=run,
        kwargs={
            "stop_event": stop_event,
            "on_transcription": collector.on_transcription,
            "on_speaker_update": collector.on_speaker_update,
            "device_index": args.device,
            "get_lang_pair": get_lang_pair,
        },
        daemon=True,
    )
    pipeline_thread.start()

    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("\n[!] Erken durduruldu.")
    stop_event.set()
    pipeline_thread.join(timeout=40)
    res = monitor.stop()

    # --- İstatistikler ---
    with collector.lock:
        e2e = list(collector.end_to_end_latencies)
        trans = list(collector.transcription_latencies)
        chunks = list(collector.chunk_durations)
        total_stt_ms = collector.total_stt_ms
        total_diar_ms = collector.total_diar_ms

    total_audio_s = sum(chunks)
    total_compute_s = (total_stt_ms + total_diar_ms) / 1000.0
    rtf = (total_compute_s / total_audio_s) if total_audio_s > 0 else None

    avg_e2e = statistics.mean(e2e) if e2e else None
    med_e2e = statistics.median(e2e) if e2e else None
    first_trans_ms = (trans[0] * 1000.0) if trans else None
    avg_trans_ms = (statistics.mean(trans) * 1000.0) if trans else None
    mean_chunk = statistics.mean(chunks) if chunks else None

    print("\n" + "=" * 70)
    print("📊  PERFORMANS RAPORU")
    print("=" * 70)
    print(f"{'Ölçüt':<42}{'Değer'}")
    print("-" * 70)
    print(f"{'Ortalama uçtan uca gecikme':<42}{_fmt(avg_e2e, 's')}")
    print(f"{'  (medyan)':<42}{_fmt(med_e2e, 's')}")
    print(f"{'Transkripsiyon gecikmesi (ilk segment)':<42}{_fmt(first_trans_ms, 'ms', nd=0)}")
    print(f"{'  (ortalama transkripsiyon gecikmesi)':<42}{_fmt(avg_trans_ms, 'ms', nd=0)}")
    print(f"{'Gerçek zaman çarpanı (RTF)':<42}{(f'{rtf:.3f}' if rtf is not None else '—')}")
    print(f"{'Tepe CPU':<42}{_fmt(res['peak_cpu_pct'], '%', nd=0)}")
    print(f"{'Tepe RAM':<42}{_fmt(res['peak_ram_mb'], 'MB', nd=0)}")
    print(f"{'Tepe VRAM (tüm GPU, pynvml)':<42}{_fmt(res['peak_vram_mb'], 'MB', nd=0)}")
    print(f"{'  (PyTorch tahsisi, torch.cuda)':<42}{_fmt(res['peak_torch_vram_mb'], 'MB', nd=0)}")
    print("-" * 70)
    print(f"   Ölçülen chunk sayısı: {len(e2e)} | Ortalama chunk süresi: {_fmt(mean_chunk, 's')}")
    print(f"   Toplam işlenen ses: {total_audio_s:.1f}s | Toplam işlem: {total_compute_s:.1f}s")

    # Eksik bağımlılık uyarıları
    if not res["have_psutil"]:
        print("   ⚠️  psutil yok → CPU/RAM ölçülemedi:  pip install psutil")
    if not res["have_pynvml"]:
        print("   ⚠️  pynvml yok → GPU VRAM ölçülemedi:  pip install nvidia-ml-py")

    print("\n   Tanım: 'uçtan uca gecikme' = chunk kapanışından final (diarize+çeviri)")
    print("   çıktıya kadar geçen süre. Kullanıcının algıladığı toplam gecikmeye")
    print(f"   ortalama tampon beklemesi (~chunk süresi/2 ≈ {(mean_chunk or 0)/2:.1f}s) eklenir.")
    print("=" * 70)


if __name__ == "__main__":
    main()
