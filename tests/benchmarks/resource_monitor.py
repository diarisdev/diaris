"""
Kaynak monitörü — bir koşu boyunca tepe CPU / RAM / VRAM kullanımını örnekler.

Arka planda bir thread sabit aralıkla örnek alır ve peak'leri saklar.
- CPU%  : süreç (process) CPU yüzdesi (çok çekirdekte 100%'ü aşabilir — toplam).
- RAM   : sürecin RSS belleği (MB).
- VRAM  : pynvml ile TÜM GPU kullanımı (MB) — birincil, gerçek sistemi yansıtır
          (Whisper/CTranslate2 torch-dışı tahsisleri de dahil). Ek olarak
          torch.cuda.max_memory_allocated() PyTorch-özelinde döküm verir.

Bağımlılıklar opsiyonel; yoksa o metrik None döner (çökmeden).
    pip install psutil nvidia-ml-py
"""

from __future__ import annotations

import threading
import time


class ResourceMonitor:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

        # Peak değerler
        self.peak_cpu_pct = 0.0
        self.peak_ram_mb = 0.0
        self.peak_vram_mb = 0.0          # pynvml (tüm GPU)
        self.peak_torch_vram_mb = 0.0    # torch.cuda (yalnızca PyTorch tahsisi)
        self.samples = 0

        # --- Opsiyonel bağımlılıklar ---
        self._psutil = None
        self._proc = None
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)  # ilk çağrı 0 döner; baseline kur
        except Exception:
            pass

        self._nvml = None
        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

        self._torch = None
        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # -- context manager --
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if self._proc is not None:
                try:
                    cpu = self._proc.cpu_percent(None)
                    self.peak_cpu_pct = max(self.peak_cpu_pct, cpu)
                    rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                    self.peak_ram_mb = max(self.peak_ram_mb, rss_mb)
                except Exception:
                    pass
            if self._nvml is not None:
                try:
                    info = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self.peak_vram_mb = max(self.peak_vram_mb, info.used / (1024 * 1024))
                except Exception:
                    pass
            self.samples += 1
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        # torch peak'i thread örneklemesi değil, kendi sayacından oku
        if self._torch is not None:
            try:
                self.peak_torch_vram_mb = self._torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self.result()

    def result(self) -> dict:
        return {
            "peak_cpu_pct": self.peak_cpu_pct if self._proc else None,
            "peak_ram_mb": self.peak_ram_mb if self._proc else None,
            "peak_vram_mb": self.peak_vram_mb if self._nvml else None,
            "peak_torch_vram_mb": self.peak_torch_vram_mb if self._torch else None,
            "samples": self.samples,
            "have_psutil": self._proc is not None,
            "have_pynvml": self._nvml is not None,
            "have_torch_cuda": self._torch is not None,
        }
