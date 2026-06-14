"""Dataset manager'ları için ortak altyapı.

- DATASETS_ROOT: tüm veri setlerinin kök dizini (<proje kökü>/datasets), gitignore'lı.
- _download_with_progress / download_targz: ortak indirme + arşiv açma yardımcıları.
- soundfile import'u TEMBELdir: sadece ses veri setleri (LibriSpeech/AMI) gerektirir;
  metin/FLORES yolunda ses kütüphanesi şart olmasın.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

# <proje kökü>/datasets  (base.py: tests/dataset_managers/base.py -> parents[2] = kök)
DATASETS_ROOT = str(Path(__file__).resolve().parents[2] / "datasets")


def _download_with_progress(url, dest_path):
    """İlerleme çubuğu ile dosya indirir."""
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    block_size = 1024 * 1024  # 1 MB

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        while True:
            block = response.read(block_size)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   İlerleme: {mb_done:.1f} / {mb_total:.1f} MB ({percent:.1f}%)",
                      end="", flush=True)
    print()  # Yeni satır


def download_targz(url, archive_path, extract_dir, label="", size_hint="") -> bool:
    """Bir .tar.gz indirir, `extract_dir` içine açar, arşivi siler.

    Returns:
        bool: Başarılıysa True.
    """
    os.makedirs(extract_dir, exist_ok=True)

    if not os.path.exists(archive_path):
        print(f"📥 {label} indiriliyor...")
        print(f"   Kaynak: {url}")
        print(f"   Hedef:  {archive_path}")
        if size_hint:
            print(f"   ⚠️  {size_hint}\n")
        try:
            _download_with_progress(url, archive_path)
        except Exception as e:
            print(f"❌ İndirme başarısız: {e}")
            return False
    else:
        print("📦 Arşiv zaten indirilmiş, açılıyor...")

    print("📂 Arşiv açılıyor...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"✅ {label} başarıyla hazırlandı.")
    except Exception as e:
        print(f"❌ Arşiv açma başarısız: {e}")
        return False

    try:
        os.remove(archive_path)
        print("🗑️  Arşiv dosyası silindi (disk tasarrufu).")
    except Exception:
        pass
    return True
