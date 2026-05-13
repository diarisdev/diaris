"""
Test Veri Seti Yöneticisi.

LibriSpeech test-clean alt kümesini indirir ve düzenler.
Her ses dosyası için ground-truth transkripsiyon bilgisini sağlar.

Desteklenen veri setleri:
  - LibriSpeech test-clean (İngilizce, temiz konuşma)

Kullanım:
    manager = DatasetManager()
    samples = manager.get_samples(limit=10)
    # -> [{"audio_path": "...", "transcript": "...", "speaker_id": "...", "duration": ...}, ...]
"""

import os
import tarfile
import urllib.request
import shutil
import soundfile as sf
from pathlib import Path


# --- Veri Seti Sabitleri ---
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_DIR_NAME = "LibriSpeech/test-clean"

# Varsayılan veri seti kök dizini (proje kökü/tests/data)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class DatasetManager:
    """
    LibriSpeech test-clean veri setini indiren, açan ve
    ses dosyası + transkripsiyon çiftlerini yöneten sınıf.
    """

    def __init__(self, data_dir=None):
        """
        Args:
            data_dir: Veri setinin indirileceği dizin.
                      None ise tests/data/ kullanılır.
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.archive_path = os.path.join(self.data_dir, "test-clean.tar.gz")
        self.dataset_root = os.path.join(self.data_dir, LIBRISPEECH_DIR_NAME)

    def is_downloaded(self):
        """Veri seti zaten indirilmiş ve açılmış mı kontrol eder."""
        return os.path.isdir(self.dataset_root)

    def download(self, force=False):
        """
        LibriSpeech test-clean veri setini indirir ve açar.

        Args:
            force: True ise mevcut veriyi silip yeniden indirir.

        Returns:
            bool: İşlem başarılı ise True.
        """
        if self.is_downloaded() and not force:
            print("✅ Veri seti zaten mevcut, indirme atlanıyor.")
            return True

        os.makedirs(self.data_dir, exist_ok=True)

        # İndirme
        if not os.path.exists(self.archive_path) or force:
            print(f"📥 LibriSpeech test-clean indiriliyor...")
            print(f"   Kaynak: {LIBRISPEECH_URL}")
            print(f"   Hedef:  {self.archive_path}")
            print(f"   ⚠️  Boyut ~350 MB, internet hızınıza bağlı olarak biraz sürebilir.\n")

            try:
                _download_with_progress(LIBRISPEECH_URL, self.archive_path)
            except Exception as e:
                print(f"❌ İndirme başarısız: {e}")
                return False
        else:
            print("📦 Arşiv zaten indirilmiş, açılıyor...")

        # Arşivi aç
        print("📂 Arşiv açılıyor...")
        try:
            with tarfile.open(self.archive_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            print("✅ Veri seti başarıyla hazırlandı.")
        except Exception as e:
            print(f"❌ Arşiv açma başarısız: {e}")
            return False

        # Arşiv dosyasını sil (disk alanı tasarrufu)
        try:
            os.remove(self.archive_path)
            print("🗑️  Arşiv dosyası silindi (disk tasarrufu).")
        except Exception:
            pass

        return True

    def get_samples(self, limit=None, min_duration=1.0, max_duration=30.0):
        """
        Veri setinden ses örneklerini ve transkripsiyon bilgilerini döndürür.

        Args:
            limit: Döndürülecek maksimum örnek sayısı. None ise tümü.
            min_duration: Minimum ses süresi (saniye).
            max_duration: Maksimum ses süresi (saniye).

        Returns:
            list[dict]: Her öğe şu anahtarları içerir:
                - audio_path (str): Ses dosyasının tam yolu
                - transcript (str): Ground-truth transkripsiyon
                - speaker_id (str): Konuşmacı ID'si
                - chapter_id (str): Bölüm ID'si
                - utterance_id (str): Söylev ID'si
                - duration (float): Ses süresi (saniye)
        """
        if not self.is_downloaded():
            print("⚠️  Veri seti bulunamadı. Önce download() çağırın.")
            return []

        samples = []

        # LibriSpeech yapısı: speaker_id/chapter_id/*.flac + *.trans.txt
        for speaker_dir in sorted(Path(self.dataset_root).iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name

            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue

                chapter_id = chapter_dir.name

                # Transkripsiyon dosyasını oku
                trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                if not trans_file.exists():
                    continue

                transcriptions = _parse_transcription_file(trans_file)

                for utt_id, transcript in transcriptions.items():
                    audio_path = chapter_dir / f"{utt_id}.flac"
                    if not audio_path.exists():
                        continue

                    # Süre filtresi
                    try:
                        info = sf.info(str(audio_path))
                        duration = info.duration
                    except Exception:
                        continue

                    if duration < min_duration or duration > max_duration:
                        continue

                    samples.append({
                        "audio_path": str(audio_path),
                        "transcript": transcript,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utt_id,
                        "duration": round(duration, 2),
                    })

                    if limit and len(samples) >= limit:
                        return samples

        return samples

    def get_summary(self):
        """Veri seti hakkında özet istatistikler döndürür."""
        samples = self.get_samples()
        if not samples:
            return {"status": "not_ready", "count": 0}

        durations = [s["duration"] for s in samples]
        speakers = set(s["speaker_id"] for s in samples)

        return {
            "status": "ready",
            "count": len(samples),
            "total_duration_min": round(sum(durations) / 60, 1),
            "avg_duration_sec": round(sum(durations) / len(durations), 1),
            "min_duration_sec": round(min(durations), 1),
            "max_duration_sec": round(max(durations), 1),
            "unique_speakers": len(speakers),
        }

    def cleanup(self):
        """Tüm veri setini siler."""
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
            print("🗑️  Veri seti silindi.")


# --- Yardımcı Fonksiyonlar ---

def _download_with_progress(url, dest_path):
    """İlerleme çubuğu ile dosya indirir."""
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    block_size = 1024 * 1024  # 1 MB

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
                print(f"\r   İlerleme: {mb_done:.1f} / {mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)

    print()  # Yeni satır


def _parse_transcription_file(trans_path):
    """
    LibriSpeech .trans.txt dosyasını ayrıştırır.

    Format: UTTERANCE_ID TRANSCRIPT TEXT HERE
    
    Returns:
        dict: {utterance_id: transcript_text, ...}
    """
    transcriptions = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                utt_id, transcript = parts
                transcriptions[utt_id] = transcript.strip()
    return transcriptions


if __name__ == "__main__":
    # Doğrudan çalıştırma: veri setini indir ve özet göster
    manager = DatasetManager()
    manager.download()

    summary = manager.get_summary()
    print(f"\n📊 Veri Seti Özeti:")
    for key, val in summary.items():
        print(f"   {key}: {val}")

    print(f"\n📎 İlk 3 örnek:")
    for sample in manager.get_samples(limit=3):
        print(f"   🔈 {os.path.basename(sample['audio_path'])}")
        print(f"      Konuşmacı: {sample['speaker_id']}, Süre: {sample['duration']}s")
        print(f"      Transkript: {sample['transcript'][:80]}...")
        print()
