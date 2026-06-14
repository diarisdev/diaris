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
from pathlib import Path

# soundfile yalnızca SES veri setleri (LibriSpeech/AMI) için gerekir. FLORES-200
# (metin çeviri) yöneticisi ses kütüphanesi gerektirmesin diye import tembel
# yapılır; ses süresi okuyan fonksiyonlar çağrılınca yüklenir.
try:
    import soundfile as sf
except ImportError:
    sf = None


# --- Veri Seti Sabitleri ---
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_DIR_NAME = "LibriSpeech/test-clean"

# FLORES-200: Meta'nın NLLB-200 makalesinde kullanılan, 200 dili kapsayan
# insan-çevirili paralel çeviri benchmark'ı. Çeviri motorlarımızın (Google,
# DeepL, yerel NLLB) kalitesini akademik makalelerle KARŞILAŞTIRILABİLİR
# biçimde ölçmek için kullanılır.
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_DIR_NAME = "flores200_dataset"

# Projedeki kısa dil kodları -> FLORES-200 kodları.
# src/translation/engine.py içindeki CTranslate2 lang_map ile aynı diller.
FLORES_LANG_MAP = {
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
}

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

        if sf is None:
            raise ImportError(
                "LibriSpeech örnekleri için 'soundfile' gerekli: pip install soundfile"
            )

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



class FloresDatasetManager:
    """
    FLORES-200 çeviri benchmark veri setini indiren, açan ve
    paralel (kaynak metin, referans çeviri) çiftlerini yöneten sınıf.

    FLORES-200 tamamen PARALEL'dir: her split'te aynı satır numarası, tüm
    dillerde aynı cümleye karşılık gelir. Böylece Google/DeepL/NLLB motorları
    AYNI cümle kümesi üzerinde adil biçimde karşılaştırılabilir.

    Split'ler:
      - dev      (997 cümle)   -> ayar/geliştirme için
      - devtest  (1012 cümle)  -> RAPORLAMA için (makalelerde bu kullanılır)

    Kullanım:
        manager = FloresDatasetManager()
        manager.download()
        pairs = manager.get_pairs("en", "tr", split="devtest", limit=100)
        # -> [{"source": "...", "reference": "...",
        #      "source_lang": "en", "target_lang": "tr", "index": 0}, ...]
    """

    def __init__(self, data_dir=None):
        """
        Args:
            data_dir: Veri setinin indirileceği dizin.
                      None ise tests/data/ kullanılır.
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.archive_path = os.path.join(self.data_dir, "flores200_dataset.tar.gz")
        self.dataset_root = os.path.join(self.data_dir, FLORES_DIR_NAME)

    def is_downloaded(self):
        """Veri seti indirilmiş ve açılmış mı kontrol eder (devtest dizinine bakar)."""
        return os.path.isdir(os.path.join(self.dataset_root, "devtest"))

    def download(self, force=False):
        """
        FLORES-200 veri setini indirir ve açar.

        Args:
            force: True ise mevcut veriyi silip yeniden indirir.

        Returns:
            bool: İşlem başarılı ise True.
        """
        if self.is_downloaded() and not force:
            print("✅ FLORES-200 zaten mevcut, indirme atlanıyor.")
            return True

        os.makedirs(self.data_dir, exist_ok=True)

        # İndirme
        if not os.path.exists(self.archive_path) or force:
            print(f"📥 FLORES-200 indiriliyor...")
            print(f"   Kaynak: {FLORES_URL}")
            print(f"   Hedef:  {self.archive_path}")
            print(f"   ⚠️  Boyut ~25 MB.\n")

            try:
                _download_with_progress(FLORES_URL, self.archive_path)
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
            print("✅ FLORES-200 başarıyla hazırlandı.")
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

    def _read_split_file(self, lang_code, split):
        """Tek bir dilin split dosyasını satır listesi olarak okur."""
        path = os.path.join(self.dataset_root, split, f"{lang_code}.{split}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FLORES dosyası bulunamadı: {path}. "
                f"Dil kodu '{lang_code}' veya split '{split}' yanlış olabilir."
            )
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def get_pairs(self, source_lang, target_lang, split="devtest", limit=None):
        """
        İki dil arasındaki paralel (kaynak, referans) çiftlerini döndürür.

        Args:
            source_lang: Kaynak dil — kısa kod ("en") veya FLORES kodu ("eng_Latn").
            target_lang: Hedef dil — kısa kod ("tr") veya FLORES kodu ("tur_Latn").
            split: "devtest" (raporlama) veya "dev" (ayar).
            limit: Döndürülecek maksimum çift sayısı. None ise tümü.

        Returns:
            list[dict]: Her öğe şu anahtarları içerir:
                - source (str): Kaynak dildeki cümle
                - reference (str): Hedef dildeki insan-çevirili referans
                - source_lang (str): İstenen kaynak dil kodu
                - target_lang (str): İstenen hedef dil kodu
                - index (int): Cümlenin split içindeki satır numarası
        """
        if not self.is_downloaded():
            print("⚠️  FLORES-200 bulunamadı. Önce download() çağırın.")
            return []

        src_code = FLORES_LANG_MAP.get(source_lang.split("-")[0].lower(), source_lang)
        tgt_code = FLORES_LANG_MAP.get(target_lang.split("-")[0].lower(), target_lang)

        src_lines = self._read_split_file(src_code, split)
        tgt_lines = self._read_split_file(tgt_code, split)

        if len(src_lines) != len(tgt_lines):
            print(
                f"⚠️  Satır sayıları uyuşmuyor ({src_code}: {len(src_lines)}, "
                f"{tgt_code}: {len(tgt_lines)}). Kısa olana göre kesiliyor."
            )

        count = min(len(src_lines), len(tgt_lines))
        if limit:
            count = min(count, limit)

        return [
            {
                "source": src_lines[i],
                "reference": tgt_lines[i],
                "source_lang": source_lang,
                "target_lang": target_lang,
                "index": i,
            }
            for i in range(count)
        ]

    def available_languages(self):
        """Bu yöneticinin desteklediği kısa dil kodlarını döndürür."""
        return sorted(FLORES_LANG_MAP.keys())

    def get_summary(self, split="devtest"):
        """Veri seti hakkında özet istatistikler döndürür."""
        if not self.is_downloaded():
            return {"status": "not_ready", "count": 0}

        try:
            eng_lines = self._read_split_file("eng_Latn", split)
        except FileNotFoundError:
            return {"status": "not_ready", "count": 0}

        return {
            "status": "ready",
            "split": split,
            "sentence_count": len(eng_lines),
            "supported_languages": len(FLORES_LANG_MAP),
            "language_codes": ", ".join(sorted(FLORES_LANG_MAP.keys())),
        }

    def cleanup(self):
        """FLORES veri setini siler (diğer test verilerine dokunmaz)."""
        if os.path.isdir(self.dataset_root):
            shutil.rmtree(self.dataset_root)
            print("🗑️  FLORES-200 silindi.")


class AmiDiarizationManager:
    """
    AMI Meeting Corpus'un küçük bir alt kümesini indirir ve
    Diarization testleri için hazırlar.
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.ami_dir = os.path.join(self.data_dir, "AMI_mini")
        self.meetings = ["EN2001a", "ES2002a"] # Test için iki meeting

    def is_downloaded(self):
        if not os.path.isdir(self.ami_dir):
            return False
        for meeting in self.meetings:
            wav_path = os.path.join(self.ami_dir, f"{meeting}.Mix-Headset.wav")
            rttm_path = os.path.join(self.ami_dir, f"{meeting}.rttm")
            if not os.path.exists(wav_path) or not os.path.exists(rttm_path):
                return False
        return True

    def download(self, force=False):
        if self.is_downloaded() and not force:
            print("✅ AMI verileri zaten mevcut.")
            return True

        os.makedirs(self.ami_dir, exist_ok=True)
        print("📥 AMI Diarization veri seti hazırlanıyor...")

        # AMI-diarization-setup repo yolu
        setup_repo_dir = os.path.join(self.data_dir, "AMI-diarization-setup")
        repo_rttm_dir = os.path.join(setup_repo_dir, "only_words", "rttms", "train")

        if not os.path.exists(setup_repo_dir):
            print("   📥 AMI-diarization-setup reposu klonlanıyor...")
            import subprocess
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/BUTSpeechFIT/AMI-diarization-setup.git", setup_repo_dir],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"❌ Repo klonlama hatası: {e}")
                return False

        for meeting in self.meetings:
            wav_path = os.path.join(self.ami_dir, f"{meeting}.Mix-Headset.wav")
            rttm_path = os.path.join(self.ami_dir, f"{meeting}.rttm")

            if not os.path.exists(wav_path) or force:
                url = f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.Mix-Headset.wav"
                print(f"   İndiriliyor: {meeting} audio...")
                try:
                    _download_with_progress(url, wav_path)
                except Exception as e:
                    print(f"❌ Ses indirme hatası ({meeting}): {e}")
                    return False

            if not os.path.exists(rttm_path) or force:
                src_rttm = os.path.join(repo_rttm_dir, f"{meeting}.rttm")
                if os.path.exists(src_rttm):
                    shutil.copy2(src_rttm, rttm_path)
                else:
                    print(f"❌ RTTM bulunamadı: {src_rttm}")
                    return False

        print("✅ AMI veri seti başarıyla hazırlandı.")
        return True

    def get_samples(self):
        """
        Diarization test örneklerini döndürür.
        """
        if not self.is_downloaded():
            print("⚠️ AMI verisi bulunamadı. Önce download() çağırın.")
            return []

        samples = []
        for meeting in self.meetings:
            wav_path = os.path.join(self.ami_dir, f"{meeting}.Mix-Headset.wav")
            rttm_path = os.path.join(self.ami_dir, f"{meeting}.rttm")
            
            # Ses süresini al
            try:
                info = sf.info(wav_path)
                duration = info.duration
            except Exception:
                duration = 0.0

            # RTTM ayrıştır
            annotations = self._parse_rttm(rttm_path)

            samples.append({
                "audio_path": wav_path,
                "rttm_path": rttm_path,
                "meeting_id": meeting,
                "duration": duration,
                "annotations": annotations
            })
        return samples

    def _parse_rttm(self, rttm_path):
        """
        RTTM dosyasını ayrıştırıp speaker annotation listesi döner.
        Format: SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
        """
        annotations = []
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    annotations.append({
                        "start": start_time,
                        "end": start_time + duration,
                        "speaker": speaker
                    })
        return annotations


def _demo_flores():
    """FLORES-200'ü indirir ve örnek bir çeviri çifti gösterir."""
    print("\n" + "=" * 60)
    print("FLORES-200 (çeviri benchmark)")
    print("=" * 60)
    manager = FloresDatasetManager()
    manager.download()

    summary = manager.get_summary()
    print(f"\n📊 Veri Seti Özeti:")
    for key, val in summary.items():
        print(f"   {key}: {val}")

    print(f"\n📎 İlk 3 paralel çift (en -> tr):")
    for pair in manager.get_pairs("en", "tr", split="devtest", limit=3):
        print(f"   [{pair['index']}] SRC: {pair['source'][:70]}")
        print(f"        REF: {pair['reference'][:70]}")
        print()


def _demo_librispeech():
    """LibriSpeech'i indirir ve özet gösterir (soundfile gerekir, ~350 MB)."""
    print("\n" + "=" * 60)
    print("LibriSpeech test-clean (transkripsiyon benchmark)")
    print("=" * 60)
    manager = DatasetManager()
    manager.download()

    try:
        summary = manager.get_summary()
        print(f"\n📊 Veri Seti Özeti:")
        for key, val in summary.items():
            print(f"   {key}: {val}")
    except ImportError as e:
        print(f"⚠️  Özet atlandı: {e}")


def _demo_ami():
    """AMI Meeting Corpus alt kümesini indirir (diarization benchmark)."""
    print("\n" + "=" * 60)
    print("AMI Meeting Corpus (diarization benchmark)")
    print("=" * 60)
    manager = AmiDiarizationManager()
    manager.download()

    samples = manager.get_samples()
    print(f"\n📊 Veri Seti Özeti:")
    print(f"   meeting_count: {len(samples)}")
    for s in samples:
        print(f"   🔈 {s['meeting_id']} | süre: {s['duration']:.1f}s | "
              f"annotation: {len(s['annotations'])}")


# Çalıştırıldığında indirilecek tüm veri setleri (sıra korunur).
_DATASET_DEMOS = {
    "flores": _demo_flores,           # çeviri (FLORES-200, ~25 MB)
    "librispeech": _demo_librispeech, # transkripsiyon (LibriSpeech, ~350 MB)
    "ami": _demo_ami,                 # diarization (AMI alt kümesi)
}


if __name__ == "__main__":
    import argparse
    import sys

    # Windows konsolu (ör. cp1254) emoji çıktısında çökmesin diye UTF-8'e geç.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Test veri setlerini indir (varsayılan: HEPSİ)."
    )
    parser.add_argument(
        "dataset", nargs="?", default="all",
        choices=["all"] + list(_DATASET_DEMOS.keys()),
        help="İndirilecek veri seti (varsayılan: all = tümü).",
    )
    args = parser.parse_args()

    selected = _DATASET_DEMOS.keys() if args.dataset == "all" else [args.dataset]

    failures = []
    for name in selected:
        try:
            _DATASET_DEMOS[name]()
        except Exception as e:
            failures.append((name, e))
            print(f"❌ '{name}' hazırlanamadı: {e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"⚠️  {len(failures)} veri seti başarısız: "
              f"{', '.join(n for n, _ in failures)}")
    else:
        print("✅ Tüm veri setleri hazır.")
    print("=" * 60)
