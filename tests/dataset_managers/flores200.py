"""FLORES-200 — Meta'nın NLLB-200 makalesindeki paralel çeviri benchmark'ı.

Çeviri motorlarının (Google/DeepL/NLLB) kalitesini akademik makalelerle
KARŞILAŞTIRILABİLİR biçimde ölçmek için. Tamamen paralel: her split'te aynı
satır numarası tüm dillerde aynı cümle.

Split'ler: dev (997) / devtest (1012 — raporlamada bu kullanılır).

Kullanım:
    m = FloresDatasetManager(); m.download()
    pairs = m.get_pairs("en", "tr", split="devtest", limit=100)
"""

from __future__ import annotations

import os
import shutil

from .base import DATASETS_ROOT, download_targz

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_INNER = "flores200_dataset"   # arşivin iç klasörü

# Projedeki kısa dil kodları -> FLORES-200 kodları (engine.py lang_map ile aynı diller)
FLORES_LANG_MAP = {
    "en": "eng_Latn", "tr": "tur_Latn", "de": "deu_Latn", "fr": "fra_Latn",
    "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl",
    "zh": "zho_Hans", "ar": "arb_Arab", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "nl": "nld_Latn",
}


class FloresDatasetManager:
    """FLORES-200 indir/aç + paralel (kaynak, referans) çiftleri."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(DATASETS_ROOT, "flores200")
        self.archive_path = os.path.join(self.data_dir, "flores200_dataset.tar.gz")
        self.dataset_root = os.path.join(self.data_dir, FLORES_INNER)

    def is_downloaded(self):
        return os.path.isdir(os.path.join(self.dataset_root, "devtest"))

    def download(self, force=False):
        if self.is_downloaded() and not force:
            print("✅ FLORES-200 zaten mevcut, indirme atlanıyor.")
            return True
        return download_targz(
            FLORES_URL, self.archive_path, self.data_dir,
            label="FLORES-200", size_hint="Boyut ~25 MB.",
        )

    def _read_split_file(self, lang_code, split):
        path = os.path.join(self.dataset_root, split, f"{lang_code}.{split}")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FLORES dosyası bulunamadı: {path}. "
                f"Dil kodu '{lang_code}' veya split '{split}' yanlış olabilir."
            )
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def get_pairs(self, source_lang, target_lang, split="devtest", limit=None):
        """İki dil arasındaki paralel (kaynak, referans) çiftlerini döndürür."""
        if not self.is_downloaded():
            print("⚠️  FLORES-200 bulunamadı. Önce download() çağırın.")
            return []

        src_code = FLORES_LANG_MAP.get(source_lang.split("-")[0].lower(), source_lang)
        tgt_code = FLORES_LANG_MAP.get(target_lang.split("-")[0].lower(), target_lang)

        src_lines = self._read_split_file(src_code, split)
        tgt_lines = self._read_split_file(tgt_code, split)
        if len(src_lines) != len(tgt_lines):
            print(f"⚠️  Satır sayıları uyuşmuyor ({src_code}: {len(src_lines)}, "
                  f"{tgt_code}: {len(tgt_lines)}). Kısa olana göre kesiliyor.")

        count = min(len(src_lines), len(tgt_lines))
        if limit:
            count = min(count, limit)
        return [
            {"source": src_lines[i], "reference": tgt_lines[i],
             "source_lang": source_lang, "target_lang": target_lang, "index": i}
            for i in range(count)
        ]

    def available_languages(self):
        return sorted(FLORES_LANG_MAP.keys())

    def get_summary(self, split="devtest"):
        if not self.is_downloaded():
            return {"status": "not_ready", "count": 0}
        try:
            eng_lines = self._read_split_file("eng_Latn", split)
        except FileNotFoundError:
            return {"status": "not_ready", "count": 0}
        return {
            "status": "ready", "split": split,
            "sentence_count": len(eng_lines),
            "supported_languages": len(FLORES_LANG_MAP),
            "language_codes": ", ".join(sorted(FLORES_LANG_MAP.keys())),
        }

    def cleanup(self):
        if os.path.isdir(self.dataset_root):
            shutil.rmtree(self.dataset_root)
            print("🗑️  FLORES-200 silindi.")
