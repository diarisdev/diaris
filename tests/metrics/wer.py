"""WER (Word Error Rate) ve CER (Character Error Rate) — kelime/karakter hatası.

Saf fonksiyonel API. Konuşmacıdan bağımsız ASR doğruluk ölçümü için. Konuşmacı-
bazlı değerlendirme cpwer.py'dedir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .edit_distance import levenshtein_sid


def normalize_text(text: str) -> str:
    """Adil karşılaştırma için metin normalizasyonu.

    - Küçük harfe çevirir
    - Noktalama işaretlerini kaldırır (kesme işareti dahil — sadeleştirme)
    - Fazla boşlukları temizler
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # noktalama kaldır
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class WerResult:
    """Tek bir referans-hipotez çiftinin WER dökümü."""
    ref_words: int
    hyp_words: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        """0.0–1.0+ aralığında oran (yüzde için *100)."""
        return self.errors / self.ref_words if self.ref_words else 0.0


@dataclass
class CerResult:
    """Tek bir referans-hipotez çiftinin CER dökümü."""
    ref_chars: int
    hyp_chars: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def cer(self) -> float:
        return self.errors / self.ref_chars if self.ref_chars else 0.0


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> WerResult:
    """İki metin arasındaki WER'i hesaplar."""
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    ref_words = ref.split()
    hyp_words = hyp.split()
    subs, ins, dels = levenshtein_sid(ref_words, hyp_words)
    return WerResult(len(ref_words), len(hyp_words), subs, ins, dels)


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> CerResult:
    """İki metin arasındaki CER'i hesaplar (boşluklar yok sayılır)."""
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    subs, ins, dels = levenshtein_sid(ref_chars, hyp_chars)
    return CerResult(len(ref_chars), len(hyp_chars), subs, ins, dels)
