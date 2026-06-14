"""CHiME-6 — çok konuşmacılı, gürültülü toplantı konuşması (ASR + diarization).

CHiME-6 verisi lisans nedeniyle OTOMATİK İNDİRİLMEZ; kullanıcı elle yerleştirir:
    datasets/chime6/audio/            (S0X_*.wav, far-field U0X.CHn.wav, worn P0X.wav)
    datasets/chime6/transcriptions/   (S0X.json)

Bu manager sadece varsayılan dizinleri çözer ve session keşfi sağlar. Far-field/
worn/GSS audio-yol çözümü benchmark runner'ında (tests/benchmarks/chime6.py)
kalır, çünkü enhance/array seçeneklerine sıkı bağlıdır.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .base import DATASETS_ROOT


# --------------------------------------------------------------------------- #
# CHiME-6 transkript ayrıştırma yardımcıları (saf; benchmark runner buradan alır)
# --------------------------------------------------------------------------- #
def parse_time_to_seconds(t_val):
    """Zaman damgasını float saniyeye çevirir.

    float/int, string float saniye ve HH:MM:SS.mmm formatlarını destekler.
    CHiME-5/6 start_time/end_time'ı cihaz-anahtarlı dict olarak tutabilir
    (ör. {"original": "0:01:20.12", ...}); senkronizasyon sonrası tüm cihazlar
    tek zaman çizgisini paylaştığından 'original' (yoksa ilk değer) alınır.
    """
    if isinstance(t_val, dict):
        if "original" in t_val:
            t_val = t_val["original"]
        elif t_val:
            t_val = next(iter(t_val.values()))
        else:
            raise ValueError("Empty time dict")

    if isinstance(t_val, (int, float)):
        return float(t_val)

    t_str = str(t_val).strip()
    try:
        return float(t_str)
    except ValueError:
        pass

    parts = t_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    else:
        raise ValueError(f"Unknown time format: {t_val}")


def clean_chime6_text(text):
    """Köşeli-parantez etiketlerini ([noise], [laughs]) ve noktalamayı temizler."""
    if not text:
        return ""
    text = re.sub(r"\[[^\]]*\]", "", text)        # [noise], [laughter], ...
    text = text.replace("-", " ")                  # tire -> boşluk
    text = re.sub(r"[^\w\s']", "", text)           # noktalama (kesme hariç)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Chime6DatasetManager:
    def __init__(self, data_dir=None):
        self.base = data_dir or os.path.join(DATASETS_ROOT, "chime6")
        self.audio_dir = os.path.join(self.base, "audio")
        self.transcriptions_dir = os.path.join(self.base, "transcriptions")

    def is_downloaded(self):
        return os.path.isdir(self.audio_dir) and os.path.isdir(self.transcriptions_dir)

    def download(self, force=False):
        """CHiME-6 elle yerleştirilir; burada sadece varlık kontrolü yapılır."""
        if self.is_downloaded():
            print(f"✅ CHiME-6 mevcut: {self.base}")
            return True
        print("⚠️  CHiME-6 otomatik indirilmez (lisanslı). Veriyi şuraya yerleştirin:")
        print(f"   {self.audio_dir}")
        print(f"   {self.transcriptions_dir}")
        return False

    def sessions(self):
        """Transkript dosyalarından session ID listesi (örn. ['S02', 'S09'])."""
        tdir = Path(self.transcriptions_dir)
        if not tdir.is_dir():
            return []
        return sorted(p.stem for p in tdir.glob("*.json"))

    def transcript_path(self, session_id):
        return os.path.join(self.transcriptions_dir, f"{session_id}.json")

    def get_summary(self):
        if not self.is_downloaded():
            return {"status": "not_ready", "count": 0}
        sessions = self.sessions()
        return {"status": "ready", "sessions": len(sessions),
                "session_ids": ", ".join(sessions)}
