"""
Konuşmacı renk paleti — tüm UI bileşenlerinin ORTAK kaynağı.

Konuşmacıya özgü, kararlı (stable) renk ataması sağlar: aynı SPEAKER_XX etiketi
her zaman aynı rengi alır; warm-up ("Calibrating"), "Unknown" ve geçici
"Çözümleniyor..." durumlarının kendi nötr renkleri vardır. Böylece altyazı
overlay'i ile diğer bileşenler (log paneli vb.) tutarlı görünür.
"""

# Konuşmacıya göre döngüsel renkler (yumuşak, koyu tema dostu)
SPEAKER_COLORS = [
    "#3498db",  # mavi
    "#2ecc71",  # yeşil
    "#e74c3c",  # kırmızı
    "#f1c40f",  # sarı
    "#9b59b6",  # mor
    "#1abc9c",  # turkuaz
    "#e67e22",  # turuncu
]

# Özel durum renkleri
CALIBRATING_COLOR = "#8a93a3"   # warm-up / kalibrasyon (nötr gri-mavi)
RESOLVING_COLOR = "#8a93a3"     # "Çözümleniyor..." (henüz diarize edilmedi)
UNKNOWN_COLOR = "#9aa6b2"       # eşleşmeyen / bilinmeyen
DEFAULT_TEXT_COLOR = "#ffffff"  # konuşmacı yoksa düz metin

_RESOLVING_TOKENS = ("çözümleniyor", "resolving", "analyzing", "analiz")
_CALIBRATING_TOKENS = ("calibrating", "kalibr")


def speaker_index(label: str):
    """SPEAKER_XX etiketinden sayısal id çıkarır; yoksa None."""
    if label and "SPEAKER_" in label:
        try:
            return int(label.split("SPEAKER_")[1].split()[0].strip("]"))
        except (ValueError, IndexError):
            return None
    return None


def color_for_speaker(label: str) -> str:
    """
    Bir konuşmacı etiketi için kararlı renk döndürür.

    - SPEAKER_XX → palette[idx % N] (aynı id hep aynı renk)
    - Calibrating / Çözümleniyor / Unknown → nötr renkler
    - Diğer (örn. isimlendirilmiş) → etiket hash'ine göre kararlı renk
    """
    if not label:
        return DEFAULT_TEXT_COLOR

    low = label.lower()
    if any(tok in low for tok in _RESOLVING_TOKENS):
        return RESOLVING_COLOR
    if any(tok in low for tok in _CALIBRATING_TOKENS):
        return CALIBRATING_COLOR
    if low in ("unknown", "bilinmeyen"):
        return UNKNOWN_COLOR

    idx = speaker_index(label)
    if idx is not None:
        return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]

    # İsimlendirilmiş/diğer etiketler: kararlı hash
    return SPEAKER_COLORS[sum(ord(c) for c in label) % len(SPEAKER_COLORS)]


def display_name(label: str) -> str:
    """SPEAKER_00 → 'Konuşmacı 1' gibi kullanıcı dostu ad; diğerleri olduğu gibi."""
    idx = speaker_index(label)
    if idx is not None:
        return f"Konuşmacı {idx + 1}"
    return label
