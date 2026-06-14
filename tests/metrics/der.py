"""DER — Diarization Error Rate (pyannote.metrics ile).

Konuşmacı ayrımı doğruluğu: false alarm + missed detection + speaker confusion.
pyannote.metrics kurulu değilse fonksiyonlar None döner.

Konvansiyon notu:
  * CHiME-6: collar=0.0, örtüşme DAHİL (skip_overlap=False) — varsayılan budur.
  * AMI çalışmalarında genelde collar=0.25 kullanılır; parametreyle verilir.

Döndürülen değerler ORANDIR (0.0–1.0+); yüzde için *100.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False


@dataclass
class DerResult:
    der: float                 # oran
    false_alarm: float
    missed: float
    confusion: float
    # Örtüşme HARİÇ varyant (opsiyonel; kaçınılmaz overlap cezasını ayırmak için)
    der_no_overlap: float | None = None
    missed_no_overlap: float | None = None
    confusion_no_overlap: float | None = None


def _to_annotation(intervals, uri: str = "meeting"):
    """[{start,end,speaker}, ...] -> pyannote.core.Annotation."""
    ann = Annotation(uri=uri)
    for it in intervals:
        if it["end"] > it["start"]:
            ann[Segment(it["start"], it["end"])] = it["speaker"]
    return ann


def load_rttm(path) -> list:
    """NIST RTTM dosyasını [{start,end,speaker}, ...] listesine ayrıştırır."""
    path = Path(path)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        if dur > 0:
            out.append({"start": start, "end": start + dur, "speaker": spk})
    return out


def _run(ref_ann, hyp_ann, collar: float, skip_overlap: bool) -> dict:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    der = metric(ref_ann, hyp_ann)
    comp = metric(ref_ann, hyp_ann, detailed=True)
    total = comp.get("total", 0.0) or 1e-8
    return {
        "der": der,
        "false_alarm": comp.get("false alarm", 0.0) / total,
        "missed": comp.get("missed detection", 0.0) / total,
        "confusion": comp.get("confusion", 0.0) / total,
    }


def compute_der(ref_intervals, hyp_intervals, collar: float = 0.0,
                skip_overlap: bool = False, with_overlap_split: bool = False,
                uri: str = "meeting") -> DerResult | None:
    """İki interval listesi arasında DER hesaplar.

    Args:
        ref_intervals / hyp_intervals: [{start, end, speaker}, ...]
        collar: Sınır toleransı (s). CHiME-6 = 0.0, AMI = 0.25.
        skip_overlap: True ise örtüşen konuşma değerlendirme dışı.
        with_overlap_split: True ise ayrıca örtüşme-hariç varyantı da doldurur.

    Returns:
        DerResult ya da pyannote yoksa None.
    """
    if not HAS_PYANNOTE:
        return None

    ref_ann = _to_annotation(ref_intervals, uri)
    hyp_ann = _to_annotation(hyp_intervals, uri)

    full = _run(ref_ann, hyp_ann, collar, skip_overlap=False)
    res = DerResult(full["der"], full["false_alarm"], full["missed"], full["confusion"])

    if with_overlap_split:
        noov = _run(ref_ann, hyp_ann, collar, skip_overlap=True)
        res.der_no_overlap = noov["der"]
        res.missed_no_overlap = noov["missed"]
        res.confusion_no_overlap = noov["confusion"]

    return res
