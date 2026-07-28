"""Karar izlerini referans anotasyonla hizalama — TEK KAYNAK.

`ami_replay --trace-embeddings` her diarization chunk'ının konuşmacı karar izini
kaydeder (skorlar, eşikler, atanan etiket, turn zaman aralıkları). Bu modül o
izleri AMI referans RTTM'iyle hizalar: "bu karar doğru muydu?" sorusunu
cevaplanabilir hale getirir.

İki tüketicisi var ve aynı hizalamayı görmeleri şart:
  * scripts/analyze_decision_accuracy.py  — karar tipine göre hata analizi
  * scripts/calibrate_speaker_posterior.py — posterior parametrelerini uydurma

Matematiği burada tutuyoruz ki iki script farklı hizalama uygulayıp
karşılaştırılamaz sayılar üretmesin (tests/metrics ile aynı gerekçe).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def load_reference(rttm_path) -> list[tuple[float, float, str]]:
    """RTTM → [(start, end, konuşmacı), ...]"""
    spans = []
    for line in Path(rttm_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        spans.append((start, start + float(parts[4]), parts[7]))
    return spans


def reference_at(spans, intervals) -> tuple[str | None, float]:
    """Verilen aralıklarda EN ÇOK örtüşen referans konuşmacı ve örtüşme süresi.

    Örtüşme, turn'lerin BİRLEŞİK ZARFI üzerinden değil her turn üzerinden ayrı
    hesaplanır: pyannote aynı yerel etiketi bir chunk içinde araları başka
    konuşmacıyla dolu iki parçaya verebilir; zarf o araya düşen konuşmayı da
    sayar ve baskın referansı yanlış seçtirir.

    Örtüşmeli konuşmada birden çok referans konuşmacı vardır; baskın olanı
    alıyoruz — tek akışlı bir sistemin verebileceği en iyi cevap budur.
    """
    overlap = defaultdict(float)
    for start, end in intervals:
        for ref_start, ref_end, speaker in spans:
            shared = min(end, ref_end) - max(start, ref_start)
            if shared > 0:
                overlap[speaker] += shared
    if not overlap:
        return None, 0.0
    best = max(overlap.items(), key=lambda kv: kv[1])
    return best[0], best[1]


def collect_decisions(trace: dict, reference) -> list[dict]:
    """İzdeki her probe'u referansla eşleştirir.

    Returns:
        [{decision, assigned, reference, duration, overlap, margin, best_score,
          raw_scores, probe_duration}, ...]
    """
    decisions = []
    for frame in trace["frames"]:
        offset = frame.get("time") or 0.0
        for probe in (frame.get("trace") or {}).get("probes", []):
            turns = probe.get("turns") or []
            if not turns:
                continue
            intervals = [(offset + float(t[0]), offset + float(t[1])) for t in turns]
            ref_speaker, shared = reference_at(reference, intervals)
            decisions.append({
                "decision": probe.get("decision", "?"),
                "assigned": probe.get("assigned"),
                "reference": ref_speaker,
                # Konuşma süresi: turn'lerin toplamı (zarf değil).
                "duration": sum(max(0.0, e - s) for s, e in intervals),
                "overlap": shared,
                "margin": probe.get("margin"),
                "best_score": probe.get("best_score"),
                # Posterior kalibrasyonu ham skorlar üzerinden yapılır.
                "raw_scores": probe.get("raw_scores") or probe.get("scores") or {},
                # Embedding'in kendi kalite süresi (tracker'ın gördüğü).
                "probe_duration": probe.get("duration"),
            })
    return decisions


def build_label_mapping(decisions) -> dict[str, str]:
    """Hipotez etiketi → referans konuşmacı (toplam örtüşmeye göre, açgözlü).

    DER/cpWER'in yaptığı optimal eşlemenin basit karşılığı: küresel olarak en
    çok örtüşen çiftten başlayarak eşleştir. Her hipotez etiketi ve her referans
    konuşmacısı en fazla bir kez kullanılır.

    NOT: hipotez etiketi sayısı referans konuşmacı sayısını aşarsa fazlalıklar
    hiçbir referansa eşlenemez ve onlara atanan her karar "yanlış" sayılır —
    bu, over-count'un maliyetinin doğru yansımasıdır.
    """
    weights = defaultdict(float)
    for decision in decisions:
        if decision["assigned"] and decision["reference"]:
            weights[(decision["assigned"], decision["reference"])] += decision["duration"]

    mapping = {}
    used_refs = set()
    for (hyp, ref), _ in sorted(weights.items(), key=lambda kv: -kv[1]):
        if hyp in mapping or ref in used_refs:
            continue
        mapping[hyp] = ref
        used_refs.add(ref)
    return mapping


def labelled_score_pairs(decisions, mapping) -> list[tuple[float, bool]]:
    """Kalibrasyon verisi: (benzerlik skoru, bu profil GERÇEK konuşmacı mı).

    Yalnızca en iyi eşleşme değil, o karar anındaki TÜM profiller etiketlenir —
    posterior tüm dağılımı modellediği için negatif örnekler de gerekir.
    """
    pairs = []
    for decision in decisions:
        reference = decision["reference"]
        if reference is None:
            continue
        for label, score in (decision["raw_scores"] or {}).items():
            pairs.append((float(score), mapping.get(label) == reference))
    return pairs
