"""Konuşmacı etiketlerinde son-işleme (finalization refinement) katmanı.

Canlı takip (SpeakerTracker) kararlarını GERİ ALAMAZ: bir chunk'ta verilen etiket
kalıcıdır ve erken verilmiş sınırda-bir karar tüm oturuma yayılabilir. Bu modül,
oturum bittikten (ya da yeterince ilerledikten) SONRA tüm zaman çizelgesine
bakarak yalnızca "tamamlanmış görüntüde" fark edilebilen hataları düzeltir.

Tasarım ilkeleri
----------------
* SAF: torch/model/IO yok. Aynı girdi → aynı çıktı (deterministik). Bu sayede
  GPU'suz, replay'siz, koşudan-koşuya oynamadan ölçülebilir.
* MUHAFAZAKÂR: her kural yalnızca süre + segment sayısı + ada sayısı + komşu
  mutabakatı kapılarının HEPSİ geçtiğinde ateşlenir. Şüphede olan dokunulmaz.
* YİNELEMELİ: bir birleştirme bir sonrakini mümkün kılabilir (passes).

Kurallar
--------
1. merge_small_islands      — tek seferlik küçük ada, iki yanında AYNI konuşmacı
2. merge_tiny_fragmented    — dağınık küçük profil, komşu oylamasıyla sahibine
3. fill_unknown_islands     — iki yanı AYNI konuşmacı olan kısa Unknown boşluğu

Kısa oturum davranışı
---------------------
Varsayılan eşik (30 sn) toplantı ölçeği içindir. Çok kısa bir oturumda (< ~1 dk)
GERÇEK konuşmacılar da 30 sn'nin altında kalır, dolayısıyla hepsi "minik aday"
sayılır; "hayaleti hayalete birleştirme" koruması devreye girer ve hiçbir
değişiklik yapılmaz. Yani refinement kısa oturumlarda güvenli bir no-op'tur —
zarar vermez, sadece işe yaramaz.

Segment biçimi: [{"speaker": str, "start": float, "end": float, "text": str}, ...]
zaman sırasına göre sıralı.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Gerçek konuşmacı olmayan sözde etiketler. Ada istatistiklerine girmezler ve
# birleştirme HEDEFİ olamazlar.
UNKNOWN_LABEL = "Unknown"
CALIBRATING_LABEL = "CALIBRATING"


def is_pseudo_label(label: str) -> bool:
    """Etiket gerçek bir konuşmacı mı, yoksa sözde/geçici bir işaret mi?"""
    if not label:
        return True
    if label in (UNKNOWN_LABEL, CALIBRATING_LABEL):
        return True
    return label.startswith("[Calibrating")


def segment_duration(segment: dict) -> float:
    return max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))


@dataclass
class RefinementConfig:
    """Kural eşikleri.

    Varsayılanlar AMI üzerinde ÖLÇÜLEREK seçildi (8 toplantı, 32 referans
    konuşmacı, scripts/refine_sweep.py):

        baseline           -> Conf 10.86 | DER 34.45 | cpWER 50.87 | 75 spk
        30/8/8/0.5 (bu)    -> Conf 10.19 | DER 33.78 | cpWER 49.59 | 55 spk

    Sıralama sistematikti: tüm dur=30 varyantları ilk 4 sırayı aldı,
    islands=8 > islands=5 ve share=0.5 > share=0.6 tutarlı çıktı. Daha küçük
    örneklemde (3 toplantı) görülen "dar tepe" davranışı 8 toplantıda kayboldu.
    """

    # 1) Tek seferlik küçük ada
    # NOT: taramada bu kuralın eşikleri ATIL çıktı (hiçbir değer sonucu
    # değiştirmedi) — asıl işi tiny_fragmented yapıyor.
    small_island_merge: bool = True
    small_island_max_duration: float = 5.0
    small_island_max_segments: int = 3

    # 2) Dağınık küçük profil (hayalet konuşmacı avcısı) — ASIL ETKİLİ KURAL
    tiny_fragmented_merge: bool = True
    tiny_fragmented_max_duration: float = 30.0   # ölçüldü: 30 > 20 > 15 > 10
    tiny_fragmented_max_segments: int = 8
    tiny_fragmented_min_islands: int = 2
    tiny_fragmented_max_islands: int = 8         # ölçüldü: 8 > 5
    tiny_fragmented_min_neighbor_share: float = 0.5  # ölçüldü: 0.5 > 0.6

    # 3) Unknown boşluk doldurma
    unknown_fill: bool = True
    unknown_fill_max_duration: float = 3.0
    unknown_fill_max_segments: int = 1

    # Yineleme sayısı. Ölçümde 3 kullanıldı; bir birleştirme bir sonrakini
    # mümkün kılabildiği için fazladan geçiş yararlı. Değişiklik kalmayınca
    # döngü erken çıkar (`if changed == 0: break`), yani maliyeti yok.
    passes: int = 3


@dataclass
class Island:
    """Ardışık, aynı etiketli segment bloğu."""
    start: int          # dahil
    end: int            # hariç
    label: str
    duration: float

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class LabelStats:
    duration: float = 0.0
    segments: int = 0
    islands: int = 0
    indexes: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Yapı çıkarma
# --------------------------------------------------------------------------- #
def build_islands(segments: list) -> list:
    """Ardışık aynı-etiketli segmentleri adalara böler."""
    islands = []
    i, n = 0, len(segments)
    while i < n:
        label = segments[i].get("speaker")
        j = i + 1
        while j < n and segments[j].get("speaker") == label:
            j += 1
        duration = sum(segment_duration(s) for s in segments[i:j])
        islands.append(Island(i, j, label, duration))
        i = j
    return islands


def label_stats(islands: list) -> dict:
    """Etiket başına toplam süre / segment / ada sayısı."""
    stats: dict = {}
    for isl in islands:
        if is_pseudo_label(isl.label):
            continue
        entry = stats.setdefault(isl.label, LabelStats())
        entry.duration += isl.duration
        entry.segments += isl.size
        entry.islands += 1
        entry.indexes.extend(range(isl.start, isl.end))
    return stats


def _prev_other_speaker(segments: list, before: int, label: str):
    """`before` indeksinden geriye doğru, `label` DIŞINDA ilk gerçek konuşmacı."""
    for k in range(before - 1, -1, -1):
        candidate = segments[k].get("speaker")
        if not is_pseudo_label(candidate) and candidate != label:
            return candidate
    return None


def _next_other_speaker(segments: list, at: int, label: str):
    """`at` indeksinden ileriye doğru, `label` DIŞINDA ilk gerçek konuşmacı."""
    for k in range(at, len(segments)):
        candidate = segments[k].get("speaker")
        if not is_pseudo_label(candidate) and candidate != label:
            return candidate
    return None


def _relabel(segments: list, indexes, new_label: str, source: str) -> int:
    """Verilen indeksleri yeni etikete taşır; izlenebilirlik için kaynak yazar."""
    changed = 0
    for idx in indexes:
        old = segments[idx].get("speaker")
        if old == new_label:
            continue
        segments[idx]["speaker"] = new_label
        segments[idx]["refined_from"] = old
        segments[idx]["refined_by"] = source
        changed += 1
    return changed


# --------------------------------------------------------------------------- #
# Kural 1 — tek seferlik küçük ada
# --------------------------------------------------------------------------- #
def merge_small_islands(segments: list, cfg: RefinementConfig) -> int:
    """Yalnızca BİR kez görünen, kısa bir etiketi; iki yanındaki AYNI konuşmacıya
    devreder. ("A ... [tek blip X] ... A"  →  hepsi A)
    """
    if not cfg.small_island_merge or len(segments) < 3:
        return 0

    islands = build_islands(segments)
    stats = label_stats(islands)

    # Kararları önce topla, sonra uygula (ada yapısı okunurken değişmesin).
    decisions = []
    for isl in islands:
        if is_pseudo_label(isl.label):
            continue
        entry = stats.get(isl.label)
        if entry is None or entry.islands != 1:
            continue
        if entry.duration > cfg.small_island_max_duration:
            continue
        if entry.segments > cfg.small_island_max_segments:
            continue
        prev_speaker = _prev_other_speaker(segments, isl.start, isl.label)
        next_speaker = _next_other_speaker(segments, isl.end, isl.label)
        if prev_speaker is None or prev_speaker != next_speaker:
            continue
        decisions.append((list(range(isl.start, isl.end)), prev_speaker))

    applied = 0
    for indexes, target in decisions:
        applied += _relabel(segments, indexes, target, "small_island_merge")
    return applied


# --------------------------------------------------------------------------- #
# Kural 2 — dağınık küçük profil (komşu oylaması)
# --------------------------------------------------------------------------- #
def merge_tiny_fragmented(segments: list, cfg: RefinementConfig) -> int:
    """Toplamı küçük ama birkaç adaya dağılmış profili, komşularının OYLADIĞI
    konuşmacıya devreder.

    Bu, "gerçek bir konuşmacının sesi küçük bir hayalet profile sızmış" durumunun
    düzeltmesidir: hayaletin her adasının önündeki/ardındaki konuşmacılar oy verir;
    net bir kazanan (beraberlik yok) ve yeterli oy payı varsa birleştirilir.
    Hedefin kendisi de "minik dağınık" adaysa birleştirme YAPILMAZ (hayalet →
    hayalet zinciri engellenir).
    """
    if not cfg.tiny_fragmented_merge or len(segments) < 3:
        return 0

    islands = build_islands(segments)
    stats = label_stats(islands)

    def _is_candidate(entry: LabelStats) -> bool:
        return (
            entry.duration <= cfg.tiny_fragmented_max_duration
            and entry.segments <= cfg.tiny_fragmented_max_segments
            and cfg.tiny_fragmented_min_islands <= entry.islands <= cfg.tiny_fragmented_max_islands
        )

    candidate_labels = {lbl for lbl, e in stats.items() if _is_candidate(e)}

    decisions = []
    for label in sorted(candidate_labels):
        entry = stats[label]

        votes: Counter = Counter()
        for isl in islands:
            if isl.label != label:
                continue
            prev_speaker = _prev_other_speaker(segments, isl.start, label)
            next_speaker = _next_other_speaker(segments, isl.end, label)
            if prev_speaker:
                votes[prev_speaker] += 1
            if next_speaker:
                votes[next_speaker] += 1

        if not votes:
            continue
        ranked = votes.most_common()
        # Beraberlik → karar verme.
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        target, target_votes = ranked[0]
        if target_votes / max(1, sum(votes.values())) < cfg.tiny_fragmented_min_neighbor_share:
            continue
        # Hayaleti başka bir hayalete devretme.
        if target in candidate_labels:
            continue
        decisions.append((list(entry.indexes), target))

    applied = 0
    for indexes, target in decisions:
        applied += _relabel(segments, indexes, target, "tiny_fragmented_merge")
    return applied


# --------------------------------------------------------------------------- #
# Kural 3 — Unknown boşluk doldurma
# --------------------------------------------------------------------------- #
def fill_unknown_islands(segments: list, cfg: RefinementConfig) -> int:
    """İki yanı AYNI gerçek konuşmacı olan kısa Unknown bloğunu o konuşmacıya verir.

    Yalnızca "Unknown" hedeflenir; CALIBRATING warm-up işareti olarak korunur
    (skorlama zaten onu dışlıyor).
    """
    if not cfg.unknown_fill or len(segments) < 3:
        return 0

    decisions = []
    i, n = 0, len(segments)
    while i < n:
        if segments[i].get("speaker") != UNKNOWN_LABEL:
            i += 1
            continue
        start = i
        while i < n and segments[i].get("speaker") == UNKNOWN_LABEL:
            i += 1
        end = i
        if start == 0 or end >= n:
            continue  # kenarlarda: iki yanlı kanıt yok
        prev_speaker = segments[start - 1].get("speaker")
        next_speaker = segments[end].get("speaker")
        if is_pseudo_label(prev_speaker) or prev_speaker != next_speaker:
            continue
        indexes = list(range(start, end))
        duration = sum(segment_duration(segments[k]) for k in indexes)
        if len(indexes) > cfg.unknown_fill_max_segments or duration > cfg.unknown_fill_max_duration:
            continue
        decisions.append((indexes, prev_speaker))

    applied = 0
    for indexes, target in decisions:
        applied += _relabel(segments, indexes, target, "unknown_fill")
    return applied


# --------------------------------------------------------------------------- #
# Orkestrasyon
# --------------------------------------------------------------------------- #
def refine_speakers(segments: list, cfg: RefinementConfig | None = None):
    """Tüm kuralları yineleyerek uygular.

    Args:
        segments: [{"speaker","start","end","text"}, ...] — zaman sıralı.
        cfg: eşikler (None → varsayılan).

    Returns:
        (refined_segments, stats) — girdi DEĞİŞTİRİLMEZ (kopya üzerinde çalışılır).
        stats: {"small_island_merge": n, "tiny_fragmented_merge": n,
                "unknown_fill": n, "passes_run": n, "total": n}
    """
    cfg = cfg or RefinementConfig()
    refined = [dict(s) for s in segments]
    totals = Counter()

    passes_run = 0
    for _ in range(max(0, cfg.passes)):
        passes_run += 1
        changed = 0
        n1 = merge_small_islands(refined, cfg)
        n2 = merge_tiny_fragmented(refined, cfg)
        n3 = fill_unknown_islands(refined, cfg)
        totals["small_island_merge"] += n1
        totals["tiny_fragmented_merge"] += n2
        totals["unknown_fill"] += n3
        changed = n1 + n2 + n3
        if changed == 0:
            break

    stats = dict(totals)
    stats["passes_run"] = passes_run
    stats["total"] = sum(totals.values())
    return refined, stats


def speaker_summary(segments: list) -> dict:
    """Etiket → toplam süre (teşhis/rapor için). Sözde etiketler de dahildir."""
    out: dict = {}
    for seg in segments:
        label = seg.get("speaker")
        out[label] = out.get(label, 0.0) + segment_duration(seg)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
