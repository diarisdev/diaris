"""cpWER — concatenated minimum-permutation Word Error Rate.

Konuşmacı-bazlı ASR metriği (CHiME-6/7/8 birincil Track-2 metriği). Hipotez
konuşmacılarını referans konuşmacılarına optimal permütasyonla (Hungarian)
eşler ve toplam kelime hatasını minimize eder.

Özellikler (eski tests/chime6_benchmark.compute_cpwer'dan merkeze taşındı):
  * Speaker pruning: far-field'da çok sayıda küçük gürültü konuşmacısı için pahalı
    Levenshtein'i atlar (top-8 + >=50 kelime "umut vaat eden" sayılır).
  * Eşleşen çiftlerin detayı KESİN Levenshtein ile yeniden hesaplanır (pruning
    yaklaşık maliyet kullanmış olsa bile cpWER ve döküm kesin kalır).
  * use_meeteval=True ise headline cpWER meeteval'in referans implementasyonuyla
    değiştirilir (literatürle birebir karşılaştırma için).

Döndürülen cpWER bir ORANDIR (0.0–1.0+); yüzde için *100.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .edit_distance import levenshtein_sid
from .wer import normalize_text


@dataclass
class CpwerResult:
    cpwer: float                                  # oran (0.0–1.0+)
    mapping: dict = field(default_factory=dict)   # {ref_speaker: hyp_speaker}
    details: dict = field(default_factory=dict)   # {ref_speaker: {sub,ins,del,ref_count,hyp_count}}
    total_ref_words: int = 0
    errors: int = 0


def _assign(cost_matrix, N):
    """Toplam maliyeti minimize eden permütasyonu döndürür (best_perm listesi)."""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_perm = [0] * N
        for r, c in zip(row_ind, col_ind):
            best_perm[r] = c
        return best_perm
    except ImportError:
        pass

    # scipy yoksa: küçük N'de permütasyon, büyük N'de açgözlü
    import itertools
    if N <= 8:
        best_cost = float("inf")
        best_perm = list(range(N))
        for perm in itertools.permutations(range(N)):
            c = sum(cost_matrix[i][perm[i]] for i in range(N))
            if c < best_cost:
                best_cost = c
                best_perm = list(perm)
        return best_perm

    best_perm = [-1] * N
    matched_cols = set()
    for i in range(N):
        min_c, min_val = -1, float("inf")
        for j in range(N):
            if j not in matched_cols and cost_matrix[i][j] < min_val:
                min_val, min_c = cost_matrix[i][j], j
        best_perm[i] = min_c
        matched_cols.add(min_c)
    return best_perm


def cpwer_from_speaker_texts(ref_texts: dict, hyp_texts: dict,
                             use_meeteval: bool = False) -> CpwerResult:
    """ref_texts/hyp_texts: {konuşmacı: birleştirilmiş_metin} (zaten normalize)."""
    ref_spks = list(ref_texts.keys())
    hyp_spks = list(hyp_texts.keys())

    if not ref_spks and not hyp_spks:
        return CpwerResult(0.0)

    K, M = len(ref_spks), len(hyp_spks)
    N = max(K, M)
    ref_list = ref_spks + [None] * (N - K)
    hyp_list = hyp_spks + [None] * (N - M)

    # Performans pruning: küçük gürültü konuşmacıları için pahalı DP'yi atla
    hyp_word_counts = {s: len(hyp_texts[s].split()) for s in hyp_spks}
    sorted_hyp = sorted(hyp_spks, key=lambda s: hyp_word_counts[s], reverse=True)
    promising = set()
    for idx, s in enumerate(sorted_hyp):
        if idx < 8 or hyp_word_counts[s] >= 50:
            promising.add(s)

    cost = [[0] * N for _ in range(N)]
    for i in range(N):
        ref_words = (ref_texts[ref_list[i]] if ref_list[i] else "").split()
        for j in range(N):
            hs = hyp_list[j]
            hyp_words = (hyp_texts[hs] if hs else "").split()
            if hs is not None and hs not in promising:
                cost[i][j] = len(ref_words) + len(hyp_words)   # üst sınır maliyet
            else:
                su, ins, de = levenshtein_sid(ref_words, hyp_words)
                cost[i][j] = su + ins + de

    best_perm = _assign(cost, N)

    # Eşleşen çiftler için detay + hatayı KESİN olarak (yeniden) hesapla
    mapping, details = {}, {}
    total_ref_words = sum(len(ref_texts[s].split()) for s in ref_spks)
    matched_errors = 0
    for i in range(N):
        rs = ref_list[i]
        hs = hyp_list[best_perm[i]]
        if rs or hs:
            r_name = rs if rs else "[None]"
            h_name = hs if hs else "[No Match]"
            mapping[r_name] = h_name
            ref_words = (ref_texts[rs] if rs else "").split()
            hyp_words = (hyp_texts[hs] if hs else "").split()
            su, ins, de = levenshtein_sid(ref_words, hyp_words)
            matched_errors += su + ins + de
            details[r_name] = {"sub": su, "ins": ins, "del": de,
                               "ref_count": len(ref_words), "hyp_count": len(hyp_words)}

    cpwer = matched_errors / total_ref_words if total_ref_words > 0 else 0.0

    # Opsiyonel: meeteval referans implementasyonuyla headline cpWER'i değiştir
    if use_meeteval and any(ref_texts.values()):
        try:
            from meeteval.wer import cp_word_error_rate
            ref_map = {s: ref_texts[s] for s in ref_spks}
            hyp_map = {s: hyp_texts[s] for s in hyp_spks} or {"dummy": ""}
            er = cp_word_error_rate(ref_map, hyp_map)
            cpwer = float(er.error_rate)
        except ImportError:
            print("WARNING: --meeteval istendi ama `meeteval` kurulu değil "
                  "(pip install meeteval). Dahili cpWER kullanılıyor.")
        except Exception as e:
            print(f"WARNING: meeteval cpWER başarısız ({e}). Dahili cpWER kullanılıyor.")

    return CpwerResult(cpwer, mapping, details, total_ref_words, matched_errors)


def _group_segments(segments, normalize: bool, skip_calibrating: bool) -> dict:
    """[{speaker,start,end,text}, ...] -> {konuşmacı: birleştirilmiş_normalize_metin}."""
    by_spk: dict[str, list] = {}
    for seg in segments:
        spk = seg["speaker"]
        if skip_calibrating and "Calibrating" in spk:
            continue
        by_spk.setdefault(spk, []).append(seg)

    out = {}
    for spk, segs in by_spk.items():
        segs = sorted(segs, key=lambda x: x.get("start", 0.0))
        text = " ".join((normalize_text(s["text"]) if normalize else s["text"]) for s in segs)
        out[spk] = normalize_text(text) if normalize else text
    return out


def cpwer_from_segments(ref_segments, hyp_segments, normalize: bool = True,
                        use_meeteval: bool = False) -> CpwerResult:
    """Segment listelerinden cpWER. Hipotezde 'Calibrating' etiketleri yok sayılır."""
    ref_texts = _group_segments(ref_segments, normalize, skip_calibrating=False)
    hyp_texts = _group_segments(hyp_segments, normalize, skip_calibrating=True)
    return cpwer_from_speaker_texts(ref_texts, hyp_texts, use_meeteval=use_meeteval)
