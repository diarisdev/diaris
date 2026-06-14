#!/usr/bin/env python3
"""Faz 4 — AMI metrik hesaplama: DER / WER / cpWER.

`ami_hyp` (replay.py çıktısı) hipotezlerini `ami_refs` (Faz 1) referanslarına karşı
skorlar. Toplantı-başına ve toplam (hataları toplayıp toplam birime bölen global)
değerleri raporlar.

Çalıştırma (paket kökünün BİR ÜSTÜNDEN):
    python -m src.eval.metrics
    python -m src.eval.metrics --refs ./tests/ami_data/ami_refs --hyp ./tests/ami_data/ami_hyp
    python -m src.eval.metrics --only IS1009a

Metrikler
---------
* DER  : pyannote.metrics, collar=0.25s, full overlap (skip_overlap=False).
         Hipotezdeki 'Unknown'/'CALIBRATING' etiketleri DÜRÜSTÇE dahildir
         (Faz 3: warm-up ve atanamayan sesler hata olarak sayılır).
* WER  : konuşmacıdan bağımsız, normalize edilmiş düz metin (jiwer).
* cpWER: konuşmacı-bazlı; ref↔hyp konuşmacı eşlemesi optimal permütasyonla
         (Hungarian) çözülür; eşleşmeyen hyp konuşmacıların metni insertion sayılır.
"""

from __future__ import annotations

import warnings as _warnings
for _msg in (r".*uem.*", r".*Mean of empty slice.*", r".*invalid value encountered.*"):
    _warnings.filterwarnings("ignore", message=_msg)

import argparse
import json
import re
from pathlib import Path


# --------------------------------------------------------------------------- #
# Metin normalizasyonu (WER/cpWER için adil karşılaştırma)
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Küçük harf, noktalama temizliği (kesme işareti korunur), boşluk sadeleştirme."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)   # kesme hariç noktalama → boşluk
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Word-level edit metrikleri — rapidfuzz (C++) öncelikli (chime6 yöntemi).
# Toplam edit mesafesi yöntemden bağımsız AYNI; rapidfuzz sadece çok daha hızlı
# (uzun birleştirilmiş metinlerde jiwer dakikalar alabiliyordu). Yedek: jiwer → DP.
# Kelime LİSTELERİ üzerinde çalışır (bir kez split → tekrar tekrar bölme yok).
# --------------------------------------------------------------------------- #
def _word_edits(ref_tokens: list, hyp_tokens: list) -> tuple[int, int, int]:
    """(substitutions, insertions, deletions) — kelime listeleri üzerinde."""
    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0:
        return 0, m, 0
    if m == 0:
        return 0, 0, n
    try:
        from rapidfuzz.distance import Levenshtein
        subs = ins = dels = 0
        for op in Levenshtein.editops(ref_tokens, hyp_tokens):
            if op.tag == "replace":
                subs += 1
            elif op.tag == "insert":
                ins += 1
            elif op.tag == "delete":
                dels += 1
        return subs, ins, dels
    except ImportError:
        pass
    try:
        import jiwer
        out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
        return out.substitutions, out.insertions, out.deletions
    except Exception:
        pass
    # Saf DP yedek (S/I/D ayrımıyla), O(m) bellek
    prev = [(j, 0, j, 0) for j in range(m + 1)]   # (cost, sub, ins, del)
    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cur[j] = prev[j - 1]
            else:
                sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3])
                insop = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2] + 1, cur[j - 1][3])
                dl = (prev[j][0] + 1, prev[j][1], prev[j][2], prev[j][3] + 1)
                cur[j] = min(sub, insop, dl, key=lambda x: x[0])
        prev = cur
    return prev[m][1], prev[m][2], prev[m][3]


def _word_distance(ref_tokens: list, hyp_tokens: list) -> int:
    """Toplam edit mesafesi (S+I+D) — cpWER maliyet matrisi için en hızlı yol
    (ops ayrımına gerek yok; rapidfuzz.distance saf C++)."""
    if not ref_tokens:
        return len(hyp_tokens)
    if not hyp_tokens:
        return len(ref_tokens)
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(ref_tokens, hyp_tokens)
    except ImportError:
        pass
    s, i, d = _word_edits(ref_tokens, hyp_tokens)
    return s + i + d


# --------------------------------------------------------------------------- #
# WER (konuşmacıdan bağımsız)
# --------------------------------------------------------------------------- #
def compute_wer(ref_flat: str, hyp_flat: str) -> dict:
    ref = normalize_text(ref_flat).split()
    hyp = normalize_text(hyp_flat).split()
    s, i, d = _word_edits(ref, hyp)
    return {"errors": s + i + d, "ref_words": len(ref),
            "sub": s, "ins": i, "del": d,
            "wer": (s + i + d) / max(len(ref), 1) * 100.0}


# --------------------------------------------------------------------------- #
# cpWER (konuşmacı-bazlı, optimal eşleme)
# --------------------------------------------------------------------------- #
def compute_cpwer(ref_speakers: dict, hyp_speakers: dict) -> dict:
    """ref/hyp: {konuşmacı: birleştirilmiş_metin}. Optimal ref↔hyp eşlemesiyle
    toplam kelime hatasını minimize eder. cpWER = toplam hata / toplam ref kelime."""
    ref_spk = [s for s in ref_speakers if normalize_text(ref_speakers[s])]
    hyp_spk = list(hyp_speakers.keys())

    ref_tok = {s: normalize_text(ref_speakers[s]).split() for s in ref_spk}
    hyp_tok = {s: normalize_text(hyp_speakers[s]).split() for s in hyp_spk}
    total_ref_words = sum(len(t) for t in ref_tok.values())

    if not ref_spk:
        ins = sum(len(t) for t in hyp_tok.values())
        return {"cpwer": 0.0 if ins == 0 else 100.0, "errors": ins,
                "ref_words": 0, "mapping": {}}

    # Maliyet matrisi: satır=ref, sütun=hyp (kareye pad). Maliyet = edit mesafesi
    # (rapidfuzz, C++ — eski jiwer'a göre çok daha hızlı; sayısal sonuç aynı).
    pad_hyp = list(hyp_spk)
    while len(pad_hyp) < len(ref_spk):
        pad_hyp.append(f"__EMPTY_{len(pad_hyp)}__")

    n, m = len(ref_spk), len(pad_hyp)
    cost = [[0] * m for _ in range(n)]
    for i, rs in enumerate(ref_spk):
        rt = ref_tok[rs]
        for j, hs in enumerate(pad_hyp):
            cost[i][j] = _word_distance(rt, hyp_tok.get(hs, []))

    # Hungarian (scipy) → yoksa permütasyon fallback
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        r_ind, c_ind = linear_sum_assignment(np.array(cost))
        pairs = list(zip(r_ind.tolist(), c_ind.tolist()))
    except Exception:
        import itertools
        best, pairs = None, None
        for perm in itertools.permutations(range(m), n):
            tot = sum(cost[i][perm[i]] for i in range(n))
            if best is None or tot < best:
                best, pairs = tot, list(enumerate(perm))

    mapping = {ref_spk[i]: pad_hyp[j] for i, j in pairs}
    mapped_errors = sum(cost[i][j] for i, j in pairs)

    # Eşleşmeyen (haritada olmayan) gerçek hyp konuşmacılarının metni = insertion
    mapped_hyp = {pad_hyp[j] for _, j in pairs}
    unmapped_ins = sum(len(hyp_tok[h]) for h in hyp_spk if h not in mapped_hyp)

    errors = mapped_errors + unmapped_ins
    return {"cpwer": errors / max(total_ref_words, 1) * 100.0,
            "errors": errors, "ref_words": total_ref_words,
            "mapping": {k: v for k, v in mapping.items() if not v.startswith("__EMPTY")}}


# --------------------------------------------------------------------------- #
# DER (pyannote.metrics)
# --------------------------------------------------------------------------- #
def _load_rttm(path: Path):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=path.stem)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        if dur <= 0:
            continue
        ann[Segment(start, start + dur)] = spk
    return ann


def compute_der(ref_rttm: Path, hyp_rttm: Path, collar: float = 0.25) -> dict | None:
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        return None
    ref = _load_rttm(ref_rttm)
    hyp = _load_rttm(hyp_rttm)

    def _run(skip_overlap: bool) -> dict:
        m = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        der = m(ref, hyp)
        comp = m(ref, hyp, detailed=True)
        total = comp.get("total", 0.0) or 1e-8
        return {"der": der * 100.0,
                "false_alarm": comp.get("false alarm", 0.0) / total * 100.0,
                "missed": comp.get("missed detection", 0.0) / total * 100.0,
                "confusion": comp.get("confusion", 0.0) / total * 100.0}

    full = _run(False)               # full = örtüşme dahil (standart)
    noov = _run(True)                # örtüşme hariç → kaçınılmaz overlap cezasını ayırır
    full["der_no_overlap"] = noov["der"]
    full["missed_no_overlap"] = noov["missed"]
    full["confusion_no_overlap"] = noov["confusion"]

    # JER (her konuşmacı eşit ağırlık) + Purity/Coverage (kümeleme kalitesi).
    # Sürüm/imza farkına karşı her metrik bağımsız, kademeli-kwargs ile;
    # eksikse zarif şekilde None'a düşer (DER/WER/cpWER'i bozmaz).
    full["jer"] = full["purity"] = full["coverage"] = None
    try:
        from pyannote.metrics.diarization import (
            JaccardErrorRate, DiarizationPurity, DiarizationCoverage,
        )

        def _score(cls):
            for kw in ({"collar": collar, "skip_overlap": False}, {"collar": collar}, {}):
                try:
                    return cls(**kw)(ref, hyp) * 100.0
                except TypeError:
                    continue
            return None

        full["jer"] = _score(JaccardErrorRate)         # konuşmacı-eşit hata
        full["purity"] = _score(DiarizationPurity)     # hyp saflığı (collapse → düşer)
        full["coverage"] = _score(DiarizationCoverage) # ref kapsama (over-segmentation → düşer)
    except Exception:
        pass
    return full


# --------------------------------------------------------------------------- #
def _evaluate_meeting(mid: str, refs: Path, hyp: Path, collar: float) -> dict | None:
    ref_tr_path = refs / "transcripts" / f"{mid}.json"
    hyp_tr_path = hyp / "transcripts" / f"{mid}.json"
    ref_rttm = refs / "rttm" / f"{mid}.rttm"
    hyp_rttm = hyp / "rttm" / f"{mid}.rttm"
    if not (ref_tr_path.exists() and hyp_tr_path.exists()):
        return None

    ref_tr = json.loads(ref_tr_path.read_text(encoding="utf-8"))
    hyp_tr = json.loads(hyp_tr_path.read_text(encoding="utf-8"))

    wer = compute_wer(ref_tr.get("flat", ""), hyp_tr.get("flat", ""))
    cpwer = compute_cpwer(ref_tr.get("speakers", {}), hyp_tr.get("speakers", {}))
    der = None
    if ref_rttm.exists() and hyp_rttm.exists():
        der = compute_der(ref_rttm, hyp_rttm, collar)

    return {"meeting_id": mid, "wer": wer, "cpwer": cpwer, "der": der,
            "ref_speakers": len(ref_tr.get("speakers", {})),
            "hyp_speakers": len([s for s in hyp_tr.get("speakers", {})
                                 if s not in ("Unknown", "CALIBRATING")])}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_refs = project_root / "tests" / "ami_data" / "ami_refs"
    default_hyp = project_root / "tests" / "ami_data" / "ami_hyp"

    ap = argparse.ArgumentParser(description="AMI DER/WER/cpWER metrikleri (Faz 4).")
    ap.add_argument("--refs", type=Path, default=default_refs)
    ap.add_argument("--hyp", type=Path, default=default_hyp)
    ap.add_argument("--only", nargs="*", default=None, help="Sadece bu toplantı(lar).")
    ap.add_argument("--collar", type=float, default=0.25, help="DER collar (s).")
    ap.add_argument("--save", type=Path, default=None,
                    help="Sonuçları JSON kaydet (baseline karşılaştırması için).")
    args = ap.parse_args()

    # İşlenecek toplantılar: hyp transcript'i olanlar
    hyp_tr_dir = args.hyp / "transcripts"
    if not hyp_tr_dir.is_dir():
        raise SystemExit(f"Hyp transkript dizini yok: {hyp_tr_dir.resolve()}")
    meetings = sorted(p.stem for p in hyp_tr_dir.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        meetings = [m for m in meetings if m in wanted]
    if not meetings:
        raise SystemExit("İşlenecek toplantı bulunamadı.")

    def _c(x, w=5):
        return f"{x:>{w}.1f}" if x is not None else f"{'n/a':>{w}}"

    results = []
    print(f"\n{'Meeting':<10} {'DER':>5} {'JER':>5} {'Miss':>5} {'Conf':>5} "
          f"{'Pur':>5} {'Cov':>5} {'WER':>5} {'cpWER':>6} {'spk':>6}")
    print("  (JER=konuşmacı-eşit hata; Pur=hyp saflığı↑iyi, Cov=ref kapsama↑iyi; "
          "over-count→Cov↓, collapse→Pur↓)")
    print("-" * 80)
    for mid in meetings:
        r = _evaluate_meeting(mid, args.refs, args.hyp, args.collar)
        if r is None:
            print(f"{mid:<10}  (referans/hyp eksik, atlandı)")
            continue
        results.append(r)
        d = r["der"]
        if d:
            print(f"{mid:<10} {d['der']:>5.1f} {_c(d['jer'])} {d['missed']:>5.1f} "
                  f"{d['confusion']:>5.1f} {_c(d['purity'])} {_c(d['coverage'])} "
                  f"{r['wer']['wer']:>5.1f} {r['cpwer']['cpwer']:>6.1f} "
                  f"{r['hyp_speakers']:>3}/{r['ref_speakers']}")
        else:
            print(f"{mid:<10}   (DER yok)   WER {r['wer']['wer']:.1f}  cpWER {r['cpwer']['cpwer']:.1f}")

    if not results:
        return

    # Toplam (global): hataları toplayıp toplam birime böl
    tot_wer_e = sum(r["wer"]["errors"] for r in results)
    tot_wer_w = sum(r["wer"]["ref_words"] for r in results)
    tot_cp_e = sum(r["cpwer"]["errors"] for r in results)
    tot_cp_w = sum(r["cpwer"]["ref_words"] for r in results)
    ders = [r["der"]["der"] for r in results if r["der"]]
    ders_noov = [r["der"]["der_no_overlap"] for r in results if r["der"]]
    jers = [r["der"]["jer"] for r in results if r["der"] and r["der"]["jer"] is not None]
    purs = [r["der"]["purity"] for r in results if r["der"] and r["der"]["purity"] is not None]
    covs = [r["der"]["coverage"] for r in results if r["der"] and r["der"]["coverage"] is not None]
    _avg = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")

    print("-" * 80)
    print(f"{'TOPLAM':<10} "
          f"DER {_avg(ders):.2f} (örtüşme hariç {_avg(ders_noov):.2f})   "
          f"WER {tot_wer_e / max(tot_wer_w, 1) * 100:.2f}   "
          f"cpWER {tot_cp_e / max(tot_cp_w, 1) * 100:.2f}")
    if jers:
        print(f"{'':<10} JER {_avg(jers):.2f}   Purity {_avg(purs):.2f}   "
              f"Coverage {_avg(covs):.2f}")
    if args.save:
        report = {
            "aggregate": {
                "num_meetings": len(results),
                "der_avg": round(_avg(ders), 2) if ders else None,
                "der_avg_no_overlap": round(_avg(ders_noov), 2) if ders_noov else None,
                "jer_avg": round(_avg(jers), 2) if jers else None,
                "purity_avg": round(_avg(purs), 2) if purs else None,
                "coverage_avg": round(_avg(covs), 2) if covs else None,
                "wer": round(tot_wer_e / max(tot_wer_w, 1) * 100, 2),
                "cpwer": round(tot_cp_e / max(tot_cp_w, 1) * 100, 2),
            },
            "meetings": [
                {
                    "meeting_id": r["meeting_id"],
                    "der": round(r["der"]["der"], 2) if r["der"] else None,
                    "der_false_alarm": round(r["der"]["false_alarm"], 2) if r["der"] else None,
                    "der_missed": round(r["der"]["missed"], 2) if r["der"] else None,
                    "der_confusion": round(r["der"]["confusion"], 2) if r["der"] else None,
                    "der_no_overlap": round(r["der"]["der_no_overlap"], 2) if r["der"] else None,
                    "der_missed_no_overlap": round(r["der"]["missed_no_overlap"], 2) if r["der"] else None,
                    "jer": round(r["der"]["jer"], 2) if r["der"] and r["der"]["jer"] is not None else None,
                    "purity": round(r["der"]["purity"], 2) if r["der"] and r["der"]["purity"] is not None else None,
                    "coverage": round(r["der"]["coverage"], 2) if r["der"] and r["der"]["coverage"] is not None else None,
                    "wer": round(r["wer"]["wer"], 2),
                    "cpwer": round(r["cpwer"]["cpwer"], 2),
                    "hyp_speakers": r["hyp_speakers"],
                    "ref_speakers": r["ref_speakers"],
                }
                for r in results
            ],
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline kaydedildi: {args.save.resolve()}")

    print(f"\n{len(results)} toplantı işlendi. "
          f"DER: meeting ortalaması; WER/cpWER: global (toplam hata / toplam kelime).\n")


if __name__ == "__main__":
    main()
