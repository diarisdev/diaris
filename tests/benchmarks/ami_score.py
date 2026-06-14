#!/usr/bin/env python3
"""AMI skorlama — replay çıktısını (ami_replay.py) referanslara karşı DER/WER/cpWER.

`ami_hyp` (ami_replay.py çıktısı) hipotezlerini `ami_refs` referanslarına karşı
skorlar. Toplantı-başına ve toplam değerleri raporlar.

Metrik MATEMATİĞİ tek kaynaktan gelir: tests.metrics (WER/cpWER/DER). Bu dosya
sadece AMI dosya düzenini okuyup tabloyu üretir.

Çalıştırma (proje kökünden):
    python -m tests.benchmarks.ami_score
    python -m tests.benchmarks.ami_score --refs ./datasets/ami/ami_refs --hyp ./datasets/ami/ami_hyp
    python -m tests.benchmarks.ami_score --only IS1009a

Metrikler
---------
* DER  : pyannote.metrics, collar=0.25s, örtüşme dahil (+ örtüşme-hariç varyant).
* WER  : konuşmacıdan bağımsız düz metin.
* cpWER: konuşmacı-bazlı optimal eşleme (Hungarian).
"""

from __future__ import annotations

import warnings as _warnings
for _msg in (r".*uem.*", r".*Mean of empty slice.*", r".*invalid value encountered.*"):
    _warnings.filterwarnings("ignore", message=_msg)

import argparse
import json
import sys
from pathlib import Path

# --- Proje kökünü path'e ekle (python -m tests.benchmarks.ami_score) ---
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.metrics import (
    normalize_text,
    compute_wer as _wer,
    cpwer_from_speaker_texts as _cpwer,
    compute_der as _der,
    load_rttm,
)


# --------------------------------------------------------------------------- #
# tests.metrics etrafında ince sarmalayıcılar (yüzde + AMI'nin beklediği dict şekli)
# --------------------------------------------------------------------------- #
def compute_wer(ref_flat: str, hyp_flat: str) -> dict:
    r = _wer(ref_flat, hyp_flat)   # normalize=True
    return {"errors": r.errors, "ref_words": r.ref_words,
            "sub": r.substitutions, "ins": r.insertions, "del": r.deletions,
            "wer": r.wer * 100.0}


def compute_cpwer(ref_speakers: dict, hyp_speakers: dict) -> dict:
    ref_n = {s: normalize_text(t) for s, t in ref_speakers.items()}
    hyp_n = {s: normalize_text(t) for s, t in hyp_speakers.items()}
    res = _cpwer(ref_n, hyp_n)
    return {"cpwer": res.cpwer * 100.0, "errors": res.errors,
            "ref_words": res.total_ref_words, "mapping": res.mapping}


def compute_der(ref_rttm: Path, hyp_rttm: Path, collar: float = 0.25) -> dict | None:
    res = _der(load_rttm(ref_rttm), load_rttm(hyp_rttm),
               collar=collar, with_overlap_split=True)
    if res is None:
        return None
    return {
        "der": res.der * 100.0,
        "false_alarm": res.false_alarm * 100.0,
        "missed": res.missed * 100.0,
        "confusion": res.confusion * 100.0,
        "der_no_overlap": (res.der_no_overlap or 0.0) * 100.0,
        "missed_no_overlap": (res.missed_no_overlap or 0.0) * 100.0,
        "confusion_no_overlap": (res.confusion_no_overlap or 0.0) * 100.0,
    }


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
    default_refs = project_root / "datasets" / "ami" / "ami_refs"
    default_hyp = project_root / "datasets" / "ami" / "ami_hyp"

    ap = argparse.ArgumentParser(description="AMI DER/WER/cpWER metrikleri.")
    ap.add_argument("--refs", type=Path, default=default_refs)
    ap.add_argument("--hyp", type=Path, default=default_hyp)
    ap.add_argument("--only", nargs="*", default=None, help="Sadece bu toplantı(lar).")
    ap.add_argument("--collar", type=float, default=0.25, help="DER collar (s).")
    ap.add_argument("--save", type=Path, default=None,
                    help="Sonuçları JSON kaydet (baseline karşılaştırması için).")
    args = ap.parse_args()

    hyp_tr_dir = args.hyp / "transcripts"
    if not hyp_tr_dir.is_dir():
        raise SystemExit(f"Hyp transkript dizini yok: {hyp_tr_dir.resolve()}")
    meetings = sorted(p.stem for p in hyp_tr_dir.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        meetings = [m for m in meetings if m in wanted]
    if not meetings:
        raise SystemExit("İşlenecek toplantı bulunamadı.")

    results = []
    print(f"\n{'Meeting':<10} {'DER':>6} {'Miss':>6} {'Conf':>6} | "
          f"{'DER-ov':>6} {'Miss-ov':>7} | {'WER':>6} {'cpWER':>7} {'spk':>6}")
    print("  (-ov = örtüşme HARİÇ; Miss düşüşü = kaçınılmaz overlap cezası)")
    print("-" * 78)
    for mid in meetings:
        r = _evaluate_meeting(mid, args.refs, args.hyp, args.collar)
        if r is None:
            print(f"{mid:<10}  (referans/hyp eksik, atlandı)")
            continue
        results.append(r)
        d = r["der"]
        if d:
            print(f"{mid:<10} {d['der']:>6.1f} {d['missed']:>6.1f} {d['confusion']:>6.1f} | "
                  f"{d['der_no_overlap']:>6.1f} {d['missed_no_overlap']:>7.1f} | "
                  f"{r['wer']['wer']:>6.1f} {r['cpwer']['cpwer']:>7.1f} "
                  f"{r['hyp_speakers']:>2}/{r['ref_speakers']}")
        else:
            print(f"{mid:<10}   (DER yok)   WER {r['wer']['wer']:.1f}  cpWER {r['cpwer']['cpwer']:.1f}")

    if not results:
        return

    tot_wer_e = sum(r["wer"]["errors"] for r in results)
    tot_wer_w = sum(r["wer"]["ref_words"] for r in results)
    tot_cp_e = sum(r["cpwer"]["errors"] for r in results)
    tot_cp_w = sum(r["cpwer"]["ref_words"] for r in results)
    ders = [r["der"]["der"] for r in results if r["der"]]
    ders_noov = [r["der"]["der_no_overlap"] for r in results if r["der"]]

    print("-" * 78)
    print(f"{'TOPLAM':<10} "
          f"DER {(sum(ders) / len(ders)) if ders else float('nan'):.2f} "
          f"(örtüşme hariç {(sum(ders_noov) / len(ders_noov)) if ders_noov else float('nan'):.2f})   "
          f"WER {tot_wer_e / max(tot_wer_w, 1) * 100:.2f}   "
          f"cpWER {tot_cp_e / max(tot_cp_w, 1) * 100:.2f}")
    if args.save:
        report = {
            "aggregate": {
                "num_meetings": len(results),
                "der_avg": round(sum(ders) / len(ders), 2) if ders else None,
                "der_avg_no_overlap": round(sum(ders_noov) / len(ders_noov), 2) if ders_noov else None,
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
