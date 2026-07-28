"""Karar tipine göre doğruluk analizi — konuşmacı karar kuralının nerede hata yaptığı.

SORU: SpeakerTracker'ın verdiği kararların hangileri yanlış? Düşük güvenli
kararlar (belirsiz margin, kısa ses yapıştırması) gerçekten daha mı kötü?

Bu, "karar kuralını iyileştirmek ne kazandırır" sorusunun ölçülmüş cevabıdır:
* Düşük güvenli kararlar çoğunlukla YANLIŞSA → güven-farkındalıklı bir kural
  (kalibre posterior) alacak ciddi bir pay var.
* Çoğunlukla DOĞRUYSA → mevcut kaba kurallar zaten iyi iş çıkarıyor; karar
  kuralına yatırım yapmak yerine başka kalemlere (over-count, miss) bakılmalı.

Kullanım:
    python -m tests.benchmarks.ami_replay --only IS1009a --trace-embeddings
    python scripts/analyze_decision_accuracy.py datasets/ami/ami_hyp/traces/IS1009a.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.embedding_trace import load_trace  # noqa: E402
from tests.benchmarks.trace_alignment import (  # noqa: E402
    build_label_mapping,
    collect_decisions,
    load_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Karar tipine göre doğruluk analizi.")
    parser.add_argument("trace", type=Path, help="ami_replay --trace-embeddings çıktısı (.npz)")
    parser.add_argument("--refs", type=Path,
                        default=PROJECT_ROOT / "datasets" / "ami" / "ami_refs" / "rttm",
                        help="Referans RTTM dizini.")
    parser.add_argument("--meeting", default=None,
                        help="Toplantı kimliği (verilmezse iz dosya adından çıkarılır).")
    args = parser.parse_args()

    meeting = args.meeting or args.trace.stem
    rttm_path = args.refs / f"{meeting}.rttm"
    if not rttm_path.exists():
        raise SystemExit(f"Referans bulunamadı: {rttm_path}")

    trace = load_trace(args.trace)
    reference = load_reference(rttm_path)
    decisions = collect_decisions(trace, reference)

    if not decisions:
        raise SystemExit(
            "İzde turn zaman aralığı yok. Bu analiz, span kaydı eklendikten SONRA "
            "üretilmiş bir iz gerektirir — replay'i yeniden çalıştırın."
        )

    mapping = build_label_mapping(decisions)

    stats = defaultdict(lambda: {"n": 0, "wrong": 0, "unknown": 0, "sec": 0.0, "wrong_sec": 0.0})
    total = Counter()
    for decision in decisions:
        bucket = stats[decision["decision"]]
        bucket["n"] += 1
        bucket["sec"] += decision["duration"]
        assigned = decision["assigned"]
        if not assigned or assigned == "Unknown":
            bucket["unknown"] += 1
            total["unknown"] += 1
            continue
        predicted_ref = mapping.get(assigned)
        if decision["reference"] is None:
            continue
        if predicted_ref != decision["reference"]:
            bucket["wrong"] += 1
            bucket["wrong_sec"] += decision["duration"]
            total["wrong"] += 1
        else:
            total["right"] += 1

    print(f"\n=== {meeting} — karar tipine göre doğruluk ===")
    # ASCII-only çıktı: Windows konsolu (cp1254) '->' dışındaki ok karakterlerini
    # kodlayamıyor ve UnicodeEncodeError ile scripti düşürüyor.
    print(f"    (hipotez -> referans eslemesi: {len(mapping)} etiket, "
          f"{len(set(s for _, _, s in reference))} referans konusmaci)\n")
    header = f"{'karar tipi':22} {'adet':>6} {'yanlış':>7} {'oran':>7} {'süre':>9} {'yanlış süre':>12}"
    print(header)
    print("-" * len(header))

    for name, bucket in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        scored = bucket["n"] - bucket["unknown"]
        rate = (bucket["wrong"] / scored * 100) if scored else 0.0
        print(f"{name:22} {bucket['n']:6d} {bucket['wrong']:7d} {rate:6.1f}% "
              f"{bucket['sec']:8.0f}s {bucket['wrong_sec']:11.0f}s")

    scored_total = total["right"] + total["wrong"]
    print("-" * len(header))
    if scored_total:
        print(f"{'TOPLAM':22} {scored_total:6d} {total['wrong']:7d} "
              f"{total['wrong'] / scored_total * 100:6.1f}%")
    if total["unknown"]:
        print(f"({total['unknown']} karar 'Unknown' — skorlamaya girmedi)")

    # Güven bantlarına göre: karar kuralı iyileştirmesinin alabileceği pay burada.
    print("\n=== margin bandına göre yanlış oranı ===")
    bands = [(None, 0.06, "belirsiz (<0.06)"), (0.06, 0.15, "orta (0.06-0.15)"),
             (0.15, None, "net (>0.15)")]
    for low, high, label in bands:
        subset = [d for d in decisions
                  if d["margin"] is not None
                  and (low is None or d["margin"] >= low)
                  and (high is None or d["margin"] < high)
                  and d["assigned"] and d["assigned"] != "Unknown"
                  and d["reference"] is not None]
        if not subset:
            continue
        wrong = sum(1 for d in subset if mapping.get(d["assigned"]) != d["reference"])
        print(f"  {label:20} {len(subset):5d} karar   yanlış {wrong:4d}  "
              f"({wrong / len(subset) * 100:5.1f}%)")


if __name__ == "__main__":
    main()
