"""Posterior parametrelerini (θ, T) AMI referansından kalibre eder.

Bugüne kadar konuşmacı eşiği TARANARAK bulundu: bir dizi değer denenip DER'e
göre en iyisi seçildi. Bu, "hangi sayı iyi sonuç verdi" sorusunu cevaplar ama
"bu skor ne anlama geliyor" sorusunu cevaplamaz — ve model/mikrofon değişince
baştan taranması gerekir.

Burada bunun yerine skor→olasılık eşlemesi ÖĞRENİLİR:

    P(aynı konuşmacı | s) = sigmoid((s − θ) / T)

θ ve T, izlerdeki her (skor, gerçekten aynı konuşmacı mı) çiftinden maksimum
olabilirlikle uydurulur. Sonuç: `P = 0.8` gerçekten "%80 doğru" demek olur ve
politika kesim noktaları (etiketi ver / profili güncelle / yeni konuşmacı aç)
anlamlı sayılara oturur.

TUTULAN-DIŞARIDA DOĞRULAMA: parametreler değerlendirildikleri veriye
uydurulursa sayılar iyimser çıkar. Bölmeler TOPLANTI bazında yapılır — aynı
toplantının kararları aynı profilleri paylaştığı için karar bazında bölmek
sızıntı olurdu.

Kullanım:
    python -m tests.benchmarks.ami_replay --only IS1009a ES2004a TS3003a \
        --trace-embeddings --out datasets/ami/ami_hyp_acc
    python scripts/calibrate_speaker_posterior.py datasets/ami/ami_hyp_acc/traces
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.embedding_trace import load_trace  # noqa: E402
from src.core.speaker_posterior import PosteriorConfig, speaker_posterior  # noqa: E402
from tests.benchmarks.trace_alignment import (  # noqa: E402
    build_label_mapping,
    collect_decisions,
    labelled_score_pairs,
    load_reference,
)

# Uydurma ızgarası. Kosinüs skorları [-1, 1] ama pratikte [0, 0.9] bandında;
# ızgara bu bandı ve makul sıcaklıkları kapsar.
THETA_GRID = [i / 100.0 for i in range(10, 86)]
TEMPERATURE_GRID = [i / 100.0 for i in range(2, 61)]


def log_likelihood(pairs, theta: float, temperature: float) -> float:
    """Lojistik modelin log-olabilirliği (taşmaya karşı kırpılmış)."""
    total = 0.0
    for score, is_same in pairs:
        z = max(-40.0, min(40.0, (score - theta) / temperature))
        probability = 1.0 / (1.0 + math.exp(-z))
        probability = min(max(probability, 1e-9), 1.0 - 1e-9)
        total += math.log(probability) if is_same else math.log(1.0 - probability)
    return total


def fit(pairs) -> tuple[float, float, float]:
    """(θ, T, log-olabilirlik) — ızgara araması.

    İki parametre için ızgara yeterli ve bağımlılıksız; gradyan tabanlı bir
    çözücü (scipy) eklemek bu boyutta kazanç sağlamaz.
    """
    best = None
    for theta in THETA_GRID:
        for temperature in TEMPERATURE_GRID:
            value = log_likelihood(pairs, theta, temperature)
            if best is None or value > best[2]:
                best = (theta, temperature, value)
    return best


def load_meeting(trace_path: Path, refs_dir: Path):
    """Bir toplantının izini referansla hizalar."""
    meeting = trace_path.stem
    rttm = refs_dir / f"{meeting}.rttm"
    if not rttm.exists():
        return None
    trace = load_trace(trace_path)
    reference = load_reference(rttm)
    decisions = collect_decisions(trace, reference)
    if not decisions:
        return None
    mapping = build_label_mapping(decisions)
    return {
        "meeting": meeting,
        "decisions": decisions,
        "mapping": mapping,
        "pairs": labelled_score_pairs(decisions, mapping),
    }


def band_report(meetings, config: PosteriorConfig, title: str) -> None:
    """Güven bandına göre hata oranı — kalibrasyonun işe yarayıp yaramadığı."""
    rows = []
    for meeting in meetings:
        for decision in meeting["decisions"]:
            assigned = decision["assigned"]
            if not assigned or assigned == "Unknown" or decision["reference"] is None:
                continue
            if not decision["raw_scores"]:
                continue
            posterior = speaker_posterior(decision["raw_scores"], config,
                                          duration=decision["probe_duration"])
            wrong = meeting["mapping"].get(assigned) != decision["reference"]
            rows.append((posterior.best_probability, wrong))

    if not rows:
        return
    print(f"\n  {title}  (theta={config.theta:.2f}, T={config.temperature:.2f})")
    print(f"  {'bant':14} {'karar':>7} {'yanlis':>7} {'oran':>8}")
    print("  " + "-" * 40)
    bands = ((None, 0.50, "<0.50"), (0.50, 0.75, "0.50-0.75"),
             (0.75, 0.90, "0.75-0.90"), (0.90, None, ">0.90"))
    for low, high, label in bands:
        subset = [r for r in rows
                  if (low is None or r[0] >= low) and (high is None or r[0] < high)]
        if not subset:
            continue
        wrong = sum(1 for _, w in subset if w)
        print(f"  {label:14} {len(subset):7d} {wrong:7d} {wrong / len(subset) * 100:7.1f}%")

    correct = [p for p, w in rows if not w]
    incorrect = [p for p, w in rows if w]
    if correct and incorrect:
        print(f"  ortalama p_best -> DOGRU {statistics.mean(correct):.3f} | "
              f"YANLIS {statistics.mean(incorrect):.3f}")


def suggest_cutpoints(meetings, config: PosteriorConfig) -> dict:
    """Hedef hata oranlarına karşılık gelen p_best kesim noktaları.

    Politika eşikleri elle seçilecek sayılar olmamalı: "hangi güvenin üstünde
    hata %5'in altında kalıyor" sorusunun cevabı veriden okunur.
    """
    rows = []
    for meeting in meetings:
        for decision in meeting["decisions"]:
            assigned = decision["assigned"]
            if not assigned or assigned == "Unknown" or decision["reference"] is None:
                continue
            if not decision["raw_scores"]:
                continue
            posterior = speaker_posterior(decision["raw_scores"], config,
                                          duration=decision["probe_duration"])
            rows.append((posterior.best_probability,
                         meeting["mapping"].get(assigned) != decision["reference"]))
    rows.sort(key=lambda r: -r[0])

    suggestions = {}
    for target in (0.10, 0.05, 0.02):
        # Yukarıdan aşağı in; hata oranı hedefi aştığı anda dur.
        wrong = 0
        cut = None
        for index, (probability, is_wrong) in enumerate(rows, start=1):
            wrong += is_wrong
            if wrong / index <= target:
                cut = probability
        if cut is not None:
            covered = sum(1 for p, _ in rows if p >= cut)
            suggestions[f"error<={target:.0%}"] = {
                "p_best": round(cut, 3),
                "coverage": round(covered / len(rows), 3) if rows else 0.0,
            }
    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description="Posterior kalibrasyonu (theta, T).")
    parser.add_argument("traces", type=Path,
                        help="İz dizini (ami_replay --trace-embeddings çıktısı) veya tek .npz")
    parser.add_argument("--refs", type=Path,
                        default=PROJECT_ROOT / "datasets" / "ami" / "ami_refs" / "rttm",
                        help="Referans RTTM dizini.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Kalibrasyon çıktısının yazılacağı JSON dosyası.")
    args = parser.parse_args()

    paths = ([args.traces] if args.traces.suffix == ".npz"
             else sorted(args.traces.glob("*.npz")))
    if not paths:
        raise SystemExit(f"İz dosyası bulunamadı: {args.traces}")

    meetings = [m for m in (load_meeting(p, args.refs) for p in paths) if m]
    if not meetings:
        raise SystemExit("Hiçbir iz referansla hizalanamadı (turn aralığı eksik olabilir).")

    all_pairs = [pair for meeting in meetings for pair in meeting["pairs"]]
    same = [s for s, y in all_pairs if y]
    different = [s for s, y in all_pairs if not y]

    print("=== Kalibrasyon verisi ===")
    print(f"  toplanti : {', '.join(m['meeting'] for m in meetings)}")
    print(f"  etiketli cift: {len(all_pairs)} "
          f"(ayni {len(same)}, farkli {len(different)})")
    if same and different:
        print(f"  AYNI  konusmaci skor: ort {statistics.mean(same):.3f} "
              f"medyan {statistics.median(same):.3f}")
        print(f"  FARKLI konusmaci skor: ort {statistics.mean(different):.3f} "
              f"medyan {statistics.median(different):.3f}")

    theta, temperature, _ = fit(all_pairs)
    print("\n=== Tum veriye uydurma (in-sample) ===")
    print(f"  theta = {theta:.2f}   T = {temperature:.2f}")

    # --- Tutulan-dışarıda: her toplantıyı sırayla dışarıda bırak -----------
    print("\n=== Tutulan-disarida dogrulama (toplanti bazinda) ===")
    holdout = []
    for index, meeting in enumerate(meetings):
        if len(meetings) < 2:
            break
        train_pairs = [p for j, m in enumerate(meetings) if j != index for p in m["pairs"]]
        fitted_theta, fitted_temperature, _ = fit(train_pairs)
        holdout.append((meeting["meeting"], fitted_theta, fitted_temperature))
        print(f"  {meeting['meeting']} disarida -> theta={fitted_theta:.2f} "
              f"T={fitted_temperature:.2f}")
        band_report([meeting],
                    PosteriorConfig(theta=fitted_theta, temperature=fitted_temperature),
                    f"{meeting['meeting']} (bu toplanti EGITIME girmedi)")

    if holdout:
        thetas = [h[1] for h in holdout]
        temperatures = [h[2] for h in holdout]
        spread = max(thetas) - min(thetas)
        print(f"\n  theta yayilimi: {min(thetas):.2f}-{max(thetas):.2f} "
              f"(fark {spread:.2f})   T yayilimi: {min(temperatures):.2f}-"
              f"{max(temperatures):.2f}")
        if spread > 0.05:
            print("  UYARI: theta toplantidan toplantiya oynuyor — tek bir kuresel "
                  "deger yerine oturum ici uyarlama gerekebilir.")

    print("\n=== Varsayilan vs kalibre (tum veri) ===")
    band_report(meetings, PosteriorConfig(), "VARSAYILAN")
    band_report(meetings, PosteriorConfig(theta=theta, temperature=temperature), "KALIBRE")

    calibrated = PosteriorConfig(theta=theta, temperature=temperature)
    cutpoints = suggest_cutpoints(meetings, calibrated)
    print("\n=== Onerilen politika kesim noktalari ===")
    for name, value in cutpoints.items():
        print(f"  {name:14} p_best >= {value['p_best']:.3f}  "
              f"(kararlarin %{value['coverage'] * 100:.0f}'i)")

    if args.out:
        payload = {
            "theta": theta,
            "temperature": temperature,
            "meetings": [m["meeting"] for m in meetings],
            "pair_count": len(all_pairs),
            "holdout": [{"meeting": m, "theta": t, "temperature": temp}
                        for m, t, temp in holdout],
            "suggested_cutpoints": cutpoints,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nKalibrasyon yazildi: {args.out}")


if __name__ == "__main__":
    main()
