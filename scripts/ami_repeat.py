"""AMI gürültü tabanı ölçümü — AYNI yapılandırmayı N kez koşup yayılımı raporlar.

Neden: bu takip sistemi yol-bağımlı (path-dependent). Toplantının başındaki tek
bir sınırda-karar, tüm koşuyu değiştirebilecek bir zincir başlatır (örn. baskın
bir profilin oluşup her şeyi yutması). Bu yüzden İKİ FARKLI yapılandırmayı
tek-koşu sonuçlarıyla karşılaştırmak yanıltıcıdır: ölçtüğünüz fark, gürültünün
içinde kaybolabilir.

Bu script hiçbir şeyi değiştirmez; aynı kodu/ayarı N kez koşar ve her metrik için
ortalama ± standart sapma ile min/max yayılımını basar. Çıktı, "hangi büyüklükteki
bir farkın anlamlı sayılabileceğini" (gürültü tabanı) belirler.

Çalıştırma (proje kökünden, eval ortamında):
    python scripts/ami_repeat.py                          # IS1009a, 3 koşu
    python scripts/ami_repeat.py --runs 5
    python scripts/ami_repeat.py --only IS1009a --runs 3 --embedding-threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows konsolu (cp1254/cp1252) UTF-8 olmayan karakterlerde çökebilir.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Raporlanan metrikler: (json_anahtarı, görünen ad, kaynak)
#   "agg"     -> aggregate bloğundan
#   "meeting" -> meetings[] ortalaması
METRICS = [
    ("der_avg", "DER", "agg"),
    ("der_avg_no_overlap", "DER-noov", "agg"),
    ("der_missed", "Miss", "meeting"),
    ("der_false_alarm", "FA", "meeting"),
    ("der_confusion", "Conf", "meeting"),
    ("wer", "WER", "agg"),
    ("cpwer", "cpWER", "agg"),
    ("hyp_speakers", "Speakers", "meeting_sum"),
]


def run_once(run_index: int, total: int, meetings: list[str],
             embedding_threshold: float) -> dict | None:
    """Tek bir tam koşu: replay + score. Metrik sözlüğü döndürür."""
    only_args = ["--only", *meetings] if meetings else []

    print(f"\n>>> [koşu {run_index}/{total}] replay...", flush=True)
    replay = subprocess.run(
        [sys.executable, "-m", "tests.benchmarks.ami_replay",
         "--embedding-threshold", f"{embedding_threshold:.2f}", *only_args],
        cwd=str(ROOT), encoding="utf-8", errors="replace")
    if replay.returncode != 0:
        print(f"    ! replay başarısız (exit {replay.returncode})")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    save_path = tmp.name
    tmp.close()
    print(f">>> [koşu {run_index}/{total}] score...", flush=True)
    score = subprocess.run(
        [sys.executable, "-m", "tests.benchmarks.ami_score",
         "--save", save_path, *only_args],
        cwd=str(ROOT), encoding="utf-8", errors="replace")
    if score.returncode != 0:
        print(f"    ! score başarısız (exit {score.returncode})")
        Path(save_path).unlink(missing_ok=True)
        return None

    try:
        data = json.loads(Path(save_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    ! sonuç JSON okunamadı: {exc}")
        return None
    finally:
        Path(save_path).unlink(missing_ok=True)

    agg = data.get("aggregate", {})
    mts = data.get("meetings", [])
    out = {}
    for key, _name, source in METRICS:
        if source == "agg":
            out[key] = agg.get(key)
        elif source == "meeting_sum":
            out[key] = sum(m.get(key, 0) for m in mts) if mts else None
        else:
            vals = [m[key] for m in mts if m.get(key) is not None]
            out[key] = (sum(vals) / len(vals)) if vals else None
    out["_ref_speakers"] = mts[0].get("ref_speakers") if mts else None
    return out


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    """(ortalama, std, min, max) — tek değerde std=0."""
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std, min(values), max(values)


def build_report(runs: list[dict], meetings: list[str], embedding_threshold: float,
                 elapsed: float) -> str:
    scope = ", ".join(meetings) if meetings else "ALL meetings"
    n = len(runs)
    lines = [
        "# AMI Noise-Floor Measurement (identical config, repeated)",
        "",
        f"- Scope: **{scope}**  |  runs: **{n}**  |  "
        f"embedding-threshold: {embedding_threshold:.2f}",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"wall time: {elapsed / 60:.1f} min",
        "",
        "Aynı kod, aynı ayar, aynı veri. Buradaki yayılım TAMAMEN gürültüdür.",
        "",
    ]

    # --- Ham koşu tablosu ---
    header = "| Run | " + " | ".join(name for _k, name, _s in METRICS) + " |"
    sep = "|---" * (len(METRICS) + 1) + "|"
    lines += [header, sep]
    for i, r in enumerate(runs, 1):
        cells = []
        for key, _name, _s in METRICS:
            v = r.get(key)
            cells.append(f"{v:.2f}" if isinstance(v, (int, float)) else "—")
        lines.append(f"| {i} | " + " | ".join(cells) + " |")

    # --- Özet istatistikler ---
    lines += ["", "## Spread", "",
              "| Metric | Mean | Std | Min | Max | Range | Noise band (±2σ) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    noise = {}
    for key, name, _s in METRICS:
        vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if not vals:
            lines.append(f"| {name} | — | — | — | — | — | — |")
            continue
        mean, std, lo, hi = _stats(vals)
        noise[name] = 2 * std
        lines.append(
            f"| {name} | {mean:.2f} | {std:.2f} | {lo:.2f} | {hi:.2f} | "
            f"{hi - lo:.2f} | ±{2 * std:.2f} |")

    lines += ["", "## How to read this", ""]
    if n < 2:
        lines.append("Tek koşu yapıldı — yayılım hesaplanamaz. `--runs 3` ile tekrarlayın.")
    else:
        cp = noise.get("cpWER", 0.0)
        der = noise.get("DER", 0.0)
        lines += [
            f"- **cpWER gürültü bandı ≈ ±{cp:.2f} puan**, **DER ≈ ±{der:.2f} puan** "
            f"({n} koşu, ±2σ).",
            "- İki yapılandırma arasındaki fark bu bandın İÇİNDEyse, o fark "
            "**ölçülmemiş sayılmalıdır** — tek koşuyla \"iyileşti/kötüleşti\" denemez.",
            "- Gerçek bir etkiyi doğrulamak için: her yapılandırmayı en az bu kadar "
            "tekrarla koşup ORTALAMALARI karşılaştırın.",
            "- WER en kararlı metriktir (ASR yolu diarization'dan bağımsız); WER'in "
            "yayılımı büyükse sorun tracker'da değil, replay/ASR tarafındadır.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AMI gürültü tabanı: aynı yapılandırmayı N kez koş.")
    ap.add_argument("--only", nargs="*", default=["IS1009a"],
                    help="Toplantı(lar). Varsayılan: IS1009a.")
    ap.add_argument("--runs", type=int, default=3,
                    help="Tekrar sayısı (varsayılan 3; 5 daha güvenilir).")
    ap.add_argument("--embedding-threshold", type=float, default=0.55,
                    help="Konuşmacı eşiği — tüm koşularda SABİT tutulur.")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "output" / f"ami_repeat_{time.strftime('%Y%m%d_%H%M%S')}.md",
                    help="Markdown rapor çıktı yolu.")
    args = ap.parse_args()

    meetings = [m for m in args.only if m]
    total = max(1, args.runs)
    t0 = time.time()

    runs = []
    for i in range(1, total + 1):
        r = run_once(i, total, meetings, args.embedding_threshold)
        if r is not None:
            runs.append(r)

    if not runs:
        print("\nHiçbir koşu tamamlanamadı — rapor üretilmedi.")
        raise SystemExit(1)

    report = build_report(runs, meetings, args.embedding_threshold, time.time() - t0)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nRapor kaydedildi: {args.report.resolve()}")


if __name__ == "__main__":
    main()
