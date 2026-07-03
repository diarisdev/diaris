"""AMI konuşmacı-eşik (embedding threshold) taraması + karşılaştırma raporu.

Her eşik için sırayla `ami_replay` (hipotez üret) ve `ami_score` (DER/WER/cpWER)
çalıştırır, sonuçları tek bir Markdown + konsol tablosunda toplar. Amaç: yeni
rezervuar tabanlı SpeakerTracker için over-collapse (çok az konuşmacı) ile
over-segmentation (çok fazla konuşmacı) arasındaki dengeyi bulmak — hedef, hyp
konuşmacı sayısını gerçek sayıya (`spk` sütunu) yaklaştırıp DER/cpWER'i düşürmek.

Not: --embedding-threshold yalnızca EVAL yolunu ayarlar (EvalAIWorker); canlı
.env varsayılanına dokunmaz (eval ↔ prod ayrımı korunur).

Çalıştırma (proje kökünden, eval ortamında):
    python scripts/ami_threshold_sweep.py
    python scripts/ami_threshold_sweep.py --only IS1009a --thresholds 0.50 0.55 0.60
    python scripts/ami_threshold_sweep.py --only IS1009a ES2004a   # birden çok toplantı
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows konsolu (cp1254/cp1252) UTF-8 olmayan karakterlerde çökebilir; rapor
# UTF-8 dosyaya yazılır ama konsol echo'su için stdout'u güvenli kıl.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def run_one(threshold: float, meetings: list[str]) -> dict | None:
    """Tek bir eşik için replay + score çalıştırır, aggregate sonucu döndürür."""
    only_args = []
    if meetings:
        only_args = ["--only", *meetings]

    replay_cmd = [
        sys.executable, "-m", "tests.benchmarks.ami_replay",
        "--embedding-threshold", f"{threshold:.2f}",
        *only_args,
    ]
    print(f"\n>>> [thr={threshold:.2f}] replay çalışıyor...", flush=True)
    replay = subprocess.run(replay_cmd, cwd=str(ROOT),
                            encoding="utf-8", errors="replace")
    if replay.returncode != 0:
        print(f"    ! replay başarısız (exit {replay.returncode})")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    save_path = tmp.name
    tmp.close()
    score_cmd = [
        sys.executable, "-m", "tests.benchmarks.ami_score",
        "--save", save_path, *only_args,
    ]
    print(f">>> [thr={threshold:.2f}] score çalışıyor...", flush=True)
    score = subprocess.run(score_cmd, cwd=str(ROOT),
                           encoding="utf-8", errors="replace")
    if score.returncode != 0:
        print(f"    ! score başarısız (exit {score.returncode})")
        return None

    try:
        data = json.loads(Path(save_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    ! sonuç JSON okunamadı: {exc}")
        return None
    finally:
        Path(save_path).unlink(missing_ok=True)

    agg = data.get("aggregate", {})
    meetings_data = data.get("meetings", [])
    ref_spk = meetings_data[0]["ref_speakers"] if meetings_data else None
    hyp_spk = sum(m.get("hyp_speakers", 0) for m in meetings_data)
    confusion = None
    if meetings_data and meetings_data[0].get("der_confusion") is not None:
        confusion = sum(m["der_confusion"] for m in meetings_data) / len(meetings_data)
    missed = None
    if meetings_data and meetings_data[0].get("der_missed") is not None:
        missed = sum(m["der_missed"] for m in meetings_data) / len(meetings_data)

    return {
        "threshold": threshold,
        "der": agg.get("der_avg"),
        "der_no_overlap": agg.get("der_avg_no_overlap"),
        "missed": missed,
        "confusion": confusion,
        "wer": agg.get("wer"),
        "cpwer": agg.get("cpwer"),
        "hyp_speakers": hyp_spk,
        "ref_speakers": ref_spk,
    }


def _fmt(value, spec="6.2f") -> str:
    return f"{value:{spec}}" if isinstance(value, (int, float)) else f"{'—':>6}"


def build_report(rows: list[dict], meetings: list[str], elapsed: float) -> str:
    scope = ", ".join(meetings) if meetings else "ALL meetings"
    lines = [
        "# AMI Embedding-Threshold Sweep",
        "",
        f"- Scope: **{scope}**",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Total wall time: {elapsed / 60:.1f} min",
        "",
        "Goal: `spk` closest to `ref` with lowest DER/cpWER. Low threshold ->",
        "over-collapse (too few speakers); high threshold -> over-segmentation",
        "(too many, high confusion).",
        "",
        "| thr | DER | DER-noov | Missed | Confusion | WER | cpWER | spk/ref |",
        "|----:|----:|---------:|-------:|----------:|----:|------:|:-------:|",
    ]
    best = None
    for r in rows:
        ref = r.get("ref_speakers")
        spk = r.get("hyp_speakers")
        spk_ref = f"{spk}/{ref}" if ref is not None else str(spk)
        lines.append(
            f"| {r['threshold']:.2f} | {_fmt(r['der'])} | {_fmt(r['der_no_overlap'])} "
            f"| {_fmt(r['missed'])} | {_fmt(r['confusion'])} | {_fmt(r['wer'])} "
            f"| {_fmt(r['cpwer'])} | {spk_ref} |"
        )
        # "En iyi": geçerli cpWER'i en düşük olan.
        if isinstance(r.get("cpwer"), (int, float)):
            if best is None or r["cpwer"] < best["cpwer"]:
                best = r

    lines.append("")
    if best is not None:
        ref = best.get("ref_speakers")
        lines.append(
            f"**Lowest cpWER:** threshold **{best['threshold']:.2f}** "
            f"-> cpWER {best['cpwer']:.2f}, DER {_fmt(best['der']).strip()}, "
            f"spk {best['hyp_speakers']}/{ref}."
        )
    else:
        lines.append("**No valid results** — check that AMI refs/audio exist and the eval env is active.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="AMI embedding-threshold sweep + rapor.")
    ap.add_argument("--only", nargs="*", default=["IS1009a"],
                    help="Skorlanacak toplantı(lar). Varsayılan: IS1009a. "
                         "Tüm set için: --only (argümansız).")
    ap.add_argument("--thresholds", nargs="*", type=float,
                    default=[0.50, 0.55, 0.58, 0.62],
                    help="Denenecek embedding eşikleri.")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "output" / f"ami_threshold_sweep_{time.strftime('%Y%m%d_%H%M%S')}.md",
                    help="Markdown rapor çıktı yolu.")
    args = ap.parse_args()

    meetings = [m for m in args.only if m]  # boş liste = tüm set

    t0 = time.time()
    rows = []
    for thr in args.thresholds:
        row = run_one(thr, meetings)
        if row is not None:
            rows.append(row)

    elapsed = time.time() - t0
    report = build_report(rows, meetings, elapsed)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nRapor kaydedildi: {args.report.resolve()}")


if __name__ == "__main__":
    main()
