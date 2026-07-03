"""WebRTC VAD ablasyonu — AMI üzerinde AÇIK vs KAPALI otomatik A/B testi.

Aynı toplantı(lar) iki kez koşulur: VAD_USE_WEBRTC=1 (WebRTC AND-kapısı aktif)
ve VAD_USE_WEBRTC=0 (karar yalnız Silero). Her mod için ami_replay (hipotez
üret) + ami_score (DER/WER/cpWER) çalıştırılır; sonuçlar tek Markdown
karşılaştırma tablosunda toplanır ve fark satırı basılır.

Hipotez (bkz. src/audio/vad.py notu): Silero her frame'de zaten çalıştığı için
WebRTC hesaplama tasarrufu sağlamaz; AND-kapısı yalnızca kaçırma (miss) ekleyip
yanlış alarmı azaltabilir. Hata profili miss-baskın olduğundan beklenti,
KAPALI modun eşit ya da hafif daha iyi çıkmasıdır. Bu script tahmini ölçüme
çevirir.

NOT: DER/cpWER koşudan koşuya oynayabilir; kararlı karşılaştırma için WER'e ve
--repeats ile ortalamaya bakın. Config env'i import anında okunduğundan her
koşu AYRI alt-süreçte yapılır.

Çalıştırma (proje kökünden, eval ortamında):
    python scripts/vad_webrtc_ablation.py
    python scripts/vad_webrtc_ablation.py --only IS1009a --repeats 3
    python scripts/vad_webrtc_ablation.py --only IS1009a ES2004a --embedding-threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import os
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


def run_once(use_webrtc: bool, meetings: list[str], embedding_threshold: float) -> dict | None:
    """Tek koşu: replay + score. Aggregate + konuşmacı sayısı döndürür."""
    env = dict(os.environ)
    env["VAD_USE_WEBRTC"] = "1" if use_webrtc else "0"
    env["PYTHONUTF8"] = "1"

    only_args = ["--only", *meetings] if meetings else []
    label = "webrtc=ON " if use_webrtc else "webrtc=OFF"

    replay_cmd = [
        sys.executable, "-m", "tests.benchmarks.ami_replay",
        "--embedding-threshold", f"{embedding_threshold:.2f}",
        *only_args,
    ]
    print(f"\n>>> [{label}] replay çalışıyor...", flush=True)
    replay = subprocess.run(replay_cmd, cwd=str(ROOT), env=env,
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
    print(f">>> [{label}] score çalışıyor...", flush=True)
    score = subprocess.run(score_cmd, cwd=str(ROOT), env=env,
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
    mts = data.get("meetings", [])

    def _avg(key):
        vals = [m[key] for m in mts if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "der": agg.get("der_avg"),
        "der_no_overlap": agg.get("der_avg_no_overlap"),
        "missed": _avg("der_missed"),
        "false_alarm": _avg("der_false_alarm"),
        "confusion": _avg("der_confusion"),
        "wer": agg.get("wer"),
        "cpwer": agg.get("cpwer"),
        "hyp_speakers": sum(m.get("hyp_speakers", 0) for m in mts),
        "ref_speakers": mts[0].get("ref_speakers") if mts else None,
    }


def _mean_rows(rows: list[dict]) -> dict:
    """Birden çok koşunun sayısal ortalaması (repeats > 1)."""
    if len(rows) == 1:
        return dict(rows[0])
    out = dict(rows[0])
    for key in ("der", "der_no_overlap", "missed", "false_alarm",
                "confusion", "wer", "cpwer", "hyp_speakers"):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def _fmt(v, nd=2) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def build_report(on: dict | None, off: dict | None, meetings: list[str],
                 repeats: int, embedding_threshold: float, elapsed: float) -> str:
    scope = ", ".join(meetings) if meetings else "ALL meetings"
    lines = [
        "# WebRTC VAD Ablation (AMI)",
        "",
        f"- Scope: **{scope}**  |  repeats: {repeats}  |  "
        f"embedding-threshold: {embedding_threshold:.2f}",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"wall time: {elapsed / 60:.1f} min",
        "",
        "| Mode | DER | Miss | FA | Conf | WER | cpWER | spk/ref |",
        "|---|----:|----:|----:|----:|----:|----:|:---:|",
    ]

    def row(name, r):
        if r is None:
            return f"| {name} | — | — | — | — | — | — | — |"
        spk = f"{_fmt(r['hyp_speakers'], 1)}/{r['ref_speakers']}"
        return (f"| {name} | {_fmt(r['der'])} | {_fmt(r['missed'])} | "
                f"{_fmt(r['false_alarm'])} | {_fmt(r['confusion'])} | "
                f"{_fmt(r['wer'])} | {_fmt(r['cpwer'])} | {spk} |")

    lines.append(row("WebRTC **ON** (AND gate)", on))
    lines.append(row("WebRTC **OFF** (Silero only)", off))

    if on and off:
        def d(key):
            a, b = off.get(key), on.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return f"{a - b:+.2f}"
            return "—"
        lines.append(f"| Δ (OFF − ON) | {d('der')} | {d('missed')} | "
                     f"{d('false_alarm')} | {d('confusion')} | {d('wer')} | "
                     f"{d('cpwer')} | |")
        lines.append("")

        # Yorum: negatif Δ = OFF daha iyi (hata düştü)
        verdict = []
        for key, name in (("wer", "WER"), ("der", "DER"), ("cpwer", "cpWER")):
            a, b = off.get(key), on.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diff = a - b
                if abs(diff) < 0.5:
                    verdict.append(f"{name}: fark gürültü sınırında ({diff:+.2f})")
                elif diff < 0:
                    verdict.append(f"{name}: OFF daha iyi ({diff:+.2f})")
                else:
                    verdict.append(f"{name}: ON daha iyi ({diff:+.2f})")
        lines.append("**Verdict:** " + "; ".join(verdict) + ".")
        lines.append("")
        lines.append("Okuma rehberi: Δ negatifse WebRTC'siz mod daha iyi. WER en kararlı "
                     "metriktir; DER/cpWER koşudan koşuya oynayabilir (--repeats ile "
                     "ortalayın). OFF eşit ya da daha iyiyse VAD_USE_WEBRTC=false yapıp "
                     "katmanı kapatmak güvenlidir.")
    else:
        lines.append("")
        lines.append("**Bir veya iki koşu başarısız — sonuç yok.**")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="WebRTC VAD A/B ablasyonu (AMI).")
    ap.add_argument("--only", nargs="*", default=["IS1009a"],
                    help="Skorlanacak toplantı(lar). Varsayılan: IS1009a.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Mod başına koşu sayısı (DER/cpWER gürültüsünü ortalar).")
    ap.add_argument("--embedding-threshold", type=float, default=0.55,
                    help="Konuşmacı eşiği (sabit tutulur; 0.55 = kalibre edilmiş değer).")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "output" / f"vad_webrtc_ablation_{time.strftime('%Y%m%d_%H%M%S')}.md",
                    help="Markdown rapor çıktı yolu.")
    args = ap.parse_args()

    meetings = [m for m in args.only if m]
    t0 = time.time()

    results: dict[bool, dict | None] = {}
    for use_webrtc in (True, False):
        runs = []
        for i in range(max(1, args.repeats)):
            if args.repeats > 1:
                print(f"\n=== {'ON' if use_webrtc else 'OFF'} — koşu {i + 1}/{args.repeats} ===")
            r = run_once(use_webrtc, meetings, args.embedding_threshold)
            if r is not None:
                runs.append(r)
        results[use_webrtc] = _mean_rows(runs) if runs else None

    report = build_report(results[True], results[False], meetings,
                          max(1, args.repeats), args.embedding_threshold,
                          time.time() - t0)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nRapor kaydedildi: {args.report.resolve()}")


if __name__ == "__main__":
    main()
