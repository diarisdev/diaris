"""Son-işleme (refinement) deneyi — GPU'suz, replay'siz, deterministik.

`ami_replay` tarafından kaydedilen HAM segmentleri (ami_hyp/segments/*.json)
okur, `src.core.speaker_refinement` kurallarını uygular, sonucu ayrı bir hipotez
dizinine yazar ve ÖNCE/SONRA metriklerini karşılaştırır.

Neden bu şekilde: replay çalıştırmak GPU ister ve bu sistemde koşudan-koşuya
büyük oynamalar (bimodal davranış) görülüyor. Refinement ise SAF bir fonksiyon —
aynı segmentlere aynı kuralları uygulamak her zaman aynı sonucu verir. Böylece
kural eşiklerini, ölçüm gürültüsüne bulaşmadan, saniyeler içinde deneyebilirsiniz.

Ön koşul: en az bir kez `ami_replay` koşup segments/ dizinini üretmiş olmak.

Çalıştırma (proje kökünden):
    python scripts/refine_experiment.py
    python scripts/refine_experiment.py --only IS1009a
    python scripts/refine_experiment.py --tiny-max-duration 8 --passes 3
    python scripts/refine_experiment.py --no-score      # sadece uygula, skorlama
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from src.core.speaker_refinement import (  # noqa: E402
    RefinementConfig,
    is_pseudo_label,
    refine_speakers,
    speaker_summary,
)


# --------------------------------------------------------------------------- #
# ami_replay ile AYNI çıktı biçimleri (skorlayıcı bunları bekliyor)
# --------------------------------------------------------------------------- #
def results_to_rttm(meeting_id: str, results: list) -> str:
    lines = []
    for r in sorted(results, key=lambda x: x["start"]):
        dur = max(0.0, float(r["end"]) - float(r["start"]))
        if dur <= 0:
            continue
        lines.append(
            f"SPEAKER {meeting_id} 1 {float(r['start']):.3f} {dur:.3f} "
            f"<NA> <NA> {r['speaker']} <NA> <NA>"
        )
    return "\n".join(lines) + "\n"


def results_to_speaker_transcript(results: list) -> dict:
    per_spk: dict = {}
    flat: list = []
    for r in sorted(results, key=lambda x: x["start"]):
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        per_spk.setdefault(r["speaker"], []).append(txt)
        flat.append(txt)
    return {
        "speakers": {spk: " ".join(parts) for spk, parts in per_spk.items()},
        "flat": " ".join(flat),
    }


def real_speaker_count(segments: list) -> int:
    return len({s["speaker"] for s in segments if not is_pseudo_label(s["speaker"])})


# --------------------------------------------------------------------------- #
def score_dir(hyp_dir: Path, refs: Path, meetings: list) -> dict | None:
    """ami_score'u çağırıp aggregate metrikleri döndürür."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    save = tmp.name
    tmp.close()
    only = ["--only", *meetings] if meetings else []
    proc = subprocess.run(
        [sys.executable, "-m", "tests.benchmarks.ami_score",
         "--hyp", str(hyp_dir), "--refs", str(refs), "--save", save, *only],
        cwd=str(ROOT), encoding="utf-8", errors="replace",
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"    ! score başarısız (exit {proc.returncode})")
        print(proc.stdout[-1500:] if proc.stdout else "")
        print(proc.stderr[-1500:] if proc.stderr else "")
        Path(save).unlink(missing_ok=True)
        return None
    try:
        data = json.loads(Path(save).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        Path(save).unlink(missing_ok=True)

    agg = data.get("aggregate", {})
    mts = data.get("meetings", [])

    def _avg(key):
        vals = [m[key] for m in mts if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "der": agg.get("der_avg"),
        "der_noov": agg.get("der_avg_no_overlap"),
        "missed": _avg("der_missed"),
        "fa": _avg("der_false_alarm"),
        "conf": _avg("der_confusion"),
        "wer": agg.get("wer"),
        "cpwer": agg.get("cpwer"),
        "speakers": sum(m.get("hyp_speakers", 0) for m in mts),
        "ref_speakers": mts[0].get("ref_speakers") if mts else None,
    }


def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def build_report(rows: list, before: dict | None, after: dict | None,
                 cfg: RefinementConfig, elapsed: float) -> str:
    lines = [
        "# Speaker Refinement Experiment (offline, deterministic)",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  {elapsed:.1f}s",
        f"- Config: small_island(dur<={cfg.small_island_max_duration}, "
        f"seg<={cfg.small_island_max_segments}) | "
        f"tiny_fragmented(dur<={cfg.tiny_fragmented_max_duration}, "
        f"seg<={cfg.tiny_fragmented_max_segments}, "
        f"islands {cfg.tiny_fragmented_min_islands}-{cfg.tiny_fragmented_max_islands}, "
        f"share>={cfg.tiny_fragmented_min_neighbor_share}) | "
        f"unknown_fill(dur<={cfg.unknown_fill_max_duration}, "
        f"seg<={cfg.unknown_fill_max_segments}) | passes={cfg.passes}",
        "",
        "## Per-meeting changes",
        "",
        "| Meeting | Segments | Speakers before | Speakers after | small_island | tiny_frag | unknown_fill | total relabeled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        st = r["stats"]
        lines.append(
            f"| {r['meeting']} | {r['n_segments']} | {r['spk_before']} | {r['spk_after']} | "
            f"{st.get('small_island_merge', 0)} | {st.get('tiny_fragmented_merge', 0)} | "
            f"{st.get('unknown_fill', 0)} | {st.get('total', 0)} |"
        )

    if before and after:
        lines += ["", "## Metrics: before vs after", "",
                  "| Metric | Before | After | Δ |", "|---|---:|---:|---:|"]
        for key, name in (("der", "DER"), ("der_noov", "DER-noov"),
                          ("missed", "Miss"), ("fa", "FA"), ("conf", "Conf"),
                          ("wer", "WER"), ("cpwer", "cpWER"), ("speakers", "Speakers")):
            b, a = before.get(key), after.get(key)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                delta = a - b
                lines.append(f"| {name} | {_fmt(b)} | {_fmt(a)} | {delta:+.2f} |")
            else:
                lines.append(f"| {name} | {_fmt(b)} | {_fmt(a)} | — |")
        lines += ["", "Negatif Δ = iyileşme (Speakers hariç: hedef referans sayısına yaklaşmak).",
                  "",
                  "Bu karşılaştırma DETERMİNİSTİKtir: aynı segmentlere aynı kurallar "
                  "uygulanır, replay/GPU yoktur. Yani buradaki fark ölçüm gürültüsü "
                  "değil, doğrudan refinement'ın etkisidir."]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline speaker-refinement deneyi.")
    ap.add_argument("--hyp", type=Path, default=ROOT / "datasets" / "ami" / "ami_hyp",
                    help="ami_replay çıktı dizini (segments/ içermeli).")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "ami" / "ami_hyp_refined",
                    help="Refined hipotezlerin yazılacağı dizin.")
    ap.add_argument("--refs", type=Path, default=ROOT / "datasets" / "ami" / "ami_refs")
    ap.add_argument("--only", nargs="*", default=None, help="Sadece bu toplantı(lar).")
    ap.add_argument("--no-score", action="store_true", help="Skorlamayı atla.")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "output" / f"refine_experiment_{time.strftime('%Y%m%d_%H%M%S')}.md")
    # Eşikler
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--small-max-duration", type=float, default=5.0)
    ap.add_argument("--small-max-segments", type=int, default=3)
    ap.add_argument("--tiny-max-duration", type=float, default=6.0)
    ap.add_argument("--tiny-max-segments", type=int, default=8)
    ap.add_argument("--tiny-min-islands", type=int, default=2)
    ap.add_argument("--tiny-max-islands", type=int, default=3)
    ap.add_argument("--tiny-min-share", type=float, default=0.5)
    ap.add_argument("--unknown-max-duration", type=float, default=3.0)
    ap.add_argument("--unknown-max-segments", type=int, default=1)
    ap.add_argument("--no-small-island", action="store_true")
    ap.add_argument("--no-tiny-fragmented", action="store_true")
    ap.add_argument("--no-unknown-fill", action="store_true")
    args = ap.parse_args()

    cfg = RefinementConfig(
        small_island_merge=not args.no_small_island,
        small_island_max_duration=args.small_max_duration,
        small_island_max_segments=args.small_max_segments,
        tiny_fragmented_merge=not args.no_tiny_fragmented,
        tiny_fragmented_max_duration=args.tiny_max_duration,
        tiny_fragmented_max_segments=args.tiny_max_segments,
        tiny_fragmented_min_islands=args.tiny_min_islands,
        tiny_fragmented_max_islands=args.tiny_max_islands,
        tiny_fragmented_min_neighbor_share=args.tiny_min_share,
        unknown_fill=not args.no_unknown_fill,
        unknown_fill_max_duration=args.unknown_max_duration,
        unknown_fill_max_segments=args.unknown_max_segments,
        passes=args.passes,
    )

    seg_dir = args.hyp / "segments"
    if not seg_dir.is_dir():
        raise SystemExit(
            f"Segment dizini yok: {seg_dir.resolve()}\n"
            "Önce `python -m tests.benchmarks.ami_replay --only IS1009a` çalıştırın "
            "(bu sürüm segments/ dizinini de yazar)."
        )

    meetings = sorted(p.stem for p in seg_dir.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        meetings = [m for m in meetings if m in wanted]
    if not meetings:
        raise SystemExit("İşlenecek toplantı bulunamadı.")

    (args.out / "rttm").mkdir(parents=True, exist_ok=True)
    (args.out / "transcripts").mkdir(parents=True, exist_ok=True)
    (args.out / "segments").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    for mid in meetings:
        segments = json.loads((seg_dir / f"{mid}.json").read_text(encoding="utf-8"))
        spk_before = real_speaker_count(segments)
        refined, stats = refine_speakers(segments, cfg)
        spk_after = real_speaker_count(refined)

        (args.out / "rttm" / f"{mid}.rttm").write_text(
            results_to_rttm(mid, refined), encoding="utf-8")
        (args.out / "transcripts" / f"{mid}.json").write_text(
            json.dumps(results_to_speaker_transcript(refined), ensure_ascii=False, indent=2),
            encoding="utf-8")
        (args.out / "segments" / f"{mid}.json").write_text(
            json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")

        rows.append({"meeting": mid, "n_segments": len(segments),
                     "spk_before": spk_before, "spk_after": spk_after, "stats": stats})

        print(f"\n[{mid}] {len(segments)} segment | konuşmacı {spk_before} -> {spk_after}")
        print(f"   small_island={stats.get('small_island_merge',0)} "
              f"tiny_frag={stats.get('tiny_fragmented_merge',0)} "
              f"unknown_fill={stats.get('unknown_fill',0)} "
              f"(toplam {stats.get('total',0)} segment yeniden etiketlendi, "
              f"{stats.get('passes_run',0)} geçiş)")
        summary = speaker_summary(refined)
        total = sum(summary.values()) or 1.0
        print("   Sonrası dağılım:")
        for lbl, dur in list(summary.items())[:12]:
            print(f"     {lbl:22s} {dur:7.1f}s ({dur/total*100:5.1f}%)")

    before = after = None
    if not args.no_score:
        print("\n>>> ÖNCE skorlanıyor...", flush=True)
        before = score_dir(args.hyp, args.refs, args.only or [])
        print(">>> SONRA skorlanıyor...", flush=True)
        after = score_dir(args.out, args.refs, args.only or [])

    report = build_report(rows, before, after, cfg, time.time() - t0)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"\nRapor: {args.report.resolve()}")
    print(f"Refined hipotezler: {args.out.resolve()}")


if __name__ == "__main__":
    main()
