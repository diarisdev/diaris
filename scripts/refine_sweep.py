"""Refinement eşik taraması — kapsamlı, deterministik, GPU'suz.

`ami_replay` çıktısındaki HAM segmentleri okur ve `speaker_refinement` kurallarının
eşik uzayını tarar. Her yapılandırma için cpWER / WER / konuşmacı sayısı hesaplanır;
en iyi adaylar için ayrıca DER hesaplanır.

Neden hızlı: skorlama SÜREÇ İÇİNDE yapılır (ami_score alt-süreci yok) ve farklı
eşiklerin çoğu AYNI etiket dizisini üretir — sonuçlar bu imzaya göre bellekte
tutulur (memoization), böylece binlerce kombinasyon saniyeler içinde taranır.

Ön koşul: `python -m tests.benchmarks.ami_replay --only IS1009a` en az bir kez
koşulmuş olmalı (segments/ dizinini üretir).

Çalıştırma:
    python scripts/refine_sweep.py --only IS1009a
    python scripts/refine_sweep.py --only IS1009a --full     # geniş ızgara
    python scripts/refine_sweep.py --only IS1009a --top 20
"""
from __future__ import annotations

import argparse
import itertools
import json
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
)
from tests.metrics import (  # noqa: E402
    compute_der,
    compute_wer,
    cpwer_from_speaker_texts,
    load_rttm,
    normalize_text,
)


# --------------------------------------------------------------------------- #
# Izgaralar
# --------------------------------------------------------------------------- #
GRID_DEFAULT = {
    "tiny_fragmented_max_duration": [6.0, 10.0, 15.0, 20.0, 30.0],
    "tiny_fragmented_max_segments": [8, 16, 24],
    "tiny_fragmented_max_islands": [3, 5, 8],
    "tiny_fragmented_min_neighbor_share": [0.5, 0.6],
    "small_island_max_duration": [5.0, 12.0],
    "small_island_max_segments": [3, 8],
    "unknown_fill_max_duration": [3.0, 8.0],
    "unknown_fill_max_segments": [1, 5],
}

GRID_FULL = {
    "tiny_fragmented_max_duration": [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 45.0],
    "tiny_fragmented_max_segments": [4, 8, 12, 16, 24, 40],
    "tiny_fragmented_max_islands": [2, 3, 4, 5, 6, 8, 12],
    "tiny_fragmented_min_neighbor_share": [0.4, 0.5, 0.6, 0.7],
    "small_island_max_duration": [3.0, 5.0, 8.0, 12.0, 20.0],
    "small_island_max_segments": [2, 3, 5, 8, 12],
    "unknown_fill_max_duration": [2.0, 3.0, 5.0, 8.0, 12.0],
    "unknown_fill_max_segments": [1, 2, 3, 5, 8],
}


def real_speakers(segments: list) -> set:
    return {s["speaker"] for s in segments if not is_pseudo_label(s["speaker"])}


def hyp_speaker_texts(segments: list) -> dict:
    """ami_replay.results_to_speaker_transcript ile AYNI gruplama."""
    per: dict = {}
    for s in sorted(segments, key=lambda x: x["start"]):
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        per.setdefault(s["speaker"], []).append(txt)
    return {k: " ".join(v) for k, v in per.items()}


def hyp_intervals(segments: list) -> list:
    out = []
    for s in sorted(segments, key=lambda x: x["start"]):
        if float(s["end"]) > float(s["start"]):
            out.append({"start": float(s["start"]), "end": float(s["end"]),
                        "speaker": s["speaker"]})
    return out


class MeetingCase:
    """Tek toplantının referansları + segmentleri; skorlama yardımcıları."""

    def __init__(self, mid: str, segments: list, refs_dir: Path):
        self.mid = mid
        self.segments = segments
        ref_tr = json.loads((refs_dir / "transcripts" / f"{mid}.json").read_text(encoding="utf-8"))
        self.ref_speakers = {k: normalize_text(v) for k, v in ref_tr.get("speakers", {}).items()}
        self.ref_flat = ref_tr.get("flat", "")
        self.n_ref_speakers = len(self.ref_speakers)
        rttm = refs_dir / "rttm" / f"{mid}.rttm"
        self.ref_intervals = load_rttm(rttm) if rttm.exists() else None

    def score_text(self, refined: list) -> dict:
        hyp = {k: normalize_text(v) for k, v in hyp_speaker_texts(refined).items()}
        cp = cpwer_from_speaker_texts(self.ref_speakers, hyp)
        flat = " ".join((s.get("text") or "").strip() for s in sorted(refined, key=lambda x: x["start"]))
        w = compute_wer(self.ref_flat, flat)
        return {
            "cpwer": cp.cpwer * 100.0,
            "cp_errors": cp.errors,
            "cp_words": cp.total_ref_words,
            "wer": w.wer * 100.0,
            "speakers": len(real_speakers(refined)),
        }

    def score_der(self, refined: list) -> dict | None:
        if self.ref_intervals is None:
            return None
        res = compute_der(self.ref_intervals, hyp_intervals(refined),
                          collar=0.25, with_overlap_split=True)
        if res is None:
            return None
        return {"der": res.der * 100.0,
                "missed": res.missed * 100.0,
                "fa": res.false_alarm * 100.0,
                "conf": res.confusion * 100.0}


def signature(refined: list) -> tuple:
    return tuple(s["speaker"] for s in refined)


def build_config(combo: dict, passes: int) -> RefinementConfig:
    return RefinementConfig(passes=passes, **combo)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refinement eşik taraması.")
    ap.add_argument("--hyp", type=Path, default=ROOT / "datasets" / "ami" / "ami_hyp")
    ap.add_argument("--refs", type=Path, default=ROOT / "datasets" / "ami" / "ami_refs")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--full", action="store_true", help="Geniş ızgara (çok daha fazla kombinasyon).")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--top", type=int, default=15, help="Raporlanacak en iyi yapılandırma sayısı.")
    ap.add_argument("--der-top", type=int, default=10, help="DER hesaplanacak en iyi N yapılandırma.")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "output" / f"refine_sweep_{time.strftime('%Y%m%d_%H%M%S')}.md")
    args = ap.parse_args()

    seg_dir = args.hyp / "segments"
    if not seg_dir.is_dir():
        raise SystemExit(
            f"Segment dizini yok: {seg_dir.resolve()}\n"
            "Önce `python -m tests.benchmarks.ami_replay --only IS1009a` çalıştırın.")

    meetings = sorted(p.stem for p in seg_dir.glob("*.json"))
    if args.only:
        wanted = set(args.only)
        meetings = [m for m in meetings if m in wanted]
    if not meetings:
        raise SystemExit("İşlenecek toplantı bulunamadı.")

    cases = []
    for mid in meetings:
        segs = json.loads((seg_dir / f"{mid}.json").read_text(encoding="utf-8"))
        cases.append(MeetingCase(mid, segs, args.refs))

    grid = GRID_FULL if args.full else GRID_DEFAULT
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Toplantı: {', '.join(meetings)}")
    print(f"Izgara: {len(combos)} kombinasyon x {len(cases)} toplantı  "
          f"(passes={args.passes}, {'FULL' if args.full else 'default'})")

    # --- Referans nokta: refinement YOK ---
    baseline = {}
    for case in cases:
        baseline[case.mid] = case.score_text(case.segments)
    base_cp = sum(baseline[c.mid]["cp_errors"] for c in cases) / max(
        1, sum(baseline[c.mid]["cp_words"] for c in cases)) * 100.0
    base_spk = sum(baseline[c.mid]["speakers"] for c in cases)
    print(f"Baseline (refinement yok): cpWER {base_cp:.2f} | speakers {base_spk}")

    # --- Tarama ---
    memo: dict = {}
    results = []
    t0 = time.time()
    for i, values in enumerate(combos, 1):
        combo = dict(zip(keys, values))
        cfg = build_config(combo, args.passes)

        sigs, per_meeting, fired = [], {}, {}
        for case in cases:
            refined, stats = refine_speakers(case.segments, cfg)
            sig = (case.mid, signature(refined))
            sigs.append(sig)
            if sig in memo:
                per_meeting[case.mid] = memo[sig]
            else:
                sc = case.score_text(refined)
                memo[sig] = sc
                per_meeting[case.mid] = sc
            fired[case.mid] = stats

        cp = sum(per_meeting[c.mid]["cp_errors"] for c in cases) / max(
            1, sum(per_meeting[c.mid]["cp_words"] for c in cases)) * 100.0
        wer = sum(per_meeting[c.mid]["wer"] for c in cases) / len(cases)
        spk = sum(per_meeting[c.mid]["speakers"] for c in cases)
        relabeled = sum(f.get("total", 0) for f in fired.values())

        results.append({"combo": combo, "cpwer": cp, "wer": wer, "speakers": spk,
                        "relabeled": relabeled, "sig": tuple(sigs)})

        if i % 200 == 0 or i == len(combos):
            print(f"  {i}/{len(combos)}  ({time.time() - t0:.1f}s, "
                  f"{len(memo)} benzersiz sonuç)", flush=True)

    # Aynı çıktıyı veren yapılandırmaları tekilleştir (en gevşek olanı temsilci al)
    unique: dict = {}
    for r in results:
        if r["sig"] not in unique:
            unique[r["sig"]] = r
    distinct = list(unique.values())
    distinct.sort(key=lambda r: (r["cpwer"], r["speakers"]))
    print(f"\n{len(distinct)} BENZERSİZ sonuç ({len(combos)} kombinasyondan).")

    # --- DER: BENZERSİZ sonuçların TAMAMI için (genelde az sayıda olur) ---
    #
    # ÖNEMLİ: yalnız cpWER'e göre sıralamak YANILTICIDIR. cpWER, hipotez
    # konuşmacısı referanstan fazlaysa eşleşmeyenleri komple "insertion" sayar;
    # bu yüzden konuşmacıları YANLIŞ da olsa birleştirip sayıyı referansa
    # eşitlemek cpWER'i mekanik olarak düşürür. DER confusion ise yanlış
    # birleştirmeyi CEZALANDIRIR — asıl hakem odur.
    der_ok = True
    for r in distinct[: args.der_top]:
        cfg = build_config(r["combo"], args.passes)
        ders = []
        for case in cases:
            refined, _ = refine_speakers(case.segments, cfg)
            d = case.score_der(refined)
            if d is None:
                der_ok = False
                break
            ders.append(d)
        if not der_ok:
            break
        r["der"] = sum(d["der"] for d in ders) / len(ders)
        r["conf"] = sum(d["conf"] for d in ders) / len(ders)
        r["missed"] = sum(d["missed"] for d in ders) / len(ders)
    top = distinct[: max(args.top, args.der_top)]

    # baseline DER
    base_der = base_conf = None
    if der_ok:
        ds = [c.score_der(c.segments) for c in cases]
        if all(d is not None for d in ds):
            base_der = sum(d["der"] for d in ds) / len(ds)
            base_conf = sum(d["conf"] for d in ds) / len(ds)

    # --- Rapor ---
    ref_total = sum(c.n_ref_speakers for c in cases)
    lines = [
        "# Refinement Threshold Sweep (offline, deterministic)",
        "",
        f"- Meetings: **{', '.join(meetings)}**  |  ref speakers: {ref_total}",
        f"- Grid: {len(combos)} combos -> **{len(distinct)} distinct outcomes**  "
        f"| passes={args.passes} | {time.time() - t0:.1f}s",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Baseline (no refinement)",
        "",
        f"- cpWER **{base_cp:.2f}** | speakers **{base_spk}** / {ref_total} ref"
        + (f" | DER **{base_der:.2f}** | Conf **{base_conf:.2f}**" if base_der is not None else ""),
        "",
        "## Top configurations (ranked by cpWER)",
        "",
        "| # | cpWER | Δ | DER | Conf | Spk | Relabeled | tiny(dur/seg/isl/share) | small(dur/seg) | unk(dur/seg) |",
        "|--:|------:|--:|----:|-----:|----:|----------:|---|---|---|",
    ]
    for i, r in enumerate(top[: args.top], 1):
        c = r["combo"]
        der = f"{r['der']:.2f}" if "der" in r else "—"
        conf = f"{r['conf']:.2f}" if "conf" in r else "—"
        lines.append(
            f"| {i} | {r['cpwer']:.2f} | {r['cpwer'] - base_cp:+.2f} | {der} | {conf} | "
            f"{r['speakers']} | {r['relabeled']} | "
            f"{c['tiny_fragmented_max_duration']:g}/{c['tiny_fragmented_max_segments']}/"
            f"{c['tiny_fragmented_max_islands']}/{c['tiny_fragmented_min_neighbor_share']:g} | "
            f"{c['small_island_max_duration']:g}/{c['small_island_max_segments']} | "
            f"{c['unknown_fill_max_duration']:g}/{c['unknown_fill_max_segments']} |"
        )

    def _apply_cmd(combo: dict) -> str:
        return (
            "python scripts/refine_experiment.py "
            + (f"--only {' '.join(meetings)} " if args.only else "")
            + f"--passes {args.passes} "
            f"--tiny-max-duration {combo['tiny_fragmented_max_duration']:g} "
            f"--tiny-max-segments {combo['tiny_fragmented_max_segments']} "
            f"--tiny-max-islands {combo['tiny_fragmented_max_islands']} "
            f"--tiny-min-share {combo['tiny_fragmented_min_neighbor_share']:g} "
            f"--small-max-duration {combo['small_island_max_duration']:g} "
            f"--small-max-segments {combo['small_island_max_segments']} "
            f"--unknown-max-duration {combo['unknown_fill_max_duration']:g} "
            f"--unknown-max-segments {combo['unknown_fill_max_segments']}"
        )

    # --- DER'e göre ikinci sıralama (asıl hakem) ---
    scored_der = [r for r in distinct if "der" in r]
    if scored_der:
        by_der = sorted(scored_der, key=lambda r: (r["conf"], r["der"]))
        lines += [
            "", "## Ranked by DER confusion (the unbiased judge)", "",
            "cpWER, hipotez konuşmacı sayısı referansa eşitlendiğinde mekanik olarak "
            "düşer — YANLIŞ birleştirme bile onu iyileştirebilir. DER confusion ise "
            "yanlış atamayı cezalandırır. Çelişirlerse **confusion'a güvenin**.",
            "",
            "| # | Conf | Δ | DER | cpWER | Spk | Relabeled | tiny(dur/seg/isl/share) |",
            "|--:|-----:|--:|----:|------:|----:|----------:|---|",
        ]
        for i, r in enumerate(by_der[: args.top], 1):
            c = r["combo"]
            dconf = f"{r['conf'] - base_conf:+.2f}" if base_conf is not None else "—"
            lines.append(
                f"| {i} | {r['conf']:.2f} | {dconf} | {r['der']:.2f} | {r['cpwer']:.2f} | "
                f"{r['speakers']} | {r['relabeled']} | "
                f"{c['tiny_fragmented_max_duration']:g}/{c['tiny_fragmented_max_segments']}/"
                f"{c['tiny_fragmented_max_islands']}/{c['tiny_fragmented_min_neighbor_share']:g} |"
            )
        bd = by_der[0]
        lines += [
            "", "### Best by confusion", "",
            f"Conf **{bd['conf']:.2f}**"
            + (f" ({bd['conf'] - base_conf:+.2f})" if base_conf is not None else "")
            + f", DER **{bd['der']:.2f}**, cpWER {bd['cpwer']:.2f}, "
            f"speakers {bd['speakers']}/{ref_total}.",
            "", "```", _apply_cmd(bd["combo"]), "```",
        ]

    best = top[0]
    lines += [
        "", "## Best by cpWER (read with the caveat above)", "",
        f"cpWER **{best['cpwer']:.2f}** ({best['cpwer'] - base_cp:+.2f} vs baseline), "
        f"speakers **{best['speakers']}**/{ref_total}, "
        f"{best['relabeled']} segments relabeled.",
        "", "```", _apply_cmd(best["combo"]), "```",
        "",
        "WER bu taramada DEĞİŞMEZ (refinement metne dokunmaz) — sabit kalması "
        "uygulamanın doğruluğunun kontrolüdür.",
        "",
        "> UYARI: eşikler tek bir toplantı üzerinde seçilirse AŞIRI UYUM riski "
        "yüksektir. Kazanan yapılandırmayı en az 2-3 BAŞKA toplantıda doğrulayın "
        "(`ami_replay --only <mid>` ile segment üretip taramayı tekrar koşun).",
    ]
    if not der_ok:
        lines += ["", "> DER hesaplanamadı (pyannote.metrics yok). Sıralama yalnız "
                  "cpWER'e göredir — bu YANILTICI olabilir, bkz. yukarıdaki uyarı."]

    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print(report)
    print("=" * 72)
    print(f"Rapor: {args.report.resolve()}")


if __name__ == "__main__":
    main()
