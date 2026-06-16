"""VAD parametre taraması — en iyi (aggressiveness, silero threshold) kombinasyonu.

Her kombinasyon için CHiME-6 **streaming** benchmark'ını ayrı bir alt-süreçte
çalıştırır (VAD_AGGRESSIVENESS / SILERO_THRESHOLD env ile geçirilir), Average
Global WER'i toplar; CSV + 3B surface grafiği üretir.

NEDEN STREAMING? VAD yalnızca streaming modda devreye girer (chunk sınırlarını
belirler). Oracle modda ground-truth segmentler kullanılır, VAD hiç çalışmaz →
tüm kombinasyonlar aynı WER'i verir. Bu yüzden tarama zorunlu olarak streaming.

NOTLAR:
  * VAD_AGGRESSIVENESS WebRTC'de yalnızca tamsayı 0/1/2/3 alır (0.1 adım yok).
  * SILERO_THRESHOLD 0-1 arası float; --thr-step ile adım ayarlanır.
  * config env'i import anında okunduğu için her kombinasyon AYRI süreçte koşar.
  * GPU transkripsiyonu koşudan-koşuya ~±1-2 WER oynayabilir; yüzey TRENDİ gösterir,
    ondalık optima gürültülüdür. --repeats ile her hücre ortalanabilir.

Çalıştırma (proje kökünden):
    python scripts/vad_sweep.py --session S02 --limit-minutes 10
    python scripts/vad_sweep.py --thr-step 0.2 --limit-minutes 5      # daha hızlı
    python scripts/vad_sweep.py --repeats 2                            # gürültüyü azalt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(aggr: int, thr: float, session: str, minutes: float) -> float:
    """Tek bir (aggr, thr) kombinasyonu için Average Global WER (%) döndürür."""
    env = dict(os.environ)
    env["VAD_AGGRESSIVENESS"] = str(aggr)
    env["SILERO_THRESHOLD"] = f"{thr:.2f}"
    env["PYTHONUTF8"] = "1"  # alt-süreçte emoji çıktısı cp1254'te çökmesin

    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out_path = tf.name
    tf.close()
    try:
        cmd = [
            sys.executable, "-m", "tests.benchmarks.chime6",
            "--session", session, "--mode", "worn",
            "--segmentation", "streaming", "--limit-minutes", str(minutes),
            "--output", out_path,
        ]
        # Alt-süreç PYTHONUTF8=1 ile UTF-8 basar; yakalamayı da UTF-8'e sabitle
        # (varsayılan cp1254 çözümü reader thread'ini çökertiyordu). errors=replace
        # güvenlik ağı.
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not os.path.getsize(out_path):
            tail = (proc.stderr or "")[-500:]
            raise RuntimeError(f"benchmark başarısız (rc={proc.returncode}). stderr: {tail}")
        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        return data["summary"]["avg_wer"] * 100.0
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _thresholds(thr_min: float, thr_max: float, step: float) -> list[float]:
    out = []
    n = int(round((thr_max - thr_min) / step)) + 1
    for i in range(n):
        out.append(round(thr_min + i * step, 2))
    return out


def _plot_surface(aggrs, thrs, grid, png_path, best):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X, Y = np.meshgrid(aggrs, thrs)  # X=aggressiveness, Y=silero threshold
    Z = np.array([[grid.get((a, t), np.nan) for a in aggrs] for t in thrs])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis_r", edgecolor="k",
                           linewidth=0.3, antialiased=True, alpha=0.92)
    # En iyi noktayı işaretle
    if best:
        ax.scatter([best[0]], [best[1]], [best[2]], color="red", s=80,
                   depthshade=False, label=f"en iyi: aggr={best[0]}, thr={best[1]:.2f}, WER={best[2]:.2f}%")
        ax.legend(loc="upper left")
    ax.set_xlabel("VAD aggressiveness (0-3)")
    ax.set_ylabel("Silero threshold")
    ax.set_zlabel("Avg Global WER (%)")
    ax.set_title("VAD parametre taraması — DÜŞÜK WER = İYİ")
    ax.set_xticks(aggrs)
    fig.colorbar(surf, shrink=0.5, label="WER %")
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="VAD (aggressiveness, silero threshold) taraması.")
    ap.add_argument("--session", default="S02")
    ap.add_argument("--limit-minutes", type=float, default=10.0)
    ap.add_argument("--aggr", default="0,1,2,3", help="Taranacak aggressiveness değerleri (vir.).")
    ap.add_argument("--thr-min", type=float, default=0.1)
    ap.add_argument("--thr-max", type=float, default=0.9)
    ap.add_argument("--thr-step", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=1, help="Her hücreyi N kez koşup ortala (gürültü için).")
    ap.add_argument("--out", default="output/vad_sweep", help="Çıktı taban yolu (.csv/.png eklenir).")
    args = ap.parse_args()

    aggrs = [int(x) for x in args.aggr.split(",") if x.strip() != ""]
    thrs = _thresholds(args.thr_min, args.thr_max, args.thr_step)

    total = len(aggrs) * len(thrs)
    est_min = total * args.repeats * (args.limit_minutes * 0.2 + 0.5)
    print(f"Tarama: {len(aggrs)} aggressiveness × {len(thrs)} threshold = {total} hücre "
          f"× {args.repeats} tekrar. Kaba süre tahmini ~{est_min:.0f} dk.\n")

    grid = {}
    rows = []
    i = 0
    for aggr in aggrs:
        for thr in thrs:
            i += 1
            print(f"[{i}/{total}] aggr={aggr} thr={thr:.2f} ...", flush=True)
            vals = []
            for r in range(args.repeats):
                try:
                    w = run_one(aggr, thr, args.session, args.limit_minutes)
                    vals.append(w)
                    print(f"    koşu {r+1}/{args.repeats}: WER={w:.2f}%")
                except Exception as e:
                    print(f"    koşu {r+1}/{args.repeats} HATA: {e}")
            wer = sum(vals) / len(vals) if vals else float("nan")
            grid[(aggr, thr)] = wer
            rows.append((aggr, thr, wer))
            print(f"    => ortalama WER={wer:.2f}%")

    # CSV
    out_base = ROOT / args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = f"{out_base}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("aggressiveness,silero_threshold,avg_global_wer\n")
        for a, t, w in rows:
            f.write(f"{a},{t:.2f},{w:.4f}\n")
    print(f"\nCSV kaydedildi: {csv_path}")

    # En iyi (en düşük WER)
    valid = [(a, t, w) for a, t, w in rows if w == w]  # nan ele
    best = min(valid, key=lambda x: x[2]) if valid else None
    if best:
        print(f"🏆 EN İYİ: aggressiveness={best[0]}, silero_threshold={best[1]:.2f}, "
              f"Avg Global WER={best[2]:.2f}%")

    # Konsol tablosu
    print("\nWER tablosu (satır=threshold, sütun=aggressiveness):")
    print("thr\\aggr | " + " ".join(f"{a:>7}" for a in aggrs))
    for t in thrs:
        print(f"  {t:.2f}   | " + " ".join(f"{grid.get((a,t), float('nan')):>7.2f}" for a in aggrs))

    # Surface grafiği
    png_path = f"{out_base}.png"
    try:
        _plot_surface(aggrs, thrs, grid, png_path, best)
        print(f"\n📊 Surface grafiği: {png_path}")
    except Exception as e:
        print(f"\n⚠️ Grafik atlandı (matplotlib gerekli: pip install matplotlib): {e}")
        print("   CSV elde; grafiği sonradan da üretebilirsin.")


if __name__ == "__main__":
    main()
