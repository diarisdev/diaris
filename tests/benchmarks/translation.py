"""FLORES-200 çeviri benchmark'ı — Google / DeepL / yerel NLLB.

Çeviri motorlarını AYNI FLORES-200 cümle kümesi üzerinde çalıştırır ve
sacreBLEU + chrF++ + COMET ile skorlar; akademik makalelerle (özellikle
NLLB-200) karşılaştırılabilir bir tablo üretir.

Çalıştırma (proje kökünden):
    python -m tests.benchmarks.translation --pairs en-tr --limit 100
    python -m tests.benchmarks.translation --pairs en-tr,tr-en --engines google,ctranslate2
    python -m tests.benchmarks.translation --pairs en-tr --no-comet      # hızlı (COMET'siz)

Notlar:
  * Varsayılan split devtest (1012 cümle) — makalelerin raporladığı split.
  * Google (ücretsiz uç nokta) ve DeepL (API anahtarı) çok sayıda ağ çağrısı
    yapar; yerel NLLB (ctranslate2) ağ gerektirmez. Hız için --limit kullanın.
  * COMET ilk çalıştırmada ~2GB model indirir, CPU'da yavaştır (--no-comet ile atla).
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.dataset_managers import FloresDatasetManager
from tests.metrics import evaluate_translation
from src.translation.engine import get_translation_engine


def _translate_batched(engine, sources, sl, tl, batch_size, label):
    """Online motorların URL/uzunluk limitine takılmaması için parti-parti çevirir.

    translate_many girişle aynı uzunlukta hizalı çıktı garantiler; partiler
    birleştirilince tam liste yine hizalı kalır.
    """
    out = []
    n = len(sources)
    for i in range(0, n, batch_size):
        batch = sources[i:i + batch_size]
        out.extend(engine.translate_many(batch, source_lang=sl, target_lang=tl))
        done = min(i + batch_size, n)
        print(f"\r    {label}: {done}/{n}", end="", flush=True)
    print()
    return out


def _engine_status(name, engine):
    """(uygun_mu, sebep) — motor GERÇEKTEN kendisi mi çeviriyor yoksa Google'a mı düşüyor?

    Fallback'i tespit eder ki üç motor sessizce aynı Google çıktısını üretmesin.
    """
    name = name.lower()
    if name == "deepl":
        if not getattr(engine, "api_key", None):
            return False, "DeepL API anahtarı yok (--api-key veya DEEPL_API_KEY); Google'a düşer"
    if name in ("ctranslate2", "nllb"):
        if getattr(engine, "translator", None) is None:
            return False, ("NLLB modeli/kütüphanesi yüklenemedi "
                           "(pip install ctranslate2 sentencepiece + model yolu); Google'a düşer")
    return True, ""


def _make_engine(name, model_path=None, api_key=None):
    name = name.lower()
    if name in ("ctranslate2", "nllb"):
        path = model_path or os.path.join(PROJECT_ROOT, "models",
                                          "ctranslate2-nllb-200-distilled-600M")
        return get_translation_engine("ctranslate2", model_path=path)
    if name == "deepl":
        return get_translation_engine("deepl", api_key=api_key)
    return get_translation_engine("google")


def run(pairs, engines, split="devtest", limit=None, model_path=None,
        api_key=None, with_comet=True, batch_size=8):
    fm = FloresDatasetManager()
    if not fm.is_downloaded():
        print("📥 FLORES-200 bulunamadı, indiriliyor...")
        if not fm.download():
            print("❌ FLORES indirilemedi.")
            return

    rows = []   # (engine, direction, n, bleu, chrf, comet)
    for direction in pairs:
        src_lang, tgt_lang = direction.split("-")
        data = fm.get_pairs(src_lang, tgt_lang, split=split, limit=limit)
        if not data:
            print(f"⚠️  {direction}: çift bulunamadı, atlanıyor.")
            continue
        sources = [d["source"] for d in data]
        references = [d["reference"] for d in data]
        print(f"\n=== {direction}  ({len(sources)} cümle, split={split}) ===")

        for eng_name in engines:
            print(f"  [{eng_name}] çevriliyor...", flush=True)
            engine = _make_engine(eng_name, model_path=model_path, api_key=api_key)
            ok, reason = _engine_status(eng_name, engine)
            if not ok:
                print(f"    ⚠️  {eng_name} ATLANDI — {reason}")
                continue
            try:
                hyps = _translate_batched(engine, sources, src_lang, tgt_lang,
                                          batch_size, eng_name)
            except Exception as e:
                print(f"    ❌ {eng_name} çeviri hatası: {e}")
                continue
            res = evaluate_translation(sources, hyps, references, with_comet=with_comet)
            rows.append((eng_name, direction, res.n_segments, res.bleu, res.chrf, res.comet))
            print(f"    BLEU={_fmt(res.bleu)}  chrF++={_fmt(res.chrf)}  COMET={_fmt(res.comet, 4)}")

    _print_table(rows)


def _fmt(v, nd=2):
    return "  n/a" if v is None else f"{v:.{nd}f}"


def _print_table(rows):
    print("\n" + "=" * 72)
    print("📊  ÇEVİRİ KALİTE TABLOSU (FLORES-200)")
    print("=" * 72)
    print(f"{'Engine':<14}{'Yön':<10}{'N':>6}{'BLEU':>9}{'chrF++':>9}{'COMET':>10}")
    print("-" * 72)
    for eng, direction, n, bleu, chrf, comet in rows:
        print(f"{eng:<14}{direction:<10}{n:>6}{_fmt(bleu):>9}{_fmt(chrf):>9}{_fmt(comet, 4):>10}")
    print("=" * 72)
    print("BLEU/chrF++ 0–100 (yüksek=iyi); COMET ~0–1 (yüksek=iyi).")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="FLORES-200 çeviri benchmark'ı.")
    ap.add_argument("--pairs", default="en-tr",
                    help="Virgülle ayrılmış yönler, örn. 'en-tr,tr-en' (varsayılan: en-tr).")
    ap.add_argument("--engines", default="google,deepl,ctranslate2",
                    help="Virgülle ayrılmış motorlar (google,deepl,ctranslate2).")
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--limit", type=int, default=None, help="Cümle sayısını sınırla (hız için).")
    ap.add_argument("--model-path", default=None, help="Yerel NLLB CTranslate2 model yolu.")
    ap.add_argument("--api-key", default=None, help="DeepL API anahtarı (yoksa Google'a düşer).")
    ap.add_argument("--no-comet", action="store_true", help="COMET'i atla (hızlı).")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Çeviri parti boyutu (online motorlar için küçük tut; varsayılan: 8).")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    run(pairs, engines, split=args.split, limit=args.limit,
        model_path=args.model_path, api_key=args.api_key,
        with_comet=not args.no_comet, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
