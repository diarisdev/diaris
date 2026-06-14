"""Çeviri kalitesi metrikleri — sacreBLEU + chrF++ + COMET.

FLORES-200 gibi paralel referanslı çeviri benchmark'ları için. Akademik
makalelerle karşılaştırılabilir standart yığın:
  * sacreBLEU  — tekrarlanabilir BLEU (her makalenin raporladığı temel).
  * chrF++     — karakter-n-gram F skoru; morfolojik diller (Türkçe) için adil.
  * COMET      — nöral, insan-yargısıyla en yüksek korelasyon (WMT/NLLB headline).
                 Model: Unbabel/wmt22-comet-da (referans tabanlı).

Skorlar KORPUS seviyesindedir (tüm test seti üzerinden tek skor), cümle başına
ortalama DEĞİL. BLEU/chrF 0–100; COMET ~0–1.

Kurulum: pip install sacrebleu unbabel-comet
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranslationResult:
    bleu: float | None = None
    chrf: float | None = None     # chrF++
    comet: float | None = None
    n_segments: int = 0


def compute_bleu(hypotheses, references) -> float:
    """Korpus sacreBLEU (0–100). references: tek referans listesi."""
    import sacrebleu
    return sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score


def compute_chrf(hypotheses, references, word_order: int = 2) -> float:
    """Korpus chrF (word_order=2 -> chrF++). references: tek referans listesi."""
    import sacrebleu
    return sacrebleu.corpus_chrf(
        list(hypotheses), [list(references)], word_order=word_order
    ).score


def compute_comet(sources, hypotheses, references,
                  model_name: str = "Unbabel/wmt22-comet-da",
                  gpus: int = 0, batch_size: int = 8, model_dir=None) -> float:
    """Referans tabanlı COMET (sistem-seviyesi skor).

    İlk çağrıda modeli indirir (~2GB). model_dir verilirse global HF cache yerine
    o dizine indirir (proje-yerel). CPU'da yavaştır ama offline raporlama için uygun.
    """
    from comet import download_model, load_from_checkpoint
    # saving_directory: modeli global ~/.cache yerine proje-yerel dizine indir
    ckpt = download_model(model_name, saving_directory=model_dir)
    model = load_from_checkpoint(ckpt)
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)
    # comet sürümüne göre system_score ya da ["system_score"]
    return float(getattr(out, "system_score", None) or out["system_score"])


def evaluate_translation(sources, hypotheses, references, with_comet: bool = True,
                         comet_model: str = "Unbabel/wmt22-comet-da",
                         gpus: int = 0, comet_dir=None) -> TranslationResult:
    """sacreBLEU + chrF++ (+ opsiyonel COMET) hepsini tek çağrıda hesaplar.

    Eksik kütüphaneler sessizce atlanır (ilgili alan None kalır) ki kısmi
    sonuç yine de raporlanabilsin. comet_dir: COMET modelinin indirileceği
    proje-yerel dizin (None ise global HF cache).
    """
    hyps = list(hypotheses)
    refs = list(references)
    srcs = list(sources)
    res = TranslationResult(n_segments=len(hyps))

    try:
        res.bleu = compute_bleu(hyps, refs)
        res.chrf = compute_chrf(hyps, refs)
    except ImportError:
        print("WARNING: `sacrebleu` kurulu değil (pip install sacrebleu); "
              "BLEU/chrF atlandı.")

    if with_comet:
        try:
            res.comet = compute_comet(srcs, hyps, refs, model_name=comet_model,
                                      gpus=gpus, model_dir=comet_dir)
        except ImportError:
            print("WARNING: `unbabel-comet` kurulu değil (pip install unbabel-comet); "
                  "COMET atlandı.")

    return res
