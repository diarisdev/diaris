"""Metrik paketi — TÜM değerlendirme matematiğinin tek kaynağı.

Alt modüller:
  edit_distance : Levenshtein S/I/D çekirdeği
  wer           : WER / CER (+ normalize_text)
  cpwer         : cpWER (konuşmacı-bazlı, meeteval + pruning)
  der           : DER (pyannote, overlap dahil/hariç)
  translation   : BLEU / chrF++ / COMET
  report        : sınıf-tabanlı aggregator'lar (TranscriptionEvaluator, DiarizationEvaluator)
"""

from .edit_distance import levenshtein_sid
from .wer import normalize_text, compute_wer, compute_cer, WerResult, CerResult
from .cpwer import (
    cpwer_from_segments,
    cpwer_from_speaker_texts,
    CpwerResult,
)
from .der import compute_der, load_rttm, DerResult
from .translation import (
    compute_bleu,
    compute_chrf,
    compute_comet,
    evaluate_translation,
    TranslationResult,
)
from .report import (
    TranscriptionEvaluator,
    DiarizationEvaluator,
    EvalResult,
    BenchmarkReport,
    DiarizationEvalResult,
    DiarizationReport,
)

__all__ = [
    "levenshtein_sid",
    "normalize_text", "compute_wer", "compute_cer", "WerResult", "CerResult",
    "cpwer_from_segments", "cpwer_from_speaker_texts", "CpwerResult",
    "compute_der", "load_rttm", "DerResult",
    "compute_bleu", "compute_chrf", "compute_comet", "evaluate_translation", "TranslationResult",
    "TranscriptionEvaluator", "DiarizationEvaluator", "EvalResult", "BenchmarkReport",
    "DiarizationEvalResult", "DiarizationReport",
]
