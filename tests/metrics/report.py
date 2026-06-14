"""Sınıf-tabanlı değerlendirme aggregator'ları (rapor toplama).

`TranscriptionEvaluator` ve `DiarizationEvaluator`: benchmark runner'larının
beklediği API (evaluate / record_failure / print_report / export_csv). Metrik
MATEMATİĞİ artık fonksiyonel çekirdekten (edit_distance, wer) gelir; bu modül
sadece örnekleri biriktirip raporlar.
"""

from dataclasses import dataclass, field

from .edit_distance import levenshtein_sid
from .wer import normalize_text as _normalize_text

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    HAS_PYANNOTE_METRICS = True
except ImportError:
    HAS_PYANNOTE_METRICS = False


@dataclass
class EvalResult:
    """Tek bir örneğin değerlendirme sonucu."""
    reference: str
    hypothesis: str
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    ref_word_count: int
    audio_path: str = ""
    speaker_id: str = ""
    duration: float = 0.0


@dataclass
class BenchmarkReport:
    """Toplu benchmark raporu."""
    results: list = field(default_factory=list)
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0

    @property
    def avg_wer(self):
        if not self.results:
            return 0.0
        return sum(r.wer for r in self.results) / len(self.results)

    @property
    def avg_cer(self):
        if not self.results:
            return 0.0
        return sum(r.cer for r in self.results) / len(self.results)

    @property
    def weighted_wer(self):
        total_errors = sum(r.insertions + r.deletions + r.substitutions for r in self.results)
        total_ref_words = sum(r.ref_word_count for r in self.results)
        if total_ref_words == 0:
            return 0.0
        return total_errors / total_ref_words

    @property
    def total_duration(self):
        return sum(r.duration for r in self.results)

    @property
    def best_result(self):
        return min(self.results, key=lambda r: r.wer) if self.results else None

    @property
    def worst_result(self):
        return max(self.results, key=lambda r: r.wer) if self.results else None


@dataclass
class DiarizationEvalResult:
    meeting_id: str
    der: float
    false_alarm: float
    missed_detection: float
    confusion: float
    duration: float


@dataclass
class DiarizationReport:
    results: list = field(default_factory=list)
    total_meetings: int = 0

    @property
    def total_duration(self):
        return sum(r.duration for r in self.results)

    @property
    def avg_der(self):
        if not self.results:
            return 0.0
        return sum(r.der for r in self.results) / len(self.results)

    @property
    def avg_false_alarm(self):
        if not self.results:
            return 0.0
        return sum(r.false_alarm for r in self.results) / len(self.results)

    @property
    def avg_missed_detection(self):
        if not self.results:
            return 0.0
        return sum(r.missed_detection for r in self.results) / len(self.results)

    @property
    def avg_confusion(self):
        if not self.results:
            return 0.0
        return sum(r.confusion for r in self.results) / len(self.results)


class TranscriptionEvaluator:
    """Transkripsiyon doğruluk değerlendirme + rapor toplama motoru."""

    def __init__(self, use_jiwer=False):
        self.use_jiwer = use_jiwer
        self._jiwer = None
        if use_jiwer:
            try:
                import jiwer
                self._jiwer = jiwer
            except ImportError:
                print("⚠️  jiwer bulunamadı, dahili hesaplama kullanılacak.")
                self.use_jiwer = False
        self.report = BenchmarkReport()

    def normalize_text(self, text):
        return _normalize_text(text)

    @staticmethod
    def _levenshtein_ops(ref, hyp):
        """Geriye dönük uyum: artık ortak çekirdeğe (edit_distance) delege eder."""
        return levenshtein_sid(ref, hyp)

    def evaluate(self, reference, hypothesis, audio_path="", speaker_id="", duration=0.0):
        ref_norm = self.normalize_text(reference)
        hyp_norm = self.normalize_text(hypothesis)
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        if self.use_jiwer and self._jiwer:
            measures = self._jiwer.compute_measures(ref_norm, hyp_norm)
            wer = measures["wer"]
            insertions = measures["insertions"]
            deletions = measures["deletions"]
            substitutions = measures["substitutions"]
        else:
            substitutions, insertions, deletions = self._levenshtein_ops(ref_words, hyp_words)
            total_errors = substitutions + insertions + deletions
            wer = total_errors / len(ref_words) if ref_words else 0.0

        ref_chars = list(ref_norm.replace(" ", ""))
        hyp_chars = list(hyp_norm.replace(" ", ""))
        c_sub, c_ins, c_del = self._levenshtein_ops(ref_chars, hyp_chars)
        cer = (c_sub + c_ins + c_del) / len(ref_chars) if ref_chars else 0.0

        result = EvalResult(
            reference=reference, hypothesis=hypothesis,
            wer=round(wer, 4), cer=round(cer, 4),
            insertions=insertions, deletions=deletions, substitutions=substitutions,
            ref_word_count=len(ref_words), audio_path=audio_path,
            speaker_id=speaker_id, duration=duration,
        )
        self.report.results.append(result)
        self.report.successful_samples += 1
        self.report.total_samples += 1
        return result

    def record_failure(self, audio_path="", reason=""):
        self.report.failed_samples += 1
        self.report.total_samples += 1

    def print_report(self, top_n_worst=5, top_n_best=5):
        report = self.report
        print("\n" + "=" * 70)
        print("📊  BENCHMARK RAPORU — Transkripsiyon Doğruluk Testi")
        print("=" * 70)
        print(f"\n📋 Genel Bilgiler:")
        print(f"   Toplam Örnek:       {report.total_samples}")
        print(f"   Başarılı:           {report.successful_samples}")
        print(f"   Başarısız:          {report.failed_samples}")
        print(f"   Toplam Ses Süresi:  {report.total_duration / 60:.1f} dakika")
        if not report.results:
            print("\n⚠️  Değerlendirilebilecek sonuç yok.")
            return
        print(f"\n📈 Doğruluk Metrikleri:")
        print(f"   Ortalama WER:       {report.avg_wer * 100:.2f}%")
        print(f"   Ağırlıklı WER:     {report.weighted_wer * 100:.2f}%")
        print(f"   Ortalama CER:       {report.avg_cer * 100:.2f}%")
        print(f"   Kelime Doğruluğu:   {(1 - report.avg_wer) * 100:.2f}%")
        print(f"   Karakter Doğruluğu: {(1 - report.avg_cer) * 100:.2f}%")

        wers = [r.wer for r in report.results]
        perfect = sum(1 for w in wers if w == 0.0)
        low = sum(1 for w in wers if 0 < w <= 0.1)
        mid = sum(1 for w in wers if 0.1 < w <= 0.3)
        high = sum(1 for w in wers if w > 0.3)
        print(f"\n📊 WER Dağılımı:")
        print(f"   Mükemmel (0%):      {perfect} örnek")
        print(f"   Düşük (0-10%):      {low} örnek")
        print(f"   Orta (10-30%):      {mid} örnek")
        print(f"   Yüksek (>30%):      {high} örnek")

        sorted_results = sorted(report.results, key=lambda r: r.wer)
        print(f"\n✅ En İyi {top_n_best} Sonuç:")
        print("-" * 70)
        for r in sorted_results[:top_n_best]:
            self._print_single_result(r)
        print(f"\n❌ En Kötü {top_n_worst} Sonuç:")
        print("-" * 70)
        for r in sorted_results[-top_n_worst:]:
            self._print_single_result(r)
        print("\n" + "=" * 70)

    def export_csv(self, filepath):
        import csv
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "audio_path", "speaker_id", "duration",
                "wer", "cer", "insertions", "deletions", "substitutions",
                "ref_word_count", "reference", "hypothesis"
            ])
            for r in self.report.results:
                writer.writerow([
                    r.audio_path, r.speaker_id, r.duration,
                    r.wer, r.cer, r.insertions, r.deletions, r.substitutions,
                    r.ref_word_count, r.reference, r.hypothesis
                ])
        print(f"📄 CSV raporu kaydedildi: {filepath}")

    def _print_single_result(self, result):
        import os
        fname = os.path.basename(result.audio_path) if result.audio_path else "N/A"
        print(f"   📁 {fname} | WER: {result.wer*100:.1f}% | CER: {result.cer*100:.1f}% | Süre: {result.duration:.1f}s")
        print(f"      REF: {result.reference[:80]}")
        print(f"      HYP: {result.hypothesis[:80]}")
        print()


class DiarizationEvaluator:
    """Diarization doğruluk değerlendirme (DER) + rapor toplama."""

    def __init__(self):
        if not HAS_PYANNOTE_METRICS:
            print("⚠️ pyannote.metrics bulunamadı! pip install pyannote.metrics")
        self.report = DiarizationReport()
        self.metric = DiarizationErrorRate() if HAS_PYANNOTE_METRICS else None

    def _create_annotation(self, intervals, uri="meeting"):
        ann = Annotation(uri=uri)
        for interval in intervals:
            ann[Segment(interval["start"], interval["end"])] = interval["speaker"]
        return ann

    def evaluate(self, meeting_id, reference_intervals, hypothesis_intervals, duration):
        if not self.metric:
            return None
        ref_ann = self._create_annotation(reference_intervals, uri=meeting_id)
        hyp_ann = self._create_annotation(hypothesis_intervals, uri=meeting_id)
        self.metric.optimal_mapping(ref_ann, hyp_ann)
        der = self.metric(ref_ann, hyp_ann, detailed=True)
        total = der["total"] or 1e-8
        result = DiarizationEvalResult(
            meeting_id=meeting_id,
            der=der["diarization error rate"],
            false_alarm=der["false alarm"] / total,
            missed_detection=der["missed detection"] / total,
            confusion=der["confusion"] / total,
            duration=duration,
        )
        self.report.results.append(result)
        self.report.total_meetings += 1
        return result

    def print_report(self):
        report = self.report
        print("\n" + "=" * 70)
        print("📊  BENCHMARK RAPORU — Diarization Doğruluk Testi (DER)")
        print("=" * 70)
        print(f"\n📋 Genel Bilgiler:")
        print(f"   Toplam Meeting:     {report.total_meetings}")
        print(f"   Toplam Ses Süresi:  {report.total_duration / 60:.1f} dakika")
        if not report.results:
            print("\n⚠️  Değerlendirilebilecek sonuç yok.")
            return
        print(f"\n📈 Doğruluk Metrikleri (Ortalama):")
        print(f"   Diarization Error Rate (DER): {report.avg_der * 100:.2f}%")
        print(f"   ├─ False Alarm:               {report.avg_false_alarm * 100:.2f}%")
        print(f"   ├─ Missed Detection:          {report.avg_missed_detection * 100:.2f}%")
        print(f"   └─ Speaker Confusion:         {report.avg_confusion * 100:.2f}%")
        print(f"\n✅ Detaylı Sonuçlar:")
        print("-" * 70)
        for r in report.results:
            print(f"   📁 {r.meeting_id} | DER: {r.der*100:.1f}% (FA: {r.false_alarm*100:.1f}%, "
                  f"Miss: {r.missed_detection*100:.1f}%, Conf: {r.confusion*100:.1f}%) | Süre: {r.duration:.1f}s")
        print("\n" + "=" * 70)
