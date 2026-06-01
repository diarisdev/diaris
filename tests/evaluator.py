"""
Transkripsiyon Doğruluk Değerlendirme Modülü.

Sistemin ürettiği transkripsiyon (hypothesis) ile gerçek transkripsiyon (reference)
arasındaki farkı ölçer.

Metrikler:
  - WER (Word Error Rate): Kelime Hata Oranı
  - CER (Character Error Rate): Karakter Hata Oranı
  - Insertion / Deletion / Substitution ayrıntıları

Kullanım:
    evaluator = TranscriptionEvaluator()
    result = evaluator.evaluate("hello world", "hello word")
    evaluator.print_report()
"""

import re
from dataclasses import dataclass, field

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    HAS_PYANNOTE_METRICS = True
except ImportError:
    HAS_PYANNOTE_METRICS = False


@dataclass
class EvalResult:
    """Tek bir örneğin değerlendirme sonucu."""
    reference: str          # Ground-truth transkripsiyon
    hypothesis: str         # Sistem çıktısı
    wer: float              # Word Error Rate (0.0 – 1.0+)
    cer: float              # Character Error Rate (0.0 – 1.0+)
    insertions: int         # Eklenen kelime sayısı
    deletions: int          # Silinen kelime sayısı
    substitutions: int      # Değiştirilen kelime sayısı
    ref_word_count: int     # Referans kelime sayısı
    audio_path: str = ""    # Ses dosyası yolu
    speaker_id: str = ""    # Konuşmacı ID'si
    duration: float = 0.0   # Ses süresi (saniye)


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
        """Kelime sayısına göre ağırlıklı WER (daha uzun örnekler daha fazla ağırlık)."""
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
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.wer)

    @property
    def worst_result(self):
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.wer)


@dataclass
class DiarizationEvalResult:
    """Tek bir örneğin diarization değerlendirme sonucu."""
    meeting_id: str
    der: float              # Diarization Error Rate
    false_alarm: float      # False Alarm Rate
    missed_detection: float # Missed Detection Rate
    confusion: float        # Speaker Confusion Rate
    duration: float         # Ses süresi


@dataclass
class DiarizationReport:
    """Toplu diarization benchmark raporu."""
    results: list = field(default_factory=list)
    total_meetings: int = 0
    
    @property
    def total_duration(self):
        return sum(r.duration for r in self.results)

    @property
    def avg_der(self):
        if not self.results: return 0.0
        return sum(r.der for r in self.results) / len(self.results)

    @property
    def avg_false_alarm(self):
        if not self.results: return 0.0
        return sum(r.false_alarm for r in self.results) / len(self.results)

    @property
    def avg_missed_detection(self):
        if not self.results: return 0.0
        return sum(r.missed_detection for r in self.results) / len(self.results)

    @property
    def avg_confusion(self):
        if not self.results: return 0.0
        return sum(r.confusion for r in self.results) / len(self.results)


class TranscriptionEvaluator:
    """
    Transkripsiyon doğruluk değerlendirme motoru.
    
    WER ve CER hesaplaması için Levenshtein mesafesi kullanır.
    Harici kütüphane gerektirmez — jiwer opsiyonel olarak kullanılabilir.
    """

    def __init__(self, use_jiwer=False):
        """
        Args:
            use_jiwer: True ise jiwer kütüphanesini kullanır (daha hassas).
                       False ise dahili hesaplama kullanılır.
        """
        self.use_jiwer = use_jiwer
        self._jiwer = None

        if use_jiwer:
            try:
                import jiwer
                self._jiwer = jiwer
            except ImportError:
                print("⚠️  jiwer bulunamadı, dahili hesaplama kullanılacak.")
                print("   Kurmak için: pip install jiwer")
                self.use_jiwer = False

        self.report = BenchmarkReport()

    def normalize_text(self, text):
        """
        Metin normalizasyonu — adil karşılaştırma için.
        
        - Küçük harfe çevirir
        - Noktalama işaretlerini kaldırır
        - Fazla boşlukları temizler
        - Sayıları kelime formuna çevirmez (basit tutuluyor)
        """
        text = text.lower().strip()
        # Noktalama kaldır
        text = re.sub(r'[^\w\s]', '', text)
        # Fazla boşlukları kaldır
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def evaluate(self, reference, hypothesis, audio_path="", speaker_id="", duration=0.0):
        """
        Tek bir referans-hipotez çiftini değerlendirir.

        Args:
            reference: Gerçek transkripsiyon (ground-truth)
            hypothesis: Sistem çıktısı
            audio_path: Ses dosyası yolu (isteğe bağlı, rapor için)
            speaker_id: Konuşmacı ID'si (isteğe bağlı, rapor için)
            duration: Ses süresi (isteğe bağlı, rapor için)

        Returns:
            EvalResult: Değerlendirme sonucu
        """
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

        # CER (karakter bazlı)
        ref_chars = list(ref_norm.replace(" ", ""))
        hyp_chars = list(hyp_norm.replace(" ", ""))
        c_sub, c_ins, c_del = self._levenshtein_ops(ref_chars, hyp_chars)
        cer = (c_sub + c_ins + c_del) / len(ref_chars) if ref_chars else 0.0

        result = EvalResult(
            reference=reference,
            hypothesis=hypothesis,
            wer=round(wer, 4),
            cer=round(cer, 4),
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions,
            ref_word_count=len(ref_words),
            audio_path=audio_path,
            speaker_id=speaker_id,
            duration=duration,
        )

        self.report.results.append(result)
        self.report.successful_samples += 1
        self.report.total_samples += 1

        return result

    def record_failure(self, audio_path="", reason=""):
        """Başarısız bir örneği kayıt eder (transkripsiyon üretilemedi)."""
        self.report.failed_samples += 1
        self.report.total_samples += 1

    def print_report(self, top_n_worst=5, top_n_best=5):
        """
        Toplu benchmark raporunu ekrana basar.

        Args:
            top_n_worst: En kötü kaç sonucu göster
            top_n_best: En iyi kaç sonucu göster
        """
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

        # WER dağılımı
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

        # En iyi sonuçlar
        sorted_results = sorted(report.results, key=lambda r: r.wer)

        print(f"\n✅ En İyi {top_n_best} Sonuç:")
        print("-" * 70)
        for r in sorted_results[:top_n_best]:
            self._print_single_result(r)

        # En kötü sonuçlar
        print(f"\n❌ En Kötü {top_n_worst} Sonuç:")
        print("-" * 70)
        for r in sorted_results[-top_n_worst:]:
            self._print_single_result(r)

        print("\n" + "=" * 70)

    def export_csv(self, filepath):
        """Sonuçları CSV dosyasına aktarır."""
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
        """Tek bir sonucu okunabilir formatta basar."""
        import os
        fname = os.path.basename(result.audio_path) if result.audio_path else "N/A"
        print(f"   📁 {fname} | WER: {result.wer*100:.1f}% | CER: {result.cer*100:.1f}% | Süre: {result.duration:.1f}s")
        print(f"      REF: {result.reference[:80]}")
        print(f"      HYP: {result.hypothesis[:80]}")
        print()

    @staticmethod
    def _levenshtein_ops(ref, hyp):
        """
        Levenshtein mesafesi üzerinden S/I/D sayılarını hesaplar.

        Dynamic Programming ile optimal hizalama bulur ve
        substitution, insertion, deletion sayılarını döndürür.

        Returns:
            tuple[int, int, int]: (substitutions, insertions, deletions)
        """
        n = len(ref)
        m = len(hyp)

        # DP tablosu: [i][j] = (toplam_hata, substitution, insertion, deletion)
        dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp[i][0] = (i, 0, 0, i)  # Tüm referans kelimeler silindi

        for j in range(1, m + 1):
            dp[0][j] = (j, 0, j, 0)  # Tüm hipotez kelimeler eklendi

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Substitution
                    sub = dp[i - 1][j - 1]
                    sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])

                    # Insertion (hipotezde fazla kelime)
                    ins = dp[i][j - 1]
                    ins_cost = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])

                    # Deletion (referansta kelime eksik)
                    dlt = dp[i - 1][j]
                    dlt_cost = (dlt[0] + 1, dlt[1], dlt[2], dlt[3] + 1)

                    dp[i][j] = min(sub_cost, ins_cost, dlt_cost, key=lambda x: x[0])

        _, subs, ins, dels = dp[n][m]
        return subs, ins, dels


class DiarizationEvaluator:
    """
    Diarization doğruluk değerlendirme motoru.
    pyannote.metrics kullanarak DER hesaplar.
    """
    def __init__(self):
        if not HAS_PYANNOTE_METRICS:
            print("⚠️ pyannote.metrics bulunamadı! Lütfen kurun: pip install pyannote.metrics")
        self.report = DiarizationReport()
        self.metric = DiarizationErrorRate() if HAS_PYANNOTE_METRICS else None

    def _create_annotation(self, intervals, uri="meeting"):
        """List of dicts -> pyannote.core.Annotation"""
        ann = Annotation(uri=uri)
        for interval in intervals:
            ann[Segment(interval["start"], interval["end"])] = interval["speaker"]
        return ann

    def evaluate(self, meeting_id, reference_intervals, hypothesis_intervals, duration):
        """
        Bir meeting için DER hesaplar.
        intervals: [{"start": float, "end": float, "speaker": str}, ...]
        """
        if not self.metric:
            return None

        ref_ann = self._create_annotation(reference_intervals, uri=meeting_id)
        hyp_ann = self._create_annotation(hypothesis_intervals, uri=meeting_id)

        # Diğer pyannote versiyonlarında optimal mapping için uem gerekebilir ama basitçe hesaplayalım
        mapping = self.metric.optimal_mapping(ref_ann, hyp_ann)
        
        # metric çağrısı componentleri biriktirir
        der = self.metric(ref_ann, hyp_ann, detailed=True)
        
        # detaylı çıktılar
        total = der["total"]
        if total == 0:
            total = 1e-8

        result = DiarizationEvalResult(
            meeting_id=meeting_id,
            der=der["diarization error rate"],
            false_alarm=der["false alarm"] / total,
            missed_detection=der["missed detection"] / total,
            confusion=der["confusion"] / total,
            duration=duration
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
            print(f"   📁 {r.meeting_id} | DER: {r.der*100:.1f}% (FA: {r.false_alarm*100:.1f}%, Miss: {r.missed_detection*100:.1f}%, Conf: {r.confusion*100:.1f}%) | Süre: {r.duration:.1f}s")
        print("\n" + "=" * 70)
