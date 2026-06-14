# tests/ — değerlendirme, benchmark ve birim testler

Bu klasör üç katmana ayrılır. **Veri setleri burada DEĞİL** — proje kökündeki
gitignore'lı `datasets/` altındadır.

```
tests/
├── unit/               # hızlı pytest birim testleri (CI). Ağır bağımlılık gerektirmez.
├── dataset_managers/   # veri seti indir/yükle. Dataset başına 1 dosya. Tek registry.
├── metrics/            # TÜM metrik matematiğinin tek kaynağı (WER/CER/cpWER/DER/BLEU/chrF/COMET).
└── benchmarks/         # ağır runner'lar. Dataset başına 1 dosya; matematiği metrics'ten çağırır.
```

## Yeni bir şey eklerken (karışmasın diye kural)

- **Yeni dataset** → `tests/dataset_managers/<ad>.py` (indir/yükle) + `__init__.py` registry'sine ekle.
  Veri `datasets/<ad>/` altına iner.
- **Yeni metrik** → `tests/metrics/` içine bir modül; `tests/metrics/__init__.py`'den export et.
  Benchmark'lar metriği BURADAN çağırır, yeniden YAZMAZ.
- **Yeni benchmark** → `tests/benchmarks/<ad>.py`. Kendi normalizasyon + metrik-seçimi +
  raporlamasını yapar; matematiği `tests.metrics`'ten alır.
- **Yeni birim test** → `tests/unit/test_*.py`.

## Çalıştırma

### Birim testler (hızlı)
```
python -m pytest tests/unit -q
```

### Veri setlerini indir
```
python -m tests.dataset_managers              # hepsi (datasets/ altına)
python -m tests.dataset_managers flores200    # tek dataset
```

### Benchmark'lar (ağır; model/veri gerektirir)
```
# Çeviri (FLORES-200): Google/DeepL/NLLB -> BLEU/chrF++/COMET
python -m tests.benchmarks.translation --pairs en-tr --limit 100

# LibriSpeech ASR (WER/CER)
python -m tests.benchmarks.librispeech_asr --limit 20

# AMI diarization (DER)
python -m tests.benchmarks.ami_diarization --mode raw

# AMI referanslarını hazırla (replay/score için gerekli; lhotse + ~birkaç GB indirir)
python -m tests.dataset_managers ami_refs

# AMI offline replay -> hipotez üret, sonra skorla
python -m tests.benchmarks.ami_replay --only IS1009a
python -m tests.benchmarks.ami_score

# CHiME-6 (WER/cpWER/DER) — komut örnekleri dosyanın başındaki docstring'de
python -m tests.benchmarks.chime6 --sanity-check
```

## Notlar
- COMET ilk çalıştırmada ~2GB model indirir; hızlı tur için `--no-comet`.
- CHiME-6 verisi lisanslıdır, otomatik inmez; `datasets/chime6/{audio,transcriptions}` altına elle koyun.
- Metrik sayıları tek kaynaktan (`tests/metrics`) gelir; kopyalama/drift yoktur.
