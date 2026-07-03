# Audio-Process

Gerçek zamanlı ses transkripsiyon ve konuşmacı ayrıştırma sistemi.
Windows üzerinde sistem sesini (WASAPI Loopback) yakalayıp Whisper ile transkribe eder, Pyannote ile konuşmacıları ayırt eder.

## Proje Yapısı

```
src/
├── config.py               # Tüm ayarlar (.env'den yüklenir)
├── pipeline.py             # Canlı kayıt döngüsü, chunk yönetimi, AI thread
├── ui/                     # PySide6 arayüzü (ana pencere, overlay, tray)
├── audio/
│   ├── device.py           # WASAPI loopback cihaz algılama
│   ├── vad.py              # İkili VAD: WebRTC (ön filtre) + Silero (doğrulama)
│   └── utils.py            # Ses yardımcı fonksiyonları
└── core/
    ├── ai_worker.py        # Whisper + Pyannote + SpeakerTracker (warm-up, embedding, clustering)
    └── formatting.py       # Çıktı formatlama
```

- `scripts/download_models.py` — Model indirme betiği
- `tests/` — Unit testler + benchmark/evaluator araçları
- `models/` — Yerel AI modelleri (gitignore'd)
- `output/` — Çıktı dosyaları (gitignore'd)

## Kurulum

**Gereksinimler:** Windows 10/11, Python 3.10+, CUDA opsiyonel.

```bash
git clone https://github.com/YusufBalmumcu/Audio-process.git
cd Audio-process
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -c constraints-windows.txt
```

`.env` dosyasını oluştur:

```bash
copy .env.example .env
```

`HF_TOKEN` alanını doldur. Hugging Face'te şu modellerin erişimini kabul etmen gerekiyor:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Modelleri indir:

```bash
python scripts/download_models.py
```

Bu komut `models/` altına şunları indirir:
- `whisper-small` — Faster-Whisper (transkripsiyon)
- `pyannote-segmentation` — Konuşmacı segmentasyonu
- `pyannote-embeddings` — Konuşmacı embedding vektörleri

## Çalıştırma

```bash
python main.py          # GUI (PySide6)
python main.py --cli    # Terminal modu (Ctrl+Q ile durdur)
```

## Yapılandırma (.env)

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `HF_TOKEN` | — | Hugging Face token (**zorunlu**) |
| `WHISPER_LANGUAGE` | `en` | Transkripsiyon dili |
| `WHISPER_MODEL` | — | Whisper model dizini (boş: GPU'da `whisper-medium`, yoksa `whisper-small`) |
| `VAD_AGGRESSIVENESS` | `2` | WebRTC agresiflik (0–3) |
| `SILERO_THRESHOLD` | `0.70` | Silero güven eşiği |
| `SILENCE_LIMIT` | `30` | Sessizlik limiti (frame) |
| `SHORT_SILENCE_LIMIT` | `15` | Uzun chunk'larda kısa sessizlik limiti |
| `SOFT_CHUNK_DURATION_MS` | `5000` | Bu süre sonrası SHORT_SILENCE_LIMIT devreye girer |
| `MAX_CHUNK_DURATION_MS` | `10000` | Maks chunk süresi |
| `PRE_ROLL_MS` | `240` | Konuşma öncesi chunk'a eklenen ham ses (onset koruması) |
| `CHUNK_OVERLAP_MS` | `480` | Max-süre kesmesinde sonraki chunk'a taşınan overlap |
| `PARTIAL_WINDOW_MS` | `4000` | Canlı satır için decode edilen buffer kuyruğu (final tam buffer görür) |
| `DIARIZATION_EMBEDDING_THRESHOLD` | `0.66` | Konuşmacı eşleme cosine similarity eşiği |
| `DIARIZATION_WARMUP_MS` | `20000` | Warm-up süresi (embedding toplama) |
| `CANDIDATE_CONFIRMATIONS_NEEDED` | `2` | Yeni konuşmacı için gereken gözlem sayısı |
| `CANDIDATE_TTL` | `15` | Aday konuşmacı bekleme süresi (chunk) |
| `MIN_NEW_SPEAKER_DURATION` | `2.0` | Yeni konuşmacı için min temiz konuşma (sn) |
| `SAVE_AUDIO_FILE` | `false` | Tüm sesi WAV olarak kaydet |
| `OUTPUT_FILENAME` | `system_recorded.wav` | Kayıt dosya adı |

## Test ve Lint

```bash
pip install -r requirements-dev.txt
pytest
pytest -m "not slow and not requires_model"   # sadece hızlı testler
ruff check .
```

## Lisans

MIT
