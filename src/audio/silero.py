"""Silero VAD yükleme — ONNX (hızlı, çevrimdışı) veya torch.hub (eski yol).

NEDEN ONNX
----------
Eski yol `torch.hub.load("snakers4/silero-vad", ...)` idi. Üç sorunu vardı:
  1. ÇALIŞMA ANINDA İNTERNET: ilk açılışta GitHub'dan indirir. Donmuş bir .exe
     içinde bu kırılgan/kırık — paketleme için asıl engel buydu.
  2. YAVAŞ: PyTorch grafiği, ONNX Runtime'a göre CPU'da belirgin daha ağır.
  3. ÖNGÖRÜLEMEZ ÖNBELLEK: model kullanıcının torch hub cache'ine iner.

ONNX yolu modeli `models/silero_vad.onnx` dosyasından yükler, yalnızca numpy +
onnxruntime kullanır (torch GEREKMEZ) ve ağa hiç çıkmaz.

BACKEND SEÇİMİ (SILERO_BACKEND)
-------------------------------
  auto  : ONNX modeli + onnxruntime varsa ONNX, yoksa torch.hub'a düş (varsayılan)
  onnx  : ONNX zorla (bulunamazsa hata)
  torch : eski torch.hub yolu (referans/karşılaştırma)

API SÖZLEŞMESİ
--------------
`load_silero_vad()` -> (model, utils)
  model.__call__(chunk, sr) -> `.item()` desteği olan skaler (olasılık)
  model.reset_states()
  utils[0] = get_speech_timestamps(audio, model, threshold=..., sampling_rate=...)

Bu, torch.hub'ın döndürdüğü sözleşmenin AYNISIDIR; çağıran kod (VADEngine,
extract_speech_only) iki backend'i ayırt etmez.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Silero v5 sabitleri (modelin kendi arayüzü — değiştirilemez)
WINDOW_SAMPLES = 512      # 16 kHz'de bir pencere
CONTEXT_SAMPLES = 64      # önceki pencereden taşınan bağlam
STATE_SHAPE = (2, 1, 128)  # LSTM durumu

_DEFAULT_ONNX_NAME = "silero_vad.onnx"


def _to_numpy_1d(x) -> np.ndarray:
    """torch.Tensor / numpy / list → 1B float32 numpy (torch import ETMEDEN)."""
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


class SileroOnnxVAD:
    """Silero v5 ONNX sarmalayıcı — torch.hub modeliyle aynı çağrı sözleşmesi.

    Resmî OnnxWrapper ile birebir aynı mantık: her çağrıda 64 örneklik bağlam
    girdinin başına eklenir, LSTM durumu çağrılar arasında taşınır.
    """

    def __init__(self, model_path: str, threads: int = 1):
        import onnxruntime  # yalnızca bu backend seçilirse gerekir

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        # CPU sağlayıcısı: VAD çok küçük bir model, GPU'ya taşımak zarar verir
        # (kernel başlatma maliyeti inference'tan uzun sürer) ve GPU'yu
        # Whisper/pyannote için boş bırakmak isteriz.
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.model_path = model_path
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, chunk, sample_rate: int = 16000):
        """Bir pencerelik konuşma olasılığı. `.item()` çağrılabilir skaler döner."""
        arr = _to_numpy_1d(chunk)
        if arr.shape[0] != WINDOW_SAMPLES:
            raise ValueError(
                f"Silero ONNX tam {WINDOW_SAMPLES} örnek bekler, {arr.shape[0]} geldi."
            )
        inp = np.concatenate([self._context, arr.reshape(1, -1)], axis=1)
        out, state = self.session.run(
            None,
            {
                "input": inp,
                "state": self._state,
                "sr": np.array(sample_rate, dtype=np.int64),
            },
        )
        self._state = state
        self._context = inp[:, -CONTEXT_SAMPLES:]
        # np.float32 → `.item()` destekler (torch tensörüyle aynı kullanım).
        return np.float32(out.reshape(-1)[0])


def get_speech_timestamps(
    audio,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    neg_threshold: float | None = None,
    return_seconds: bool = False,
):
    """Konuşma aralıklarını bulur — Silero'nun resmî algoritmasının bire bir portu.

    Resmî sürümden farkları YALNIZCA şunlardır:
      * numpy tabanlı (torch gerekmez)
      * `max_speech_duration_s` desteklenmez (varsayılan zaten sonsuzdur ve bu
        projede hiç kullanılmıyor) — ilgili dallar çıkarıldı.
    Eşikler, histerezis (neg_threshold), min süre filtreleri ve kenar dolgusu
    (speech_pad) resmî davranışla aynıdır.

    Returns: [{"start": örnek, "end": örnek}, ...]
    """
    arr = _to_numpy_1d(audio)
    if sampling_rate not in (8000, 16000):
        raise ValueError("Silero yalnızca 8000/16000 örnekleme hızını destekler.")
    window = WINDOW_SAMPLES if sampling_rate == 16000 else 256

    # Model-bağımsızlık: ONNX sarmalayıcı numpy alır, torch (TorchScript) modeli
    # ise tensör ister ve numpy verilirse tip hatası fırlatır. Hangi biçimin
    # gerektiğine döngüden ÖNCE bir kez karar veriyoruz.
    _needs_tensor = not isinstance(model, SileroOnnxVAD)
    if _needs_tensor:
        import torch as _torch

        def _feed(c):
            return model(_torch.from_numpy(c), sampling_rate)
    else:
        def _feed(c):
            return model(c, sampling_rate)

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    n = arr.shape[0]

    probs = []
    for start in range(0, n, window):
        chunk = arr[start:start + window]
        if chunk.shape[0] < window:
            chunk = np.pad(chunk, (0, window - chunk.shape[0]))
        probs.append(float(_feed(np.ascontiguousarray(chunk)).item()))

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    triggered = False
    speeches: list = []
    current: dict = {}
    temp_end = 0

    for i, prob in enumerate(probs):
        cur = window * i

        if (prob >= threshold) and temp_end:
            temp_end = 0

        if (prob >= threshold) and not triggered:
            triggered = True
            current["start"] = cur
            continue

        if (prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur
            if (cur - temp_end) < min_silence_samples:
                continue
            current["end"] = temp_end
            if (current["end"] - current["start"]) > min_speech_samples:
                speeches.append(current)
            current = {}
            temp_end = 0
            triggered = False

    if current and (n - current["start"]) > min_speech_samples:
        current["end"] = n
        speeches.append(current)

    # Kenar dolgusu — komşu segmentler çakışırsa boşluk ikiye bölünür.
    for i, sp in enumerate(speeches):
        if i == 0:
            sp["start"] = int(max(0, sp["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            gap = speeches[i + 1]["start"] - sp["end"]
            if gap < 2 * speech_pad_samples:
                sp["end"] += int(gap // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - gap // 2))
            else:
                sp["end"] = int(min(n, sp["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
        else:
            sp["end"] = int(min(n, sp["end"] + speech_pad_samples))

    if return_seconds:
        # Resmî sürümdeki kırpma korunur: yuvarlama sesin dışına taşabilir
        # (ör. 0.576 sn -> 0.6), bu yüzden 0 ve ses uzunluğuyla sınırlanır.
        audio_length_seconds = n / sampling_rate
        for sp in speeches:
            sp["start"] = max(round(sp["start"] / sampling_rate, 1), 0)
            sp["end"] = min(round(sp["end"] / sampling_rate, 1), audio_length_seconds)
    return speeches


def find_onnx_model(models_dir: str | os.PathLike | None = None) -> Path | None:
    """ONNX modelini arar: models/ → SILERO_ONNX_PATH → torch hub cache."""
    explicit = os.getenv("SILERO_ONNX_PATH", "").strip()
    if explicit and Path(explicit).is_file():
        return Path(explicit)

    if models_dir:
        candidate = Path(models_dir) / _DEFAULT_ONNX_NAME
        if candidate.is_file():
            return candidate

    # Son çare: torch.hub önbelleği (paketlemede güvenilmez, geliştirmede pratik)
    hub = Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
    cached = hub / "src" / "silero_vad" / "data" / _DEFAULT_ONNX_NAME
    if cached.is_file():
        return cached
    return None


def _load_torch_hub():
    """Eski yol — torch.hub (ağ gerektirebilir)."""
    import torch
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    return model, utils


def load_silero_vad(backend: str | None = None, models_dir: str | os.PathLike | None = None):
    """Silero VAD yükler. Dönüş: (model, utils) — utils[0] = get_speech_timestamps.

    backend: "auto" | "onnx" | "torch" (None ise SILERO_BACKEND env, o da yoksa "auto")
    """
    backend = (backend or os.getenv("SILERO_BACKEND", "auto")).strip().lower()

    if backend in ("auto", "onnx"):
        path = find_onnx_model(models_dir)
        if path is not None:
            try:
                threads = int(os.getenv("SILERO_ONNX_THREADS", "1"))
                model = SileroOnnxVAD(str(path), threads=threads)
                logger.info("Silero VAD backend: ONNX (%s)", path)
                return model, (get_speech_timestamps,)
            except Exception as exc:
                if backend == "onnx":
                    raise
                logger.warning("ONNX Silero yüklenemedi (%s) — torch.hub'a düşülüyor.", exc)
        elif backend == "onnx":
            raise FileNotFoundError(
                f"'{_DEFAULT_ONNX_NAME}' bulunamadı. `python scripts/download_models.py` "
                "çalıştırın ya da SILERO_ONNX_PATH ayarlayın."
            )

    logger.info("Silero VAD backend: torch.hub")
    return _load_torch_hub()
