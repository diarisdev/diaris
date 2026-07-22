"""Silero backend (ONNX / torch.hub) birim testleri.

ONNX gerektiren testler onnxruntime + model dosyası yoksa atlanır; saf mantık
(get_speech_timestamps durum makinesi) sahte modelle her ortamda koşar.
"""

import numpy as np
import pytest

from src.audio.silero import (
    CONTEXT_SAMPLES,
    STATE_SHAPE,
    WINDOW_SAMPLES,
    find_onnx_model,
    get_speech_timestamps,
)


class FakeModel:
    """Verilen olasılık dizisini sırayla döndüren sahte VAD.

    get_speech_timestamps'in durum makinesini modelden bağımsız test etmeyi
    sağlar (torch/onnx kurulu olmasa bile).
    """

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0
        self.resets = 0

    def reset_states(self):
        self.resets += 1
        self.i = 0

    def __call__(self, chunk, sample_rate=16000):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return np.float32(p)


def _audio(n_windows):
    return np.zeros(n_windows * WINDOW_SAMPLES, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Durum makinesi (modelden bağımsız)
# --------------------------------------------------------------------------- #
def test_resets_model_state_before_scanning():
    model = FakeModel([0.0] * 10)
    get_speech_timestamps(_audio(10), model, sampling_rate=16000)
    assert model.resets == 1


def test_no_speech_returns_empty():
    model = FakeModel([0.0] * 20)
    assert get_speech_timestamps(_audio(20), model, sampling_rate=16000) == []


def test_detects_a_single_speech_region():
    # 20 pencere: 5-15 arası konuşma (10 pencere = 320 ms > 250 ms min süre)
    probs = [0.0] * 5 + [0.9] * 10 + [0.0] * 5
    model = FakeModel(probs)
    out = get_speech_timestamps(_audio(20), model, sampling_rate=16000)
    assert len(out) == 1
    assert out[0]["start"] < out[0]["end"]


def test_short_speech_below_min_duration_is_dropped():
    # 2 pencere ≈ 64 ms — min_speech_duration_ms=250'nin altında
    probs = [0.0] * 5 + [0.9] * 2 + [0.0] * 13
    model = FakeModel(probs)
    assert get_speech_timestamps(_audio(20), model, sampling_rate=16000) == []


def test_brief_dip_does_not_split_one_region():
    """Kısa sessizlik (min_silence_duration_ms=100 altı) bölme yapmaz."""
    probs = [0.0] * 3 + [0.9] * 10 + [0.1] * 1 + [0.9] * 10 + [0.0] * 3
    model = FakeModel(probs)
    out = get_speech_timestamps(_audio(27), model, sampling_rate=16000)
    assert len(out) == 1


def test_long_silence_splits_into_two_regions():
    probs = [0.0] * 3 + [0.9] * 12 + [0.0] * 12 + [0.9] * 12 + [0.0] * 3
    model = FakeModel(probs)
    out = get_speech_timestamps(_audio(42), model, sampling_rate=16000)
    assert len(out) == 2
    assert out[0]["end"] <= out[1]["start"]


def test_hysteresis_uses_neg_threshold():
    """0.5 ile 0.35 arası değerler konuşmayı SONLANDIRMAZ (histerezis)."""
    probs = [0.0] * 3 + [0.9] * 8 + [0.4] * 8 + [0.9] * 8 + [0.0] * 3
    model = FakeModel(probs)
    out = get_speech_timestamps(_audio(30), model, threshold=0.5, sampling_rate=16000)
    assert len(out) == 1


def test_trailing_speech_is_closed_at_end_of_audio():
    probs = [0.0] * 3 + [0.9] * 15
    model = FakeModel(probs)
    out = get_speech_timestamps(_audio(18), model, sampling_rate=16000)
    assert len(out) == 1
    assert out[0]["end"] == 18 * WINDOW_SAMPLES


def test_return_seconds_converts_units():
    probs = [0.0] * 3 + [0.9] * 15
    model = FakeModel(probs)
    out = get_speech_timestamps(_audio(18), model, sampling_rate=16000, return_seconds=True)
    assert out and out[0]["end"] <= 18 * WINDOW_SAMPLES / 16000


def test_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError):
        get_speech_timestamps(_audio(5), FakeModel([0.0] * 5), sampling_rate=44100)


# --------------------------------------------------------------------------- #
# ONNX backend (model + onnxruntime varsa)
# --------------------------------------------------------------------------- #
onnx_available = pytest.mark.skipif(
    find_onnx_model("models") is None or pytest.importorskip is None,
    reason="silero_vad.onnx bulunamadı",
)


@onnx_available
def test_onnx_wrapper_contract():
    pytest.importorskip("onnxruntime")
    from src.audio.silero import SileroOnnxVAD

    model = SileroOnnxVAD(str(find_onnx_model("models")))
    # torch modeliyle aynı sözleşme: .item() ve reset_states()
    out = model(np.zeros(WINDOW_SAMPLES, dtype=np.float32), 16000)
    assert hasattr(out, "item")
    assert 0.0 <= float(out.item()) <= 1.0

    assert model._state.shape == STATE_SHAPE
    assert model._context.shape == (1, CONTEXT_SAMPLES)
    model.reset_states()
    assert not model._state.any()


@onnx_available
def test_onnx_rejects_wrong_window_size():
    pytest.importorskip("onnxruntime")
    from src.audio.silero import SileroOnnxVAD

    model = SileroOnnxVAD(str(find_onnx_model("models")))
    with pytest.raises(ValueError):
        model(np.zeros(256, dtype=np.float32), 16000)


@onnx_available
def test_onnx_carries_state_between_windows():
    """Bağlam (context) pencereler arasında taşınmalı — durum sıfırlanınca
    aynı girdi aynı çıktıyı vermeli."""
    pytest.importorskip("onnxruntime")
    from src.audio.silero import SileroOnnxVAD

    model = SileroOnnxVAD(str(find_onnx_model("models")))
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(WINDOW_SAMPLES) * 0.1).astype(np.float32)

    model.reset_states()
    first = float(model(w, 16000).item())
    second = float(model(w, 16000).item())
    model.reset_states()
    first_again = float(model(w, 16000).item())

    assert first == pytest.approx(first_again, abs=1e-6)
    # Durum taşındığı için ikinci çağrı ilkinden farklı olmalı.
    assert first != pytest.approx(second, abs=1e-9)
