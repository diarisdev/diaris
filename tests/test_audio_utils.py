import pytest

from src.audio.utils import calculate_chunk_duration_ms


def test_calculate_chunk_duration_ms_for_mono_int16():
    chunk = b"\x00\x00" * 160

    assert calculate_chunk_duration_ms(chunk, rate=16000, channels=1) == 10.0


def test_calculate_chunk_duration_ms_for_stereo_int16():
    chunk = b"\x00\x00" * 960

    assert calculate_chunk_duration_ms(chunk, rate=48000, channels=2) == 10.0


def test_calculate_chunk_duration_rejects_invalid_rate():
    with pytest.raises(ValueError):
        calculate_chunk_duration_ms(b"", rate=0, channels=1)

