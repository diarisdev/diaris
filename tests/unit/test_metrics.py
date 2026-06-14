"""Ortak metrik çekirdeği testleri (eski test_evaluator + test_chime6_benchmark birleşik).

WER/normalize, cpWER (perfect / permütasyon+hata / eşleşmeyen konuşmacı) ve
CHiME-6 zaman-damgası ayrıştırma. Hepsi ağır bağımlılık GEREKTİRMEZ.
"""

import pytest

from tests.metrics import (
    TranscriptionEvaluator,
    normalize_text,
    compute_wer,
    cpwer_from_segments,
)
from tests.dataset_managers import parse_time_to_seconds


# ---------------------------------------------------------------- WER / normalize
def test_normalize_text_lowercases_and_removes_punctuation():
    assert normalize_text(" Hello,   WORLD! ") == "hello world"


def test_compute_wer_substitution():
    r = compute_wer("hello world", "hello word")
    assert r.substitutions == 1
    assert r.insertions == 0
    assert r.deletions == 0
    assert r.wer == 0.5


def test_transcription_evaluator_records_sample():
    ev = TranscriptionEvaluator()
    res = ev.evaluate("hello world", "hello word")
    assert res.substitutions == 1
    assert res.wer == 0.5
    assert ev.report.successful_samples == 1


# ---------------------------------------------------------------- cpWER
def test_cpwer_perfect_matching():
    ref = [{"speaker": "P01", "start": 0.0, "end": 2.0, "text": "hello how are you"},
           {"speaker": "P02", "start": 1.0, "end": 3.0, "text": "i am fine thank you"}]
    hyp = [{"speaker": "SPEAKER_00", "start": 0.1, "end": 2.1, "text": "hello how are you"},
           {"speaker": "SPEAKER_01", "start": 1.1, "end": 3.1, "text": "i am fine thank you"}]
    r = cpwer_from_segments(ref, hyp)
    assert r.cpwer == 0.0
    assert r.mapping["P01"] == "SPEAKER_00"
    assert r.mapping["P02"] == "SPEAKER_01"
    assert r.details["P01"]["sub"] == 0


def test_cpwer_with_errors_and_permutation():
    ref = [{"speaker": "P01", "start": 0.0, "end": 2.0, "text": "hello world"},
           {"speaker": "P02", "start": 1.0, "end": 3.0, "text": "good morning sunshine"}]
    # İsimler permüte: SPEAKER_00 -> P02, SPEAKER_01 -> P01 (1 substitution)
    hyp = [{"speaker": "SPEAKER_00", "start": 1.1, "end": 3.1, "text": "good morning sunshine"},
           {"speaker": "SPEAKER_01", "start": 0.1, "end": 2.1, "text": "hello word"}]
    r = cpwer_from_segments(ref, hyp)
    assert r.cpwer == pytest.approx(0.2)   # 1 hata / 5 ref kelime
    assert r.mapping["P01"] == "SPEAKER_01"
    assert r.mapping["P02"] == "SPEAKER_00"
    assert r.details["P01"]["sub"] == 1
    assert r.details["P02"]["sub"] == 0


def test_cpwer_mismatched_speaker_counts():
    ref = [{"speaker": "P01", "start": 0.0, "end": 2.0, "text": "hello world"}]
    hyp = [{"speaker": "SPEAKER_00", "start": 0.1, "end": 2.1, "text": "hello world"},
           {"speaker": "SPEAKER_01", "start": 3.0, "end": 4.0, "text": "extra noise"}]
    r = cpwer_from_segments(ref, hyp)
    # P01 <-> SPEAKER_00 mükemmel; SPEAKER_01 -> [None], 2 insertion / 2 ref = 1.0
    assert r.cpwer == pytest.approx(1.0)
    assert r.mapping["P01"] == "SPEAKER_00"
    assert r.details["[None]"]["ins"] == 2


# ---------------------------------------------------------------- CHiME-6 zaman
def test_parse_time_to_seconds():
    assert parse_time_to_seconds(12.34) == 12.34
    assert parse_time_to_seconds(5) == 5.0
    assert parse_time_to_seconds("123.45") == 123.45
    assert parse_time_to_seconds("02:15.50") == 135.50
    assert parse_time_to_seconds("00:45") == 45.0
    assert parse_time_to_seconds("01:02:03.500") == 3723.5
    assert parse_time_to_seconds("00:00:10") == 10.0
    assert parse_time_to_seconds({"original": "0:01:00"}) == 60.0
