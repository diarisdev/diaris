import pytest
from tests.evaluator import TranscriptionEvaluator
from tests.chime6_benchmark import parse_time_to_seconds, compute_cpwer


def test_parse_time_to_seconds():
    # Direct float/int
    assert parse_time_to_seconds(12.34) == 12.34
    assert parse_time_to_seconds(5) == 5.0
    
    # String float
    assert parse_time_to_seconds("123.45") == 123.45
    
    # MM:SS.mmm
    assert parse_time_to_seconds("02:15.50") == 135.50
    assert parse_time_to_seconds("00:45") == 45.0
    
    # HH:MM:SS.mmm
    assert parse_time_to_seconds("01:02:03.500") == 3723.5
    assert parse_time_to_seconds("00:00:10") == 10.0


def test_compute_cpwer_perfect_matching():
    evaluator = TranscriptionEvaluator()
    
    # Perfect alignment
    ref_segs = [
        {"speaker": "P01", "start": 0.0, "end": 2.0, "text": "hello how are you"},
        {"speaker": "P02", "start": 1.0, "end": 3.0, "text": "i am fine thank you"}
    ]
    
    hyp_segs = [
        {"speaker": "SPEAKER_00", "start": 0.1, "end": 2.1, "text": "hello how are you"},
        {"speaker": "SPEAKER_01", "start": 1.1, "end": 3.1, "text": "i am fine thank you"}
    ]
    
    cpwer, mapping, details = compute_cpwer(ref_segs, hyp_segs, evaluator)
    
    assert cpwer == 0.0
    assert mapping["P01"] == "SPEAKER_00"
    assert mapping["P02"] == "SPEAKER_01"
    assert details["P01"]["sub"] == 0
    assert details["P01"]["ins"] == 0
    assert details["P01"]["del"] == 0


def test_compute_cpwer_with_errors_and_permutation():
    evaluator = TranscriptionEvaluator()
    
    ref_segs = [
        {"speaker": "P01", "start": 0.0, "end": 2.0, "text": "hello world"},
        {"speaker": "P02", "start": 1.0, "end": 3.0, "text": "good morning sunshine"}
    ]
    
    # Permuted names: SPEAKER_00 matches P02, SPEAKER_01 matches P01 (with 1 substitution)
    hyp_segs = [
        {"speaker": "SPEAKER_00", "start": 1.1, "end": 3.1, "text": "good morning sunshine"},
        {"speaker": "SPEAKER_01", "start": 0.1, "end": 2.1, "text": "hello word"}  # "world" -> "word" (1 substitution)
    ]
    
    cpwer, mapping, details = compute_cpwer(ref_segs, hyp_segs, evaluator)
    
    # Total reference words = 2 (hello world) + 3 (good morning sunshine) = 5
    # Total errors = 1 substitution ("world" vs "word")
    # cpWER should be 1 / 5 = 0.2
    assert cpwer == pytest.approx(0.2)
    assert mapping["P01"] == "SPEAKER_01"
    assert mapping["P02"] == "SPEAKER_00"
    assert details["P01"]["sub"] == 1
    assert details["P01"]["ins"] == 0
    assert details["P01"]["del"] == 0
    assert details["P02"]["sub"] == 0
    assert details["P02"]["ins"] == 0
    assert details["P02"]["del"] == 0


def test_compute_cpwer_mismatched_speaker_counts():
    evaluator = TranscriptionEvaluator()
    
    ref_segs = [
        {"speaker": "P01", "start": 0.0, "end": 2.0, "text": "hello world"}
    ]
    
    # 2 hyp speakers, 1 ref speaker
    hyp_segs = [
        {"speaker": "SPEAKER_00", "start": 0.1, "end": 2.1, "text": "hello world"},
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 4.0, "text": "extra noise"}
    ]
    
    cpwer, mapping, details = compute_cpwer(ref_segs, hyp_segs, evaluator)
    
    # P01 matches SPEAKER_00 perfectly (0 errors).
    # SPEAKER_01 matches [None] ref speaker -> all words are insertions.
    # Total ref words = 2 (hello world)
    # Total errors = 2 insertions ("extra noise")
    # cpWER = 2 / 2 = 1.0
    assert cpwer == pytest.approx(1.0)
    assert mapping["P01"] == "SPEAKER_00"
    assert details["P01"]["sub"] == 0
    
    # Verify the virtual mapping details
    virtual_ref = None
    for r_spk, h_spk in mapping.items():
        if h_spk == "SPEAKER_01":
            virtual_ref = r_spk
    
    assert virtual_ref == "[None]"
    assert details["[None]"]["ins"] == 2
