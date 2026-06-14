from src.core.formatting import format_results


def test_format_results_returns_empty_string_for_no_results():
    assert format_results(None, return_str=True) == ""
    assert format_results([], return_str=True) == ""


def test_format_results_includes_speaker_times_and_text():
    output = format_results(
        [{"speaker": "SPEAKER_00", "start": 1.23, "end": 2.34, "text": "hello"}],
        return_str=True,
    )

    assert "[SPEAKER_00] 1.2s - 2.3s: hello" in output

