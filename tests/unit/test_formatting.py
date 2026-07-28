from src.core.formatting import format_results, parse_result_line


def test_format_results_returns_empty_string_for_no_results():
    assert format_results(None, return_str=True) == ""
    assert format_results([], return_str=True) == ""


def test_format_results_includes_speaker_times_and_text():
    output = format_results(
        [{"speaker": "SPEAKER_00", "start": 1.23, "end": 2.34, "text": "hello"}],
        return_str=True,
    )

    assert "[SPEAKER_00] 1.2s - 2.3s: hello" in output


# --------------------------------------------------------------------------- #
# parse_result_line — format_results'ın tersi
#
# İkisi aynı modülde: arayüzün iki ayrı yeri (günlük paneli + altyazı overlay'i)
# bu satırları ayrıştırıyor; biçim değişirse ayrıştırıcı unutulmasın.
# --------------------------------------------------------------------------- #
def _result(speaker="SPEAKER_00", start=1.0, end=2.5, text="merhaba dünya"):
    return {"speaker": speaker, "start": start, "end": end, "text": text}


def test_round_trip_recovers_every_field():
    line = format_results([_result()], return_str=True)
    assert parse_result_line(line) == {
        "speaker": "SPEAKER_00", "start": 1.0, "end": 2.5, "text": "merhaba dünya"}


def test_round_trip_survives_multi_speaker_output():
    """Bir chunk birden çok konuşmacı satırı üretebilir."""
    text = format_results(
        [_result(speaker="SPEAKER_00"),
         _result(speaker="SPEAKER_01", start=2.5, end=4.0)],
        return_str=True)
    parsed = [parse_result_line(line) for line in text.split("\n")]

    assert [p["speaker"] for p in parsed] == ["SPEAKER_00", "SPEAKER_01"]
    assert parsed[1]["start"] == 2.5


def test_nested_brackets_in_speaker_are_stripped():
    """Warm-up etiketi '[Calibrating... 12s]' iç içe parantezle yazılır."""
    assert parse_result_line(
        "[[Calibrating... 12s]] 0.0s - 3.0s: merhaba")["speaker"] == "Calibrating... 12s"


def test_pseudo_labels_parse_normally():
    parsed = parse_result_line("[Çözümleniyor...] 0.0s - 1.0s: bir şey")
    assert parsed["speaker"] == "Çözümleniyor..."
    assert parsed["text"] == "bir şey"


def test_text_containing_colons_and_brackets_is_preserved():
    """Metnin içindeki ':' ve '[' ayrıştırmayı bozmamalı."""
    parsed = parse_result_line("[SPEAKER_00] 1.0s - 2.0s: saat 14:30'da [not] var")
    assert parsed["text"] == "saat 14:30'da [not] var"


def test_non_matching_lines_return_none():
    for line in ("", "   ", "düz metin", "[SPEAKER_00] eksik zaman: x",
                 "[Canlı] devam eden cümle", None):
        assert parse_result_line(line) is None


def test_whitespace_is_tolerated():
    assert parse_result_line(
        "  [SPEAKER_01] 5.0s - 6.0s: selam  ")["speaker"] == "SPEAKER_01"


def test_empty_text_still_parses():
    parsed = parse_result_line("[SPEAKER_00] 1.0s - 2.0s: ")
    assert parsed is not None and parsed["text"] == ""


def test_integer_seconds_are_accepted():
    """format_results bir ondalık basar ama ayrıştırıcı katı olmamalı."""
    parsed = parse_result_line("[SPEAKER_00] 1s - 2s: selam")
    assert parsed["start"] == 1.0 and parsed["end"] == 2.0

