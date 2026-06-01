from tests.dataset_manager import _parse_transcription_file, AmiDiarizationManager


def test_parse_transcription_file(tmp_path):
    transcript_file = tmp_path / "sample.trans.txt"
    transcript_file.write_text("utt-1 HELLO WORLD\nutt-2 SECOND LINE\n", encoding="utf-8")

    assert _parse_transcription_file(transcript_file) == {
        "utt-1": "HELLO WORLD",
        "utt-2": "SECOND LINE",
    }


def test_parse_rttm(tmp_path):
    rttm_file = tmp_path / "meeting.rttm"
    rttm_file.write_text(
        "SPEAKER meeting 1 1.50 2.25 <NA> <NA> speaker_a <NA> <NA>\n",
        encoding="utf-8",
    )

    manager = AmiDiarizationManager(data_dir=str(tmp_path))

    assert manager._parse_rttm(rttm_file) == [
        {"start": 1.5, "end": 3.75, "speaker": "speaker_a"}
    ]

