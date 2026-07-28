from tests.dataset_managers import _parse_transcription_file, AmiDiarizationManager


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



# --------------------------------------------------------------------------- #
# Ses yolu çözümleme
#
# Regresyon: meetings.json mutlak yolları indirme anında dondurur. Proje dizini
# yeniden adlandırılınca (Audio-process -> diaris) bu yollar ölüyor ve replay
# HER toplantıyı sessizce atlıyordu.
# --------------------------------------------------------------------------- #
def _ami_tree(tmp_path):
    """datasets/ami/ami_corpus/wav_db/M1/audio/M1.wav iskeleti kurar."""
    audio = tmp_path / "datasets" / "ami" / "ami_corpus" / "wav_db" / "M1" / "audio"
    audio.mkdir(parents=True)
    wav = audio / "M1.Mix-Headset.wav"
    wav.write_bytes(b"RIFF")
    return tmp_path / "datasets", wav


def test_existing_path_is_returned_as_is(tmp_path):
    from tests.dataset_managers.ami import resolve_audio_path
    datasets, wav = _ami_tree(tmp_path)
    assert resolve_audio_path(str(wav), datasets_root=datasets) == wav


def test_stale_absolute_path_is_rebased(tmp_path):
    """Eski proje adını taşıyan mutlak yol mevcut köke bağlanmalı."""
    from tests.dataset_managers.ami import resolve_audio_path
    datasets, wav = _ami_tree(tmp_path)
    stale = r"C:\Users\x\Github\Audio-process\datasets\ami\ami_corpus\wav_db\M1\audio\M1.Mix-Headset.wav"
    assert resolve_audio_path(stale, datasets_root=datasets) == wav


def test_relative_path_is_resolved_under_datasets(tmp_path):
    from tests.dataset_managers.ami import resolve_audio_path
    datasets, wav = _ami_tree(tmp_path)
    relative = "ami_corpus/wav_db/M1/audio/M1.Mix-Headset.wav"
    assert resolve_audio_path(relative, datasets_root=datasets) == wav


def test_filename_search_is_the_last_resort(tmp_path):
    """Dizin yapısı tamamen değişse bile dosya adıyla bulunmalı."""
    from tests.dataset_managers.ami import resolve_audio_path
    datasets, wav = _ami_tree(tmp_path)
    stale = r"D:\bambaska\bir\yer\M1.Mix-Headset.wav"
    assert resolve_audio_path(stale, datasets_root=datasets) == wav


def test_missing_audio_returns_none(tmp_path):
    from tests.dataset_managers.ami import resolve_audio_path
    datasets, _ = _ami_tree(tmp_path)
    assert resolve_audio_path(r"D:\yok\OLMAYAN.wav", datasets_root=datasets) is None
    assert resolve_audio_path("", datasets_root=datasets) is None
    assert resolve_audio_path(None, datasets_root=datasets) is None
