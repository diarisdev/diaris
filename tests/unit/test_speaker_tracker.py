import pytest


@pytest.mark.requires_model
def test_speaker_tracker_maps_similar_embedding_to_known_speaker():
    torch = pytest.importorskip("torch")
    pytest.importorskip("faster_whisper")
    pytest.importorskip("pyannote.audio")
    pytest.importorskip("torchaudio")

    from src.core.ai_worker import SpeakerTracker

    tracker = SpeakerTracker(threshold=0.75, warmup_ms=0)
    tracker.known_speakers["SPEAKER_00"] = torch.tensor([1.0, 0.0])
    tracker._next_id = 1
    tracker._warmup_complete = True

    mapping = tracker.map_speakers({"local": torch.tensor([0.99, 0.01])})

    assert mapping == {"local": "SPEAKER_00"}


def test_speaker_tracker_warmup_normalization():
    torch = pytest.importorskip("torch")
    from src.core.speaker_tracker import SpeakerTracker

    tracker = SpeakerTracker(threshold=0.70, warmup_ms=1000)
    emb1 = torch.tensor([2.0, 0.0])
    emb2 = torch.tensor([2.0, 0.0])
    tracker._warmup_buffer = [emb1, emb2]
    tracker._warmup_audio_ms = 1000

    tracker._finalize_warmup()

    assert "SPEAKER_00" in tracker.known_speakers
    centroid = tracker.known_speakers["SPEAKER_00"]
    assert torch.isclose(torch.norm(centroid), torch.tensor(1.0))

