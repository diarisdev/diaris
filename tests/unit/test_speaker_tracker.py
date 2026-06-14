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
