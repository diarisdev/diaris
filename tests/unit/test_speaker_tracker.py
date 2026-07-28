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


# --------------------------------------------------------------------------- #
# Warm-up kapıları
#
# Regresyon: chunk süresi eskiden embedding BAŞINA sayılıyordu, yani N
# konuşmacılı chunk süreyi N kat ilerletiyordu ve kalibrasyon kalabalık
# toplantıda en erken (en az veriyle) bitiyordu.
# --------------------------------------------------------------------------- #
def _tracker(warmup_ms):
    from src.core.speaker_tracker import SpeakerTracker
    return SpeakerTracker(threshold=0.70, warmup_ms=warmup_ms)


def test_warmup_counts_chunk_duration_once_regardless_of_speaker_count():
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=20000)

    voice = torch.tensor([1.0, 0.0])
    # Tek chunk, 3 konuşmacı, 5 saniye.
    tracker.add_warmup_chunk([voice, voice, voice], 5000)

    # 15000 DEĞİL: süre chunk başına bir kez sayılır.
    assert tracker._warmup_audio_ms == 5000
    assert tracker.is_warming_up
    assert tracker.warmup_remaining_ms == 15000


def test_warmup_does_not_complete_before_the_audio_gate():
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=20000)

    voice = torch.tensor([1.0, 0.0])
    for _ in range(3):
        assert tracker.add_warmup_chunk([voice, voice], 5000) is False

    assert tracker.is_warming_up
    assert tracker.known_speakers == {}


def test_warmup_waits_for_enough_embeddings_after_the_audio_gate():
    """Süre dolsa bile az örnekle kümeleme yapılmaz."""
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=10000)

    voice = torch.tensor([1.0, 0.0])
    tracker.add_warmup_chunk([voice], 4000)
    tracker.add_warmup_chunk([voice], 4000)
    tracker.add_warmup_chunk([voice], 4000)  # 12000 >= 10000 ama 3 < 6 embedding

    assert tracker.is_warming_up
    assert tracker.warmup_remaining_ms == 0  # süre kapısı açık, embedding kapısı kapalı


def test_warmup_audio_ceiling_releases_the_embedding_gate():
    """Embedding kapısı kalibrasyonu sonsuza kadar erteleyemez."""
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=10000)

    voice = torch.tensor([1.0, 0.0])
    tracker.add_warmup_chunk([voice], 9000)
    assert tracker.is_warming_up
    # 20000 ms == warmup_ms * WARMUP_MAX_AUDIO_FACTOR → 2 embedding'le bitir.
    assert tracker.add_warmup_chunk([voice], 11000) is True
    assert not tracker.is_warming_up
    assert tracker.known_speakers


def test_warmup_completes_when_both_gates_pass():
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=5000)

    voice = torch.tensor([1.0, 0.0])
    assert tracker.add_warmup_chunk([voice] * 6, 5000) is True
    assert not tracker.is_warming_up
    assert len(tracker.known_speakers) == 1


def test_warmup_keeps_every_embedding_of_the_completing_chunk():
    """Bitiren chunk'ın embedding'leri kümelemeye TAM girer (erken çıkış yok)."""
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=1000)

    voice = torch.tensor([1.0, 0.0])
    tracker.add_warmup_chunk([voice] * 7, 2000)

    assert not tracker.is_warming_up
    label, = tracker.known_speakers
    assert len(tracker._reservoirs[label]) == 7


def test_warmup_trace_is_recorded_even_without_debug():
    """Kalibrasyonun kaç embedding'le bittiği her zaman kayda geçmeli."""
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=1000)

    voice = torch.tensor([1.0, 0.0])
    tracker.add_warmup_chunk([voice] * 6, 2000)

    trace = tracker.last_warmup_trace
    assert trace is not None
    assert trace["embedding_count"] == 6
    assert trace["audio_ms"] == 2000
    assert trace["similarity_matrix"].shape == (6, 6)
    assert sum(len(idx) for idx in trace["clusters"].values()) == 6


# --------------------------------------------------------------------------- #
# Karar izi (embedding görünümü)
# --------------------------------------------------------------------------- #
def _active_tracker(torch):
    from src.core.speaker_tracker import SpeakerTracker
    tracker = SpeakerTracker(threshold=0.75, warmup_ms=0)
    tracker._warmup_complete = True
    tracker._register_speaker([torch.tensor([1.0, 0.0])])
    tracker._register_speaker([torch.tensor([0.0, 1.0])])
    return tracker


def test_no_trace_is_built_while_debug_is_disabled():
    torch = pytest.importorskip("torch")
    tracker = _active_tracker(torch)

    tracker.map_speakers({"local": torch.tensor([0.99, 0.01])})

    assert tracker.last_trace is None


def test_trace_records_scores_threshold_and_decision():
    torch = pytest.importorskip("torch")
    tracker = _active_tracker(torch)
    tracker.debug_enabled = True

    tracker.map_speakers({"local": torch.tensor([0.99, 0.01])}, {"local": 5.0})

    trace = tracker.last_trace
    assert trace is not None
    probe, = trace["probes"]
    assert probe["local_label"] == "local"
    assert probe["assigned"] == "SPEAKER_00"
    assert probe["decision"] == "matched"
    assert probe["passed_threshold"] is True
    # Her bilinen konuşmacı skorlanmış olmalı.
    assert set(probe["scores"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert probe["scores"]["SPEAKER_00"] > probe["scores"]["SPEAKER_01"]
    assert probe["embedding"].shape == (2,)
    # Profil görüntüsü de gelmeli (grafik bunu çiziyor).
    assert set(trace["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert trace["speakers"]["SPEAKER_00"]["centroid"].shape == (2,)


def test_trace_marks_short_utterances_as_sticky():
    """Kısa ses yeni konuşmacı yaratmaz; iz bunu ayırt edebilmeli."""
    torch = pytest.importorskip("torch")
    tracker = _active_tracker(torch)
    tracker.debug_enabled = True

    # Hiçbir profile benzemeyen ama ÇOK KISA ses → sticky.
    tracker.map_speakers({"local": torch.tensor([0.7, 0.7])}, {"local": 0.4})

    probe, = tracker.last_trace["probes"]
    assert probe["decision"] == "sticky_short"
    assert probe["reliable_duration"] is False
    assert probe["reservoir_updated"] is False


def test_trace_reports_infinite_margin_as_none():
    """Tek konuşmacıda rekabet yok — margin sonsuz, arayüze None gitmeli."""
    torch = pytest.importorskip("torch")
    from src.core.speaker_tracker import SpeakerTracker

    tracker = SpeakerTracker(threshold=0.75, warmup_ms=0)
    tracker._warmup_complete = True
    tracker._register_speaker([torch.tensor([1.0, 0.0])])
    tracker.debug_enabled = True

    tracker.map_speakers({"local": torch.tensor([0.99, 0.01])}, {"local": 5.0})

    probe, = tracker.last_trace["probes"]
    assert probe["margin"] is None
    assert probe["has_margin"] is True


def test_reset_clears_traces():
    torch = pytest.importorskip("torch")
    tracker = _active_tracker(torch)
    tracker.debug_enabled = True
    tracker.map_speakers({"local": torch.tensor([0.99, 0.01])}, {"local": 5.0})

    tracker.reset()

    assert tracker.last_trace is None
    assert tracker.last_warmup_trace is None


def test_warmup_ignores_chunks_without_embeddings():
    """Embedding üretmeyen chunk kalibrasyon süresini ilerletmez."""
    pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=10000)

    assert tracker.add_warmup_chunk([], 8000) is False
    assert tracker.add_warmup_chunk([None], 8000) is False
    assert tracker._warmup_audio_ms == 0
    assert tracker.is_warming_up

