"""embedding_trace birim testleri — iz kaydı gidiş-dönüş bozulmadan çalışmalı.

Saf numpy (torch/Qt yok).
"""

import numpy as np
import pytest

from src.core.embedding_trace import EmbeddingTraceWriter, load_trace

DIM = 8


def vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def make_trace(chunk_index: int = 1) -> dict:
    return {
        "chunk_index": chunk_index,
        "probes": [{
            "local_label": "SPEAKER_00",
            "assigned": "SPEAKER_01",
            "embedding": vec(chunk_index),
            "duration": 3.5,
            "quality": 0.87,
            "maturity": 0.72,
            "effective_threshold": 0.681,
            "scores": {"SPEAKER_00": 0.51, "SPEAKER_01": 0.79},
            "best": "SPEAKER_01",
            "best_score": 0.79,
            "margin": 0.28,
            "has_margin": True,
            "passed_threshold": True,
            "reliable_duration": True,
            "decision": "matched",
            "reservoir_updated": True,
        }],
        "speakers": {
            "SPEAKER_01": {
                "centroid": vec(100),
                "reservoir": [vec(101), vec(102)],
                "speech_seconds": 12.5,
            },
        },
        "merged": {},
        "candidate_count": 0,
        "merge_threshold": 0.85,
    }


def test_round_trip_preserves_vectors_exactly(tmp_path):
    writer = EmbeddingTraceWriter(source="test:meeting")
    trace = make_trace()
    writer.add(trace, warming_up=False, time_sec=12.0)
    path = writer.save(tmp_path / "trace.npz")

    loaded = load_trace(path)

    assert loaded["source"] == "test:meeting"
    assert loaded["dim"] == DIM
    frame, = loaded["frames"]
    assert frame["warming_up"] is False
    assert frame["time"] == 12.0

    probe = frame["trace"]["probes"][0]
    assert np.allclose(probe["embedding"], trace["probes"][0]["embedding"])
    assert probe["embedding"].shape == (DIM,)
    speaker = frame["trace"]["speakers"]["SPEAKER_01"]
    assert np.allclose(speaker["centroid"], trace["speakers"]["SPEAKER_01"]["centroid"])
    assert len(speaker["reservoir"]) == 2
    assert np.allclose(speaker["reservoir"][1], trace["speakers"]["SPEAKER_01"]["reservoir"][1])


def test_scalar_fields_survive_round_trip(tmp_path):
    writer = EmbeddingTraceWriter()
    writer.add(make_trace(), time_sec=1.0)
    loaded = load_trace(writer.save(tmp_path / "t.npz"))

    probe = loaded["frames"][0]["trace"]["probes"][0]
    assert probe["decision"] == "matched"
    assert probe["scores"] == {"SPEAKER_00": 0.51, "SPEAKER_01": 0.79}
    assert probe["has_margin"] is True
    assert probe["margin"] == pytest.approx(0.28)
    assert loaded["frames"][0]["trace"]["chunk_index"] == 1


def test_none_margin_survives(tmp_path):
    """Tek konuşmacıda margin None gider — round-trip bunu bozmamalı."""
    trace = make_trace()
    trace["probes"][0]["margin"] = None
    writer = EmbeddingTraceWriter()
    writer.add(trace)
    loaded = load_trace(writer.save(tmp_path / "t.npz"))
    assert loaded["frames"][0]["trace"]["probes"][0]["margin"] is None


def test_warmup_matrix_is_stored_once_as_a_matrix(tmp_path):
    """n×n matris havuza girmemeli (vektör değil) ve şekli korunmalı."""
    writer = EmbeddingTraceWriter()
    writer.add(make_trace())
    writer.set_warmup({
        "embedding_count": 3,
        "audio_ms": 20000,
        "similarity_matrix": np.eye(3, dtype=np.float32),
        "clusters": {"SPEAKER_00": [0, 1], "SPEAKER_01": [2]},
        "threshold": 0.66,
        "filtered": 0,
    })
    loaded = load_trace(writer.save(tmp_path / "t.npz"))

    warmup = loaded["warmup"]
    assert warmup["embedding_count"] == 3
    assert np.allclose(np.array(warmup["similarity_matrix"]), np.eye(3))
    assert warmup["clusters"]["SPEAKER_00"] == [0, 1]


def test_multiple_frames_keep_their_order(tmp_path):
    writer = EmbeddingTraceWriter()
    for index in range(5):
        writer.add(make_trace(chunk_index=index), time_sec=float(index))
    assert len(writer) == 5

    loaded = load_trace(writer.save(tmp_path / "t.npz"))
    assert [f["trace"]["chunk_index"] for f in loaded["frames"]] == [0, 1, 2, 3, 4]
    assert [f["time"] for f in loaded["frames"]] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_empty_traces_are_skipped(tmp_path):
    writer = EmbeddingTraceWriter()
    writer.add(None)
    writer.add({})
    assert len(writer) == 0

    loaded = load_trace(writer.save(tmp_path / "t.npz"))
    assert loaded["frames"] == []
    assert loaded["warmup"] is None


def test_vectors_are_pooled_not_duplicated(tmp_path):
    """Aynı vektör tekrar tekrar yazılsa da dosya kare sayısıyla lineer büyümeli."""
    writer = EmbeddingTraceWriter()
    for index in range(20):
        writer.add(make_trace(chunk_index=index))
    path = writer.save(tmp_path / "t.npz")

    with np.load(path, allow_pickle=False) as data:
        # Kare başına 1 probe + 1 centroid + 2 rezervuar = 4 vektör.
        assert data["vectors"].shape == (80, DIM)
        assert data["vectors"].dtype == np.float32


def test_unknown_version_is_rejected(tmp_path):
    import json
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        vectors=np.zeros((0, DIM), dtype=np.float32),
        meta=np.array(json.dumps({"version": 999, "frames": []})),
    )
    with pytest.raises(ValueError, match="sürümü"):
        load_trace(path)
