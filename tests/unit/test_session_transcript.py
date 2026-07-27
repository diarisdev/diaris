"""SessionTranscript birim testleri.

Canlı pipeline'ın oturum-sonu konuşmacı düzeltmesini besleyen biriktirme
katmanı. Saf Python (torch/model yok) — testler CI'da tam hızda koşar.
"""

from src.core.session_transcript import SessionTranscript


def seg(speaker, start, end, text="word"):
    return {"speaker": speaker, "start": float(start), "end": float(end), "text": text}


# Hayalet konuşmacı senaryosu: SPEAKER_00 uzun uzun konuşur, araya iki kez kısa
# bir SPEAKER_01 "adası" sızar. Refinement (tiny_fragmented) komşu oylamasıyla
# bu adaları asıl konuşmacıya devreder.
def _ghost_speaker_session():
    transcript = SessionTranscript()
    transcript.add_chunk(0, [seg("SPEAKER_00", 0, 20, "uzun konuşma bir")])
    transcript.add_chunk(1, [seg("SPEAKER_01", 0, 2, "kısa ada bir")])
    transcript.add_chunk(2, [seg("SPEAKER_00", 0, 20, "uzun konuşma iki")])
    transcript.add_chunk(3, [seg("SPEAKER_01", 0, 2, "kısa ada iki")])
    transcript.add_chunk(4, [seg("SPEAKER_00", 0, 20, "uzun konuşma üç")])
    return transcript


def test_empty_session_returns_no_updates():
    updates, stats = SessionTranscript().refine()
    assert updates == {}
    assert stats["total"] == 0


def test_stable_session_is_a_no_op():
    """İki konuşmacı da olgun ve dengeliyse hiçbir şeye dokunulmaz."""
    transcript = SessionTranscript()
    transcript.add_chunk(0, [seg("SPEAKER_00", 0, 40)])
    transcript.add_chunk(1, [seg("SPEAKER_01", 0, 40)])
    transcript.add_chunk(2, [seg("SPEAKER_00", 0, 40)])

    updates, stats = transcript.refine()
    assert updates == {}
    assert stats["total"] == 0


def test_ghost_speaker_islands_are_relabelled():
    updates, stats = _ghost_speaker_session().refine()

    assert stats["total"] == 2
    # YALNIZCA etiketi değişen chunk'lar döner — dokunulmayanlar listede olmamalı.
    assert set(updates) == {1, 3}
    for text in updates.values():
        assert "SPEAKER_00" in text
        assert "SPEAKER_01" not in text


def test_updates_keep_the_original_text_and_timestamps():
    updates, _ = _ghost_speaker_session().refine()
    assert updates[1] == "[SPEAKER_00] 0.0s - 2.0s: kısa ada bir"


def test_multi_segment_chunk_is_reformatted_as_a_whole():
    """Bir chunk'ın tek segmenti değişse bile satır bütün olarak yeniden yazılır."""
    transcript = SessionTranscript()
    transcript.add_chunk(0, [seg("SPEAKER_00", 0, 20, "birinci")])
    transcript.add_chunk(1, [
        seg("SPEAKER_01", 0, 2, "hayalet"),
        seg("SPEAKER_00", 2, 12, "asıl konuşmacı"),
    ])
    transcript.add_chunk(2, [seg("SPEAKER_01", 0, 2, "hayalet iki")])
    transcript.add_chunk(3, [seg("SPEAKER_00", 0, 20, "ikinci")])

    updates, stats = transcript.refine()

    assert stats["total"] == 2
    assert set(updates) == {1, 2}
    # Değişmeyen segment de satırda korunur.
    assert "asıl konuşmacı" in updates[1]
    assert updates[1].count("SPEAKER_00") == 2


def test_add_chunk_copies_segments():
    """Çağıran listeyi sonradan değiştirirse oturum geçmişi bozulmamalı."""
    transcript = SessionTranscript()
    live_segments = [seg("SPEAKER_00", 0, 20, "orijinal")]
    transcript.add_chunk(0, live_segments)
    live_segments[0]["speaker"] = "BOZULDU"
    live_segments[0]["text"] = "üzerine yazıldı"

    transcript.add_chunk(1, [seg("SPEAKER_01", 0, 2)])
    transcript.add_chunk(2, [seg("SPEAKER_00", 0, 20)])
    transcript.add_chunk(3, [seg("SPEAKER_01", 0, 2)])
    transcript.add_chunk(4, [seg("SPEAKER_00", 0, 20)])

    updates, _ = transcript.refine()
    assert set(updates) == {1, 3}
    assert "BOZULDU" not in "".join(updates.values())


def test_empty_segment_lists_are_ignored():
    transcript = SessionTranscript()
    transcript.add_chunk(0, [])
    transcript.add_chunk(1, None)
    updates, stats = transcript.refine()
    assert updates == {}
    assert stats["total"] == 0


def test_reset_clears_history():
    transcript = _ghost_speaker_session()
    transcript.reset()
    updates, stats = transcript.refine()
    assert updates == {}
    assert stats["total"] == 0
