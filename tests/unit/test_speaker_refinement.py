"""speaker_refinement birim testleri.

Bu modül saf Python (torch/model yok) — testler CI'da tam hızda koşar.
"""

from src.core.speaker_refinement import (
    RefinementConfig,
    build_islands,
    fill_unknown_islands,
    is_pseudo_label,
    label_stats,
    merge_small_islands,
    merge_tiny_fragmented,
    refine_speakers,
    speaker_summary,
)


def seg(speaker, start, end, text="word"):
    return {"speaker": speaker, "start": float(start), "end": float(end), "text": text}


# Kural mantığını test ederken eşikleri AÇIKÇA veriyoruz: varsayılanlar
# ölçümle değişebilir, kuralın davranışı değişmemeli.
RULE_CFG = RefinementConfig(
    tiny_fragmented_max_duration=6.0,
    tiny_fragmented_max_islands=3,
)


# --------------------------------------------------------------------------- #
# Temel yapılar
# --------------------------------------------------------------------------- #
def test_pseudo_labels_are_detected():
    assert is_pseudo_label("Unknown")
    assert is_pseudo_label("CALIBRATING")
    assert is_pseudo_label("[Calibrating... 5s]")
    assert is_pseudo_label("")
    assert not is_pseudo_label("SPEAKER_01")


def test_islands_group_consecutive_same_speaker():
    segs = [seg("A", 0, 2), seg("A", 2, 4), seg("B", 4, 6), seg("A", 6, 8)]
    islands = build_islands(segs)
    assert [i.label for i in islands] == ["A", "B", "A"]
    assert islands[0].duration == 4.0

    stats = label_stats(islands)
    assert stats["A"].islands == 2
    assert stats["A"].segments == 3
    assert stats["A"].duration == 6.0


# --------------------------------------------------------------------------- #
# Kural 1 — tek seferlik küçük ada
# --------------------------------------------------------------------------- #
def test_small_island_merges_between_identical_neighbours():
    segs = [seg("A", 0, 5), seg("A", 5, 10), seg("X", 10, 11),
            seg("A", 11, 16), seg("A", 16, 21)]
    assert merge_small_islands(segs, RULE_CFG) == 1
    assert segs[2]["speaker"] == "A"
    assert segs[2]["refined_from"] == "X"
    assert segs[2]["refined_by"] == "small_island_merge"


def test_small_island_kept_when_neighbours_differ():
    segs = [seg("A", 0, 5), seg("X", 5, 6), seg("B", 6, 11)]
    assert merge_small_islands(segs, RULE_CFG) == 0
    assert segs[1]["speaker"] == "X"


def test_small_island_declines_when_label_recurs():
    """İki adada görünen etiket kural 1'in değil, kural 2'nin işi."""
    segs = [seg("A", 0, 5), seg("X", 5, 6), seg("A", 6, 11),
            seg("X", 11, 12), seg("A", 12, 17)]
    assert merge_small_islands(segs, RULE_CFG) == 0


def test_small_island_kept_when_too_long():
    segs = [seg("A", 0, 5), seg("X", 5, 20), seg("A", 20, 25)]
    assert merge_small_islands(segs, RULE_CFG) == 0


# --------------------------------------------------------------------------- #
# Kural 2 — dağınık küçük profil
# --------------------------------------------------------------------------- #
def test_tiny_fragmented_merges_into_voted_neighbour():
    segs = [seg("A", 0, 10), seg("X", 10, 11), seg("A", 11, 20),
            seg("X", 20, 21), seg("A", 21, 30)]
    assert merge_tiny_fragmented(segs, RULE_CFG) == 2
    assert all(s["speaker"] == "A" for s in segs)


def test_tiny_fragmented_declines_on_vote_tie():
    segs = [seg("A", 0, 10), seg("X", 10, 11), seg("B", 11, 20),
            seg("X", 20, 21), seg("A", 21, 30)]
    before = [s["speaker"] for s in segs]
    assert merge_tiny_fragmented(segs, RULE_CFG) == 0
    assert [s["speaker"] for s in segs] == before


def test_tiny_fragmented_never_merges_ghost_into_ghost():
    segs = [seg("X", 0, 1), seg("Y", 1, 2), seg("X", 2, 3), seg("Y", 3, 4)]
    assert merge_tiny_fragmented(segs, RULE_CFG) == 0


def test_tiny_fragmented_leaves_large_profiles_alone():
    segs = [seg("A", 0, 60), seg("B", 60, 120), seg("A", 120, 180)]
    assert merge_tiny_fragmented(segs, RULE_CFG) == 0


# --------------------------------------------------------------------------- #
# Kural 3 — Unknown doldurma
# --------------------------------------------------------------------------- #
def test_unknown_filled_when_bounded_by_same_speaker():
    segs = [seg("A", 0, 5), seg("Unknown", 5, 6), seg("A", 6, 11)]
    assert fill_unknown_islands(segs, RULE_CFG) == 1
    assert segs[1]["speaker"] == "A"


def test_unknown_kept_between_different_speakers():
    segs = [seg("A", 0, 5), seg("Unknown", 5, 6), seg("B", 6, 11)]
    assert fill_unknown_islands(segs, RULE_CFG) == 0


def test_unknown_at_edges_is_kept():
    segs = [seg("Unknown", 0, 1), seg("A", 1, 5)]
    assert fill_unknown_islands(segs, RULE_CFG) == 0


def test_calibrating_is_never_filled():
    segs = [seg("A", 0, 5), seg("CALIBRATING", 5, 6), seg("A", 6, 11)]
    assert fill_unknown_islands(segs, RULE_CFG) == 0
    assert segs[1]["speaker"] == "CALIBRATING"


# --------------------------------------------------------------------------- #
# Orkestrasyon
# --------------------------------------------------------------------------- #
def test_refine_does_not_mutate_input():
    original = [seg("A", 0, 10), seg("X", 10, 11), seg("A", 11, 20),
                seg("X", 20, 21), seg("A", 21, 30)]
    snapshot = [dict(s) for s in original]
    refine_speakers(original, RULE_CFG)
    assert original == snapshot


def test_refine_is_deterministic():
    segs = [seg("A", 0, 10), seg("X", 10, 11), seg("A", 11, 20),
            seg("Unknown", 20, 21), seg("A", 21, 30)]
    first, _ = refine_speakers(segs, RULE_CFG)
    second, _ = refine_speakers(segs, RULE_CFG)
    assert [s["speaker"] for s in first] == [s["speaker"] for s in second]


def test_refine_handles_empty_and_single_segment():
    assert refine_speakers([], RULE_CFG)[0] == []
    assert len(refine_speakers([seg("A", 0, 1)], RULE_CFG)[0]) == 1


def test_all_rules_disabled_is_a_noop():
    segs = [seg("A", 0, 10), seg("X", 10, 11), seg("A", 11, 20),
            seg("X", 20, 21), seg("A", 21, 30)]
    cfg = RefinementConfig(small_island_merge=False, tiny_fragmented_merge=False,
                           unknown_fill=False)
    refined, stats = refine_speakers(segs, cfg)
    assert [s["speaker"] for s in refined] == [s["speaker"] for s in segs]
    assert stats["total"] == 0


def test_speaker_summary_sums_durations():
    segs = [seg("A", 0, 10), seg("B", 10, 15)]
    assert speaker_summary(segs) == {"A": 10.0, "B": 5.0}


# --------------------------------------------------------------------------- #
# Varsayılan yapılandırma (AMI'de ölçülerek seçildi: 30/8/8/0.5)
# --------------------------------------------------------------------------- #
def test_defaults_match_measured_optimum():
    cfg = RefinementConfig()
    assert cfg.tiny_fragmented_max_duration == 30.0
    assert cfg.tiny_fragmented_max_segments == 8
    assert cfg.tiny_fragmented_max_islands == 8
    assert cfg.tiny_fragmented_min_neighbor_share == 0.5


def test_defaults_clean_ghosts_on_realistic_meeting_scale():
    """Gerçek toplantı ölçeğinde (dakikalar) hayaletler sahibine döner.

    X iki adada da A ile çevrili → oylama nettir (A=4) ve X, A'ya devredilir.
    """
    segs = [
        seg("A", 0, 120), seg("X", 120, 122), seg("A", 122, 240),
        seg("B", 240, 300),
        seg("A", 300, 360), seg("X", 360, 362), seg("A", 362, 480),
    ]
    refined, stats = refine_speakers(segs, RefinementConfig())
    assert "X" not in {s["speaker"] for s in refined}
    assert stats["tiny_fragmented_merge"] == 2


def test_ghost_with_conflicting_neighbours_is_left_alone():
    """Hayalet farklı bağlamlarda görünüyorsa (bir kez A, bir kez B arasında)
    oylama berabere kalır ve birleştirme YAPILMAZ — sahibi belirsizken tahmin
    yürütmek yanlış atama üretir.
    """
    segs = [
        seg("A", 0, 120), seg("X", 120, 122), seg("A", 122, 240),
        seg("B", 240, 300), seg("X", 300, 302), seg("B", 302, 360),
        seg("A", 360, 480),
    ]
    refined, stats = refine_speakers(segs, RefinementConfig())
    assert [s["speaker"] for s in refined] == [s["speaker"] for s in segs]
    assert stats["total"] == 0


def test_defaults_are_a_safe_noop_on_very_short_sessions():
    """30 sn'lik bir oturumda TÜM etiketler 'minik' sayılır; hayalet->hayalet
    koruması devreye girer ve hiçbir birleştirme yapılmaz (güvenli no-op).

    Bu, kısa canlı oturumlarda refinement'ın zarar vermemesini garanti eder.
    """
    segs = [seg("A", 0, 10), seg("X", 10, 11), seg("A", 11, 20),
            seg("X", 20, 21), seg("A", 21, 30)]
    refined, stats = refine_speakers(segs, RefinementConfig())
    assert [s["speaker"] for s in refined] == [s["speaker"] for s in segs]
    assert stats["total"] == 0
