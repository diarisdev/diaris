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
# Warm-up gürültü filtresi
#
# REGRESYON: eski kural (`n >= 6` ise tek üyeli kümeleri at) gerçek AMI
# verisinde 4 konuşmacıyı 1-2'ye çöktürüyordu. Tek üyeli kümeler gürültü değil,
# warm-up penceresinde bir kez konuşmuş insanlardı.
# --------------------------------------------------------------------------- #
def _filter(clusters, n):
    from src.core.speaker_tracker import SpeakerTracker
    return SpeakerTracker(threshold=0.70, warmup_ms=1000)._filter_noise_clusters(
        clusters, n)


def test_singletons_survive_when_data_is_thin():
    """ÖLÇÜLEN SENARYO: IS1009a, 6 embedding, kümeler [3,1,1,1], gerçek 4 kişi."""
    clusters = {0: [0, 1, 2], 1: [3], 2: [4], 3: [5]}
    assert len(_filter(clusters, n=6)) == 4, "gerçek konuşmacılar elenmemeliydi"


def test_singletons_survive_when_they_are_the_majority():
    """Kümelerin çoğu tek üyeliyse bu gürültü değil, yetersiz örneklemedir."""
    clusters = {0: list(range(6)), 1: [6], 2: [7], 3: [8], 4: [9], 5: [10], 6: [11]}
    assert len(_filter(clusters, n=12)) == 7


def test_lone_singleton_is_dropped_when_evidence_is_plentiful():
    """Herkes birkaç kez konuşmuşken YALNIZ kalan bir embedding gerçekten sıra dışı."""
    clusters = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11], 4: [12]}
    filtered = _filter(clusters, n=13)
    assert len(filtered) == 4
    assert 4 not in filtered


def test_clusters_without_singletons_pass_through():
    clusters = {0: [0, 1], 1: [2, 3]}
    assert _filter(clusters, n=4) == clusters


def test_filter_never_returns_nothing():
    """Her şey elenirse en büyük küme kurtarılmalı — konuşmacısız kalmayalım."""
    clusters = {0: [0], 1: [1]}
    assert len(_filter(clusters, n=2)) >= 1


def test_warmup_end_to_end_keeps_under_sampled_speakers():
    """Gerçek akış: 4 farklı ses, biri iki kez duyulmuş → 4 profil olmalı."""
    torch = pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=1000)

    voices = [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.05, 0.0, 0.0]),   # aynı kişi, ikinci gözlem
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.7, 0.7]),    # kimseye tam benzemiyor
    ]
    tracker.add_warmup_chunk(voices, 2000)

    assert not tracker.is_warming_up
    # Eski filtre burada [2,1,1,1,1] kümelerinden yalnız [2]'yi bırakıp
    # 1 konuşmacıya çökertirdi.
    assert len(tracker.known_speakers) >= 4


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


# --------------------------------------------------------------------------- #
# Kohort (AS-norm) normalizasyonu
# --------------------------------------------------------------------------- #
def _cohort_tracker(cohort_norm):
    from src.core.speaker_tracker import SpeakerTracker
    tracker = SpeakerTracker(threshold=0.60, warmup_ms=0, cohort_norm=cohort_norm)
    tracker._warmup_complete = True
    return tracker


def test_cohort_normalization_is_a_no_op_when_disabled():
    """Varsayılan (0.0) mevcut davranışı BİREBİR korumalı."""
    tracker = _cohort_tracker(0.0)
    raw = {"A": 0.80, "B": 0.40, "C": 0.30}
    assert tracker._cohort_normalize(raw) == raw
    assert tracker._cohort_reference is None


def test_cohort_normalization_needs_enough_profiles():
    """2 profille 'diğerleri' tek örnek; istatistik anlamsız → dokunma."""
    tracker = _cohort_tracker(1.0)
    raw = {"A": 0.80, "B": 0.40}
    assert tracker._cohort_normalize(raw) == raw


def test_first_observation_seeds_the_reference_and_leaves_the_best_untouched():
    """İlk chunk referansı tohumlar; eşiğin sınadığı EN İYİ skor kaymaz."""
    tracker = _cohort_tracker(1.0)
    raw = {"A": 0.80, "B": 0.40, "C": 0.30}
    out = tracker._cohort_normalize(raw)

    # impostor ortalaması = en iyi (0.80) hariç = (0.40 + 0.30) / 2
    assert tracker._cohort_reference == pytest.approx(0.35, abs=1e-9)
    # Referans bu utterance'tan tohumlandığı için en iyi skorun düzeltmesi sıfır.
    assert out["A"] == pytest.approx(raw["A"], abs=1e-9)


def test_margins_are_scaled_by_the_cohort_size():
    """Belgelenmiş yan etki: margin (1 + alpha/(K-1)) katına çıkar.

    Dönüşüm s_i'de doğrusal olduğu için sıralama korunur ama skorlar arası
    mesafe genişler — MIN_DECISION_MARGIN kapısı bu ölçekte yeniden okunmalı.
    """
    tracker = _cohort_tracker(1.0)
    tracker._cohort_reference = 0.30
    raw = {"A": 0.80, "B": 0.40, "C": 0.30}          # K = 3 → beklenen kat 1.5

    out = tracker._cohort_normalize(raw)

    raw_margin = raw["A"] - raw["B"]
    new_margin = out["A"] - out["B"]
    assert new_margin == pytest.approx(raw_margin * 1.5, rel=1e-9)


def test_noisy_embedding_is_pushed_down():
    """Her şeye yakın duran (gürültülü) embedding cezalandırılmalı."""
    tracker = _cohort_tracker(1.0)
    tracker._cohort_reference = 0.30          # öğrenilmiş normal seviye

    noisy = {"A": 0.72, "B": 0.68, "C": 0.66}  # hepsi yüksek → şüpheli
    out = tracker._cohort_normalize(noisy)

    assert out["A"] < noisy["A"], "gürültülü embedding aşağı çekilmeliydi"


def test_clean_embedding_is_lifted():
    """Diğer herkesten uzak duran (temiz) embedding ödüllendirilmeli."""
    tracker = _cohort_tracker(1.0)
    tracker._cohort_reference = 0.30

    clean = {"A": 0.64, "B": 0.10, "C": 0.05}  # sadece A'ya yakın
    out = tracker._cohort_normalize(clean)

    assert out["A"] > clean["A"], "temiz embedding yukarı çekilmeliydi"


def test_normalization_preserves_ranking():
    """Dönüşüm monoton: en-yakın-komşu sıralaması ASLA değişmemeli."""
    tracker = _cohort_tracker(1.0)
    tracker._cohort_reference = 0.30
    raw = {"A": 0.81, "B": 0.55, "C": 0.42, "D": 0.20}

    out = tracker._cohort_normalize(raw)

    assert sorted(raw, key=raw.get, reverse=True) == sorted(out, key=out.get, reverse=True)


def test_strength_scales_the_correction():
    """alpha düzeltmenin büyüklüğünü doğrusal ölçeklemeli."""
    raw = {"A": 0.72, "B": 0.68, "C": 0.66}

    half = _cohort_tracker(0.5)
    half._cohort_reference = 0.30
    full = _cohort_tracker(1.0)
    full._cohort_reference = 0.30

    shift_half = raw["A"] - half._cohort_normalize(raw)["A"]
    shift_full = raw["A"] - full._cohort_normalize(raw)["A"]
    assert shift_full == pytest.approx(2 * shift_half, rel=1e-9)


def test_reference_tracks_the_impostor_level():
    """Referans EMA ile gözlenen impostor seviyesine yaklaşmalı."""
    tracker = _cohort_tracker(1.0)
    tracker._cohort_reference = 0.30

    for _ in range(200):
        tracker._cohort_normalize({"A": 0.90, "B": 0.50, "C": 0.50})

    # impostor ortalaması sabit 0.50 → referans oraya yakınsamalı
    assert tracker._cohort_reference == pytest.approx(0.50, abs=0.01)


def test_reset_clears_the_learned_reference():
    tracker = _cohort_tracker(1.0)
    tracker._cohort_normalize({"A": 0.8, "B": 0.4, "C": 0.3})
    assert tracker._cohort_reference is not None

    tracker.reset()
    assert tracker._cohort_reference is None


def test_map_speakers_records_raw_and_normalized_scores():
    """İz her iki skoru da taşımalı — düzeltmenin etkisi görülebilsin."""
    torch = pytest.importorskip("torch")
    tracker = _cohort_tracker(1.0)
    for vector in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        tracker._register_speaker([torch.tensor(vector)])
    tracker.debug_enabled = True

    tracker.map_speakers({"local": torch.tensor([0.9, 0.3, 0.2])}, {"local": 4.0})

    probe, = tracker.last_trace["probes"]
    assert set(probe["raw_scores"]) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
    assert set(probe["scores"]) == set(probe["raw_scores"])
    assert probe["cohort_reference"] is not None


# --------------------------------------------------------------------------- #
# Posterior entegrasyonu
#
# En kritik test: posterior KAPALIYKEN karar yolu birebir eskisi gibi olmalı.
# A/B tek anahtarla yapılabilsin diye kapılar tek yerde toplandı; o refactor
# mevcut davranışı değiştirmemiş olmalı.
# --------------------------------------------------------------------------- #
def _tracker_with_profiles(torch, **kwargs):
    """3 dik profil, 4 boyutlu uzayda.

    4. boyut bilerek boş: "hiçbir profile benzemeyen" bir ses ancak profillerin
    germediği bir yönde var olabilir. 3 boyutta 3 dik profille en uzak nokta
    bile hepsine 0.577 benzer — gerçek bir yabancı temsil edilemez.
    """
    from src.core.speaker_tracker import SpeakerTracker
    tracker = SpeakerTracker(threshold=0.60, warmup_ms=0, **kwargs)
    tracker._warmup_complete = True
    for vector in ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]):
        tracker._register_speaker([torch.tensor(vector)])
    return tracker


def _settings(assign=0.45, learn=0.80, new_speaker=0.60, **config_kwargs):
    from src.core.speaker_posterior import (
        PosteriorConfig, PosteriorPolicy, PosteriorSettings,
    )
    return PosteriorSettings(
        config=PosteriorConfig(theta=0.55, temperature=0.09, **config_kwargs),
        policy=PosteriorPolicy(assign=assign, learn=learn, new_speaker=new_speaker),
    )


def test_tracker_defaults_to_the_legacy_decision_path():
    """posterior=None → eski kapılar, eski karar isimleri."""
    torch = pytest.importorskip("torch")
    tracker = _tracker_with_profiles(torch)
    tracker.debug_enabled = True

    tracker.map_speakers({"local": torch.tensor([0.99, 0.05, 0.0, 0.0])}, {"local": 5.0})

    probe, = tracker.last_trace["probes"]
    assert probe["decision"] == "matched"
    assert probe["p_best"] is None          # posterior kapalı → güven yok
    assert tracker.posterior is None


def test_legacy_path_still_sticks_short_utterances():
    """Kapalıyken kısa/eşleşmeyen ses eskisi gibi 'sticky_short' olmalı."""
    torch = pytest.importorskip("torch")
    tracker = _tracker_with_profiles(torch)
    tracker.debug_enabled = True

    tracker.map_speakers({"local": torch.tensor([0.6, 0.6, 0.5, 0.0])}, {"local": 0.4})

    probe, = tracker.last_trace["probes"]
    assert probe["decision"] == "sticky_short"


def test_posterior_path_records_confidence():
    torch = pytest.importorskip("torch")
    tracker = _tracker_with_profiles(torch, posterior=_settings())
    tracker.debug_enabled = True

    tracker.map_speakers({"local": torch.tensor([0.99, 0.05, 0.0, 0.0])}, {"local": 5.0})

    probe, = tracker.last_trace["probes"]
    assert probe["decision"] == "matched"
    assert 0.0 <= probe["p_best"] <= 1.0
    assert probe["p_unknown"] is not None


def test_posterior_does_not_credit_speech_to_low_confidence_matches():
    """ÖLÇÜLEN SORUN: düşük güvenli atamanın süresi yanlış profili besliyordu.

    Posterior yolunda etiket yine verilir (ekranda bir şey görünmeli) ama
    profilin olgunluk hanesine yazılmaz.
    """
    torch = pytest.importorskip("torch")
    legacy = _tracker_with_profiles(torch)
    modern = _tracker_with_profiles(torch, posterior=_settings(assign=0.99))

    ambiguous = torch.tensor([0.60, 0.58, 0.0, 0.0])
    # Profiller kayıt anında tohum olgunluk kredisi alır; taban sıfır değil,
    # bu yüzden FARKA bakıyoruz.
    legacy_before = sum(legacy._speech_seconds.values())
    modern_before = sum(modern._speech_seconds.values())

    legacy.map_speakers({"local": ambiguous}, {"local": 0.4})
    modern.map_speakers({"local": ambiguous}, {"local": 0.4})

    assert sum(legacy._speech_seconds.values()) > legacy_before   # eski: yazıyor
    assert sum(modern._speech_seconds.values()) == modern_before  # yeni: yazmıyor


def test_posterior_still_labels_low_confidence_utterances():
    """Etiket kaybolmamalı — yalnızca öğrenme durur (DER riski yok)."""
    torch = pytest.importorskip("torch")
    tracker = _tracker_with_profiles(torch, posterior=_settings(assign=0.99))
    tracker.debug_enabled = True

    mapping = tracker.map_speakers({"local": torch.tensor([0.60, 0.58, 0.0, 0.0])},
                                   {"local": 0.4})

    assert mapping["local"] in tracker.known_speakers
    probe, = tracker.last_trace["probes"]
    assert probe["decision"] == "low_confidence"


def test_new_speaker_gate_controls_candidate_creation():
    """Hayalet konuşmacı kapısı: yalnızca ham "hiçbiri değil" kanitı yeterliyse açılır."""
    torch = pytest.importorskip("torch")
    stranger = torch.tensor([0.0, 0.0, 0.0, 1.0])   # hiçbir profile benzemiyor
    familiar = torch.tensor([0.95, 0.10, 0.0, 0.0])  # SPEAKER_00'a çok benziyor

    opens = _tracker_with_profiles(torch, posterior=_settings(new_speaker=0.60))
    opens.map_speakers({"local": stranger}, {"local": 5.0})
    assert len(opens._candidates) == 1, "gerçek yabancı için aday açılmalıydı"

    # Tanıdık ses aday açmamalı (zaten güvenle eşleşiyor).
    quiet = _tracker_with_profiles(torch, posterior=_settings(new_speaker=0.60))
    quiet.map_speakers({"local": familiar}, {"local": 5.0})
    assert len(quiet._candidates) == 0


def test_competitor_penalty_does_not_leak_into_speaker_creation():
    """TASARIM DÜZELTMESİ: β yeni konuşmacı açmayı KOLAYLAŞTIRMAMALI.

    β·log(K) "hiçbiri değil" kütlesini yükseltir. Diaris'te yüksek unknown
    doğrudan aday açma demek olduğu için, β'yı o karara karıştırmak profil
    kümesi şiştikçe daha çok konuşmacı üretirdi — hayaleti besleyen döngü.
    Yeni konuşmacı kararı HAM kanıta bakmalı, β'dan etkilenmemeli.
    """
    torch = pytest.importorskip("torch")
    # SPEAKER_00'a sınırda benzeyen ses: β devreye girerse unknown şişer.
    borderline = torch.tensor([0.62, 0.30, 0.20, 0.0])

    without_beta = _tracker_with_profiles(
        torch, posterior=_settings(new_speaker=0.60, competitor_slope=0.0))
    with_beta = _tracker_with_profiles(
        torch, posterior=_settings(new_speaker=0.60, competitor_slope=2.0))

    without_beta.map_speakers({"local": borderline}, {"local": 5.0})
    with_beta.map_speakers({"local": borderline}, {"local": 5.0})

    assert len(with_beta._candidates) == len(without_beta._candidates),         "beta yeni konuşmacı kararını etkilememeli"


def test_posterior_does_not_change_the_ranking():
    """Hangi profil seçildiği değişmemeli — sıralamaya dokunmuyoruz."""
    torch = pytest.importorskip("torch")
    embedding = torch.tensor([0.9, 0.3, 0.1, 0.0])

    legacy = _tracker_with_profiles(torch)
    modern = _tracker_with_profiles(torch, posterior=_settings())

    assert (legacy.map_speakers({"local": embedding}, {"local": 5.0})
            == modern.map_speakers({"local": embedding}, {"local": 5.0}))


def test_warmup_ignores_chunks_without_embeddings():
    """Embedding üretmeyen chunk kalibrasyon süresini ilerletmez."""
    pytest.importorskip("torch")
    tracker = _tracker(warmup_ms=10000)

    assert tracker.add_warmup_chunk([], 8000) is False
    assert tracker.add_warmup_chunk([None], 8000) is False
    assert tracker._warmup_audio_ms == 0
    assert tracker.is_warming_up

