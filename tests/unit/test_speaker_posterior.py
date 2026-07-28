"""speaker_posterior birim testleri.

Saf Python (torch/model yok) — CI'da tam hızda koşar.

Testler DAVRANIŞI sabitler, kalibre edilmemiş varsayılan SAYILARI değil:
θ ve T AMI'den ölçülecek, ama "net eşleşme yüksek güven verir", "berabere
kalınca güven düşer", "profil sayısı arttıkça şüphe artar" gibi özellikler
kalibrasyondan bağımsız olarak doğru kalmalıdır.
"""

import math

import pytest

from src.core.speaker_posterior import (
    UNKNOWN,
    Posterior,
    PosteriorConfig,
    PosteriorPolicy,
    load_settings,
    speaker_posterior,
)

CFG = PosteriorConfig()


# --------------------------------------------------------------------------- #
# Temel özellikler
# --------------------------------------------------------------------------- #
def test_no_profiles_means_certainly_unknown():
    posterior = speaker_posterior({})
    assert posterior.probabilities == {UNKNOWN: 1.0}
    assert posterior.best_label is None
    assert posterior.unknown_probability == 1.0


def test_probabilities_form_a_distribution():
    posterior = speaker_posterior({"A": 0.80, "B": 0.55, "C": 0.31})
    assert UNKNOWN in posterior.probabilities
    assert sum(posterior.probabilities.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in posterior.probabilities.values())


def test_ranking_is_preserved():
    """En yüksek skor daima en yüksek olasılığı almalı — sıralamaya dokunmuyoruz."""
    scores = {"A": 0.81, "B": 0.55, "C": 0.42, "D": 0.20}
    posterior = speaker_posterior(scores)

    by_score = sorted(scores, key=scores.get, reverse=True)
    by_probability = sorted(scores, key=lambda k: posterior.probabilities[k], reverse=True)
    assert by_score == by_probability
    assert posterior.best_label == "A"


# --------------------------------------------------------------------------- #
# Güvenin anlamı
# --------------------------------------------------------------------------- #
def test_clear_match_yields_high_confidence():
    """Aynı konuşmacı (~0.80) vs diğerleri (~0.30) → net karar.

    Mutlak bir 'unknown < 0.05' beklemiyoruz: o değer doğrudan θ ve T'ye bağlı
    ve ikisi de kalibrasyonla değişecek. Kalibrasyondan bağımsız doğru kalması
    gereken özellik, unknown'ın en iyi seçeneğin yanında KÜÇÜK kalmasıdır.
    """
    posterior = speaker_posterior({"A": 0.80, "B": 0.30, "C": 0.28})
    assert posterior.best_probability > 0.9
    assert posterior.unknown_probability < posterior.best_probability / 5


def test_tie_collapses_confidence():
    """İki profil neredeyse berabere → hiçbiri güvenilir değil.

    Ölçüldü: margin < 0.06 olan kararların %63'ü yanlış. Posterior bunu
    'yüksek olasılık' diye raporlamamalı.
    """
    posterior = speaker_posterior({"A": 0.72, "B": 0.70})
    assert posterior.best_probability < 0.75
    assert posterior.probabilities["A"] == pytest.approx(
        posterior.probabilities["B"], abs=0.15)


def test_everything_below_theta_makes_unknown_win():
    """Hiçbir profil yeterince yakın değilse 'hiçbiri değil' kazanmalı."""
    posterior = speaker_posterior({"A": 0.40, "B": 0.35, "C": 0.30})
    assert posterior.is_unknown_dominant
    assert posterior.unknown_probability > posterior.best_probability


def test_score_at_theta_is_the_fifty_percent_point():
    """θ'da olan tek profil, unknown ile başa baş olmalı (β=0, b₀=0 iken)."""
    config = PosteriorConfig(competitor_slope=0.0, unknown_bias=0.0, duration_slope=0.0)
    posterior = speaker_posterior({"A": config.theta}, config)
    assert posterior.best_probability == pytest.approx(0.5, abs=1e-9)
    assert posterior.unknown_probability == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# Rakip sayısı cezası (β·log K)
# --------------------------------------------------------------------------- #
def test_competitor_penalty_is_excluded_from_the_raw_unknown_mass():
    """β atama güvenini düşürür ama YENİ KONUŞMACI kararına sızmamalı.

    Diaris'te yüksek "unknown" doğrudan aday açma anlamına geliyor; β'yı oraya
    da karıştırmak profil kümesi şiştikçe daha çok konuşmacı ürettirirdi —
    hayalet profilleri besleyen geri besleme döngüsü.
    """
    scores = {"A": 0.62, "B": 0.30, "C": 0.28, "D": 0.25}
    neutral = speaker_posterior(scores, PosteriorConfig(competitor_slope=0.0,
                                                        duration_slope=0.0))
    penalised = speaker_posterior(scores, PosteriorConfig(competitor_slope=1.5,
                                                          duration_slope=0.0))

    # Cezalı sürümde şüphe artar (atama güveni düşer)...
    assert penalised.unknown_probability > neutral.unknown_probability
    assert penalised.best_probability < neutral.best_probability
    # ...ama yeni konuşmacı kararının baktığı HAM kütle değişmez.
    assert penalised.unknown_probability_raw == pytest.approx(
        neutral.unknown_probability_raw, rel=1e-9)


def test_policy_creation_uses_raw_mass_not_the_penalised_one():
    policy = PosteriorPolicy(new_speaker=0.60)
    # Ceza yüzünden şişmiş unknown (0.70) yeni konuşmacı açtırmamalı.
    inflated = _posterior(best=0.25, unknown=0.70, raw_unknown=0.30)
    assert not policy.may_create_speaker(inflated)


def test_more_profiles_raise_suspicion():
    """Aynı en-iyi skor, daha çok rakiple daha az güven vermeli.

    "7 profilin en iyisi eşiği geçti" ile "2 profilin en iyisi geçti" aynı
    iddia değildir.
    """
    few = speaker_posterior({"A": 0.70, "B": 0.30})
    many = speaker_posterior({"A": 0.70, **{f"X{i}": 0.30 for i in range(8)}})
    assert many.unknown_probability > few.unknown_probability
    assert many.best_probability < few.best_probability


def test_single_profile_gets_no_competitor_penalty():
    """K=1 → log(1)=0; rekabet yokken ceza da yok."""
    config = PosteriorConfig(competitor_slope=5.0, duration_slope=0.0)
    lone = speaker_posterior({"A": 0.80}, config)
    neutral = speaker_posterior({"A": 0.80}, PosteriorConfig(competitor_slope=0.0,
                                                             duration_slope=0.0))
    assert lone.unknown_probability == pytest.approx(neutral.unknown_probability)


def test_competitor_slope_scales_the_penalty():
    scores = {"A": 0.70, "B": 0.31, "C": 0.30, "D": 0.29}
    mild = speaker_posterior(scores, PosteriorConfig(competitor_slope=0.1, duration_slope=0.0))
    harsh = speaker_posterior(scores, PosteriorConfig(competitor_slope=1.0, duration_slope=0.0))
    assert harsh.unknown_probability > mild.unknown_probability


# --------------------------------------------------------------------------- #
# Süre → sıcaklık
# --------------------------------------------------------------------------- #
def test_short_audio_flattens_the_posterior():
    """Kısa ses = gürültülü embedding = daha temkinli karar."""
    scores = {"A": 0.80, "B": 0.30, "C": 0.28}
    long_utterance = speaker_posterior(scores, CFG, duration=6.0)
    short_utterance = speaker_posterior(scores, CFG, duration=0.4)

    assert short_utterance.temperature > long_utterance.temperature
    assert short_utterance.best_probability < long_utterance.best_probability


def test_unknown_duration_is_not_penalised():
    scores = {"A": 0.80, "B": 0.30}
    assert (speaker_posterior(scores, CFG, duration=None).temperature
            == pytest.approx(CFG.temperature))
    assert (speaker_posterior(scores, CFG, duration=float("inf")).temperature
            == pytest.approx(CFG.temperature))


def test_zero_duration_does_not_explode():
    """Sıfır/negatif süre bölme taşması yapmamalı."""
    posterior = speaker_posterior({"A": 0.80, "B": 0.30}, CFG, duration=0.0)
    assert math.isfinite(posterior.temperature)
    assert sum(posterior.probabilities.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Sayısal kararlılık
# --------------------------------------------------------------------------- #
def test_extreme_scores_stay_finite():
    posterior = speaker_posterior({"A": 50.0, "B": -50.0}, PosteriorConfig(temperature=0.01))
    assert sum(posterior.probabilities.values()) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in posterior.probabilities.values())
    assert posterior.best_label == "A"


def test_identical_scores_split_evenly():
    posterior = speaker_posterior({"A": 0.5, "B": 0.5, "C": 0.5})
    assert (posterior.probabilities["A"] == pytest.approx(posterior.probabilities["B"])
            == pytest.approx(posterior.probabilities["C"]))


# --------------------------------------------------------------------------- #
# Politika kesim noktaları
# --------------------------------------------------------------------------- #
def _posterior(best: float, unknown: float, raw_unknown: float | None = None) -> Posterior:
    return Posterior(probabilities={"A": best, UNKNOWN: unknown}, best_label="A",
                     best_probability=best, unknown_probability=unknown,
                     temperature=0.1,
                     unknown_probability_raw=unknown if raw_unknown is None else raw_unknown)


def test_policy_separates_assign_from_learn():
    """Etiketi vermek ile profili güncellemek AYRI kararlar."""
    policy = PosteriorPolicy(assign=0.75, learn=0.90, new_speaker=0.60)
    borderline = _posterior(best=0.80, unknown=0.20)

    assert policy.is_confident(borderline)      # etiket güvenle verilir
    assert not policy.may_learn(borderline)     # ama profil güncellenmez


def test_policy_gates_speaker_creation_on_unknown_mass():
    policy = PosteriorPolicy(new_speaker=0.60)
    assert policy.may_create_speaker(_posterior(best=0.25, unknown=0.70))
    assert not policy.may_create_speaker(_posterior(best=0.55, unknown=0.40))


def test_policy_rejects_everything_without_a_best_label():
    policy = PosteriorPolicy()
    empty = speaker_posterior({})
    assert not policy.is_confident(empty)
    assert not policy.may_learn(empty)


def test_end_to_end_confident_match_may_learn():
    policy = PosteriorPolicy()
    posterior = speaker_posterior({"A": 0.85, "B": 0.25}, CFG, duration=5.0)
    assert policy.is_confident(posterior)
    assert policy.may_learn(posterior)
    assert not policy.may_create_speaker(posterior)


def test_end_to_end_short_ambiguous_utterance_learns_nothing():
    """Ölçülen sorunlu senaryo: kısa + belirsiz ses profili kirletmemeli."""
    policy = PosteriorPolicy()
    posterior = speaker_posterior({"A": 0.62, "B": 0.60}, CFG, duration=0.5)
    assert not policy.may_learn(posterior)


# --------------------------------------------------------------------------- #
# Kalibrasyon dosyasının yüklenmesi
# --------------------------------------------------------------------------- #
def _write_calibration(tmp_path, **overrides):
    import json
    payload = {
        "theta": 0.55,
        "temperature": 0.09,
        "suggested_cutpoints": {
            "error<=10%": {"p_best": 0.270, "coverage": 0.642},
            "error<=5%": {"p_best": 0.459, "coverage": 0.487},
            "error<=2%": {"p_best": 0.986, "coverage": 0.001},
        },
    }
    payload.update(overrides)
    path = tmp_path / "speaker_posterior.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_settings_reads_calibrated_parameters(tmp_path):
    settings = load_settings(_write_calibration(tmp_path))
    assert settings is not None
    assert settings.config.theta == pytest.approx(0.55)
    assert settings.config.temperature == pytest.approx(0.09)


def test_load_settings_uses_the_five_percent_cut_for_assignment(tmp_path):
    settings = load_settings(_write_calibration(tmp_path))
    assert settings.policy.assign == pytest.approx(0.459)


def test_learn_gate_ignores_cuts_with_negligible_coverage(tmp_path):
    """En sıkı kesim kapsaması yoksa rezervuarları dondurur — seçilmemeli.

    Ölçülen veride 'hata <= %2' kesimi p=0.986'ya (kapsama %0.1) düşüyordu;
    onu öğrenme kapısı yapmak profillerin bir daha hiç güncellenmemesi demekti.
    """
    settings = load_settings(_write_calibration(tmp_path))
    assert settings.policy.learn < 0.9
    # Kapsaması yeterli olan en sıkı kesim seçilmeli (burada %5 -> 0.459).
    assert settings.policy.learn == pytest.approx(0.459)
    # Öğrenme kapısı atama kapısından asla gevşek olamaz.
    assert settings.policy.learn >= settings.policy.assign


def test_learn_gate_picks_the_strictest_usable_cut(tmp_path):
    path = _write_calibration(tmp_path, **{"suggested_cutpoints": {
        "error<=10%": {"p_best": 0.27, "coverage": 0.64},
        "error<=5%": {"p_best": 0.46, "coverage": 0.49},
        "error<=2%": {"p_best": 0.80, "coverage": 0.25},   # kapsama yeterli
    }})
    settings = load_settings(path)
    assert settings.policy.learn == pytest.approx(0.80)


def test_competitor_slope_can_be_overridden(tmp_path):
    settings = load_settings(_write_calibration(tmp_path), competitor_slope=0.75)
    assert settings.config.competitor_slope == pytest.approx(0.75)


def test_missing_or_broken_calibration_returns_none(tmp_path):
    assert load_settings(tmp_path / "yok.json") is None
    broken = tmp_path / "bozuk.json"
    broken.write_text("{ bu json degil", encoding="utf-8")
    assert load_settings(broken) is None


def test_new_speaker_gate_can_be_overridden(tmp_path):
    """Aşırı/eksik sayım dengesi bir tercih — kalibrasyondan türemez, taranır."""
    default = load_settings(_write_calibration(tmp_path))
    tuned = load_settings(_write_calibration(tmp_path), new_speaker=0.85)
    assert default.policy.new_speaker == pytest.approx(PosteriorPolicy.new_speaker)
    assert tuned.policy.new_speaker == pytest.approx(0.85)


def test_new_speaker_gate_can_come_from_the_calibration_file(tmp_path):
    path = _write_calibration(tmp_path, new_speaker=0.7)
    assert load_settings(path).policy.new_speaker == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Canlı yol bağlantısı
# --------------------------------------------------------------------------- #
def test_live_path_falls_back_when_calibration_is_missing(monkeypatch, tmp_path):
    """Kalibrasyon yoksa uygulama ÇALIŞMAYA DEVAM etmeli (eski eşik yolu).

    Paketlenmiş .exe'de dosya eksik olabilir; teşhis edilebilir bir uyarı
    verilip eski davranışa dönmek, açılışta çökmekten iyidir.
    """
    pytest.importorskip("torch")
    pytest.importorskip("faster_whisper")
    pytest.importorskip("pyannote.audio")
    import src.core.ai_worker as ai_worker

    monkeypatch.setattr(ai_worker, "DIARIZATION_POSTERIOR", True)
    monkeypatch.setattr(ai_worker, "POSTERIOR_CALIBRATION_PATH", str(tmp_path / "yok.json"))
    assert ai_worker.load_posterior_settings_or_none() is None


def test_live_path_can_be_switched_off(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("faster_whisper")
    pytest.importorskip("pyannote.audio")
    import src.core.ai_worker as ai_worker

    monkeypatch.setattr(ai_worker, "DIARIZATION_POSTERIOR", False)
    monkeypatch.setattr(ai_worker, "POSTERIOR_CALIBRATION_PATH",
                        str(_write_calibration(tmp_path)))
    assert ai_worker.load_posterior_settings_or_none() is None


def test_live_path_loads_calibration_when_enabled(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("faster_whisper")
    pytest.importorskip("pyannote.audio")
    import src.core.ai_worker as ai_worker

    monkeypatch.setattr(ai_worker, "DIARIZATION_POSTERIOR", True)
    monkeypatch.setattr(ai_worker, "POSTERIOR_CALIBRATION_PATH",
                        str(_write_calibration(tmp_path)))
    monkeypatch.setattr(ai_worker, "POSTERIOR_NEW_SPEAKER", 0.55)

    settings = ai_worker.load_posterior_settings_or_none()
    assert settings is not None
    assert settings.config.theta == pytest.approx(0.55)
    assert settings.policy.new_speaker == pytest.approx(0.55)


def test_shipped_calibration_file_is_loadable():
    """Repodaki kalibrasyon dosyası gerçekten okunabilir olmalı.

    Canlı varsayılan AÇIK; bu dosya bozulursa sistem sessizce eski yola döner
    ve iyileşme kaybolur — test bunu erken yakalar.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    settings = load_settings(root / "calibration" / "speaker_posterior.json")
    assert settings is not None
    assert 0.0 < settings.config.theta < 1.0
    assert settings.config.temperature > 0.0
    assert settings.policy.learn >= settings.policy.assign
