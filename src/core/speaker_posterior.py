"""Konuşmacı kararı için kalibre edilebilir posterior.

Bugünkü karar tek bir sert eşiktir: `best_score >= effective_threshold`. Üç
yapısal sorunu var:

1. UÇURUM — 0.659 ile 0.661 tamamen farklı kod yollarına gider, oysa aradaki
   fark gürültüdür (aynı kodun iki koşusunda 0.19 DER oynaması ölçüldü).
2. RAKİP SAYISINA KÖR — "7 profilin en iyisi eşiği geçti", "2 profilin en iyisi
   geçti" ile aynı sayılır; oysa ilki çok daha zayıf bir iddiadır.
3. İKİLİ — karar ya "eşleşti" ya "eşleşmedi". Oysa arkasında üç ayrı soru var
   ("etiketi ver mi", "profili güncelle mi", "yeni konuşmacı aç mı") ve her biri
   ayrı, elle ayarlanmış, birbirine bağlı bir kapıyla cevaplanıyor.

Bu modül eşiği KALDIRMAZ; onu bir kapı olmaktan çıkarıp REFERANS NOKTASINA
çevirir ve kararı olasılığa taşır:

    ℓ_i        = (s_i − θ) / T
    ℓ_unknown  = (θ − max s_i) / T + b₀ + β·log(K)
    P          = softmax(ℓ)

* θ — "aynı konuşmacı" olasılığının %50 olduğu skor (AMI'den kalibre edilir).
* T — skor farklarının olasılığa ne kadar keskin çevrildiği.
* β — profil sayısı (K) arttıkça "hiçbiri değil" seçeneğinin ağırlığı.
       `speaker_tracker.py`'de ölçülmüş "10 konuşmacı varken herkes bir eşleşme
       bulur" tuzağının prensipli karşılığı.

"unknown" bir KAPI değil, YARIŞMACIDIR: kendi logit'i vardır ve diğerleriyle
aynı softmax'ta yarışır. En iyi eşleşme θ'nın altındaysa logit'i pozitife geçer.

SIRALAMAYA DOKUNMAZ: ℓ_i, s_i'de monotondur; en yüksek skorlu profil daima en
yüksek olasılığı alır. (Sıralı listede aşağı inmek ölçülerek denendi ve
diarization'ı ciddi biçimde bozdu — bkz. speaker_tracker.map_speakers.)

Saf Python (torch/model/IO yok) — birim testleri modelsiz koşar.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

UNKNOWN = "unknown"

# Süre bilinmiyorsa/sonsuzsa sıcaklık cezası uygulanmaz; sıfır süre için bölme
# taşmasını önleyen taban.
_MIN_DURATION = 0.05

# Öğrenme kapısı bu kapsamanın altına inemez: çok sıkı bir kapı rezervuarları
# dondurur ve profiller hiç güncellenmez.
MIN_LEARN_COVERAGE = 0.10


@dataclass(frozen=True)
class PosteriorConfig:
    """Posterior parametreleri.

    VARSAYILANLAR GEÇİCİDİR: θ ve T, AMI referansından lojistik uydurmayla
    ölçülecek (scripts/calibrate_speaker_posterior.py). Buradaki değerler
    yalnızca kalibrasyon öncesi makul bir başlangıçtır — mevcut eşik (0.66) ve
    skor ölçeğimizde (aynı konuşmacı ~0.75, farklı ~0.30) anlamlı davranan bir
    sıcaklık.
    """

    # %50 noktası. Bugünkü DIARIZATION_EMBEDDING_THRESHOLD'un yerini alır ama
    # kapı olarak değil, orijin olarak.
    theta: float = 0.66
    # Taban sıcaklık. 0.45'lik bir skor farkı (aynı vs farklı konuşmacı) ~0.99
    # olasılığa, 0.06'lık sınır margin'i ~0.64'e karşılık gelir.
    temperature: float = 0.10
    # "unknown" için sabit kayma. 0 = nötr.
    unknown_bias: float = 0.0
    # Rakip sayısı cezası: β·log(K). Taranacak asıl parametre.
    competitor_slope: float = 0.25
    # Süre → sıcaklık: T(d) = T·(1 + k/d). Kısa ses daha DÜZ posterior verir,
    # yani kendiliğinden temkinli olur. Eşiğe zam yapan quality/maturity
    # cezalarının yerini alır.
    duration_slope: float = 1.0

    def temperature_for(self, duration: float | None) -> float:
        """Süreye göre etkin sıcaklık.

        Kısa sesten çıkan embedding gürültülüdür; onu daha düşük güvenle
        değerlendirmenin doğru yolu eşiği oynatmak değil, olasılık dağılımını
        düzleştirmektir.
        """
        if self.duration_slope <= 0.0 or duration is None:
            return self.temperature
        if duration == float("inf") or duration != duration:  # inf / NaN
            return self.temperature
        return self.temperature * (1.0 + self.duration_slope / max(duration, _MIN_DURATION))


@dataclass(frozen=True)
class Posterior:
    """Bir embedding için {konuşmacılar} ∪ {unknown} üzerinde olasılık dağılımı."""

    probabilities: dict           # {etiket: p}, "unknown" dahil; toplamı 1
    best_label: str | None        # en olası GERÇEK konuşmacı (unknown hariç)
    best_probability: float       # onun olasılığı
    unknown_probability: float    # β·log(K) cezası DAHİL
    temperature: float            # bu karar için kullanılan etkin sıcaklık
    # β·log(K) cezası HARİÇ "hiçbiri değil" kütlesi.
    #
    # NEDEN İKİ AYRI DEĞER: β, profil sayısı arttıkça şüpheyi artırır ve bu
    # ATAMA için doğrudur (kalabalık profil kümesinde eşleşmeye daha az güven).
    # Ama Diaris'te yüksek "unknown" YENİ KONUŞMACI AÇMA anlamına gelir; β'yı
    # oraya da karıştırmak, tam da hayalet profillerin çoğaldığı durumda daha
    # çok konuşmacı yaratırdı — yani sorunu besleyen bir geri besleme döngüsü.
    # Yeni konuşmacı kararı bu yüzden HAM kanıta bakar.
    unknown_probability_raw: float = 0.0

    @property
    def is_unknown_dominant(self) -> bool:
        return self.unknown_probability >= self.best_probability


def _softmax(logits: dict) -> dict:
    """Sayısal olarak kararlı softmax (en büyük logit çıkarılır)."""
    if not logits:
        return {}
    peak = max(logits.values())
    exponentials = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(exponentials.values())
    if total <= 0.0:
        uniform = 1.0 / len(logits)
        return {key: uniform for key in logits}
    return {key: value / total for key, value in exponentials.items()}


def speaker_posterior(scores, config: PosteriorConfig | None = None,
                      duration: float | None = None) -> Posterior:
    """Benzerlik skorlarını olasılık dağılımına çevirir.

    Args:
        scores: {konuşmacı_etiketi: benzerlik_skoru}. Boşsa sonuç saf "unknown".
        config: parametreler (None → varsayılanlar).
        duration: bu embedding'in temiz konuşma süresi (sn). Verilirse sıcaklık
            kısa seslerde yükseltilir (daha düz, daha temkinli posterior).

    Returns:
        Posterior. "unknown" her zaman dağılımın içindedir.
    """
    config = config or PosteriorConfig()
    temperature = config.temperature_for(duration)

    if not scores:
        # Hiç profil yok: kararsızlık değil, kesinlik — bu ses bilinmeyendir.
        return Posterior(probabilities={UNKNOWN: 1.0}, best_label=None,
                         best_probability=0.0, unknown_probability=1.0,
                         temperature=temperature, unknown_probability_raw=1.0)

    count = len(scores)
    top_score = max(scores.values())

    logits = {label: (score - config.theta) / temperature
              for label, score in scores.items()}
    # "unknown" logit'i: en iyi eşleşme θ'nın NE KADAR altında kaldığı.
    base_unknown = (config.theta - top_score) / temperature + config.unknown_bias
    # Rakip cezası: profil sayısıyla büyüyen şüphe. K=1'de log(1)=0 → ceza yok.
    penalty = config.competitor_slope * math.log(count)

    logits[UNKNOWN] = base_unknown + penalty
    probabilities = _softmax(logits)

    # Cezasız dağılım: yeni konuşmacı kararı bunun üzerinden verilir.
    raw_logits = dict(logits)
    raw_logits[UNKNOWN] = base_unknown
    raw_unknown = _softmax(raw_logits).get(UNKNOWN, 0.0) if penalty else \
        probabilities.get(UNKNOWN, 0.0)

    best_label = max(scores, key=lambda label: scores[label])
    return Posterior(
        probabilities=probabilities,
        best_label=best_label,
        best_probability=probabilities.get(best_label, 0.0),
        unknown_probability=probabilities.get(UNKNOWN, 0.0),
        temperature=temperature,
        unknown_probability_raw=raw_unknown,
    )


@dataclass(frozen=True)
class PosteriorPolicy:
    """Posterior'dan üç ayrı kararın türetildiği kesim noktaları.

    Bugün bu üç soru üç ayrı, elle ayarlanmış ve BİRBİRİNE BAĞLI kapıyla
    cevaplanıyor (effective_threshold / MIN_DECISION_MARGIN + 0.85 /
    MIN_NEW_SPEAKER_DURATION). Kohort deneyi tam bu bağımlılık yüzünden
    yorumlanamaz çıktı: skor ölçeği kayınca üçü birden kaydı.

    Kalibrasyon sonrası bu sayılar gerçek olasılık anlamına gelir — "profili
    yalnızca %90'dan eminken güncelle" cümlesi doğrudan okunabilir.
    """

    # Etiketi güvenle ver. Altında da etiket VERİLİR (en iyi tahmin) ama
    # düşük güvenli işaretlenir.
    assign: float = 0.75
    # Profili güncelle (rezervuar + olgunluk hanesi). Kirlenmeyi burası önler.
    learn: float = 0.90
    # Yeni konuşmacı adayı açmaya izin ver.
    new_speaker: float = 0.60

    def is_confident(self, posterior: Posterior) -> bool:
        return posterior.best_label is not None and posterior.best_probability >= self.assign

    def may_learn(self, posterior: Posterior) -> bool:
        """Profil güncellemesi yalnızca YÜKSEK güvende.

        Ölçüldü: düşük güvenli kararların ~%46-49'u yanlış. Bugün bunların
        konuşma süresi yine de profile yazılıyor ve yanlış profili besliyor.
        """
        return posterior.best_label is not None and posterior.best_probability >= self.learn

    def may_create_speaker(self, posterior: Posterior) -> bool:
        """Yeni konuşmacı ancak "hiçbiri değil" güçlüyken açılabilir.

        HAM (β cezasız) kütleye bakar. β profil sayısıyla şüpheyi artırır; onu
        buraya karıştırmak, profil kümesi şiştikçe daha ÇOK yeni konuşmacı
        açtırırdı — hayalet üretimini besleyen bir geri besleme döngüsü.
        Yeni konuşmacı kararı, "bu ses gerçekten hiçbirine benzemiyor" ham
        kanıtına dayanmalı.
        """
        return posterior.unknown_probability_raw >= self.new_speaker


@dataclass(frozen=True)
class PosteriorSettings:
    """Kalibre parametreler + politika kesim noktaları, tek pakette."""

    config: PosteriorConfig
    policy: PosteriorPolicy


def load_settings(path, competitor_slope: float | None = None,
                  new_speaker: float | None = None) -> PosteriorSettings | None:
    """Kalibrasyon dosyasından ayarları yükler.

    Dosya `scripts/calibrate_speaker_posterior.py` tarafından üretilir ve θ/T'yi
    AMI referansından ÖLÇÜLMÜŞ olarak taşır. Politika kesim noktaları da oradaki
    "hedef hata oranına karşılık gelen p_best" önerilerinden okunur — elle
    seçilmiş sayılar olmasınlar diye.

    Args:
        path: kalibrasyon JSON'u. Yoksa None döner (çağıran posterior'u kapatır).
        competitor_slope: β için üst-geçersiz kılma (AMI taramasında kullanılır).
        new_speaker: yeni konuşmacı açma kapısı için üst-geçersiz kılma. Bu
            değer kalibrasyondan TÜRETİLEMEZ (aşırı/eksik sayım dengesi bir
            tercihtir, ölçüm değil) — taranarak seçilir.

    Returns:
        PosteriorSettings, ya da dosya yok/bozuksa None.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    config = PosteriorConfig(
        theta=float(payload.get("theta", PosteriorConfig.theta)),
        temperature=float(payload.get("temperature", PosteriorConfig.temperature)),
    )
    if competitor_slope is not None:
        config = replace(config, competitor_slope=float(competitor_slope))

    cuts = payload.get("suggested_cutpoints") or {}

    def cut(name: str, fallback: float) -> float:
        entry = cuts.get(name)
        if isinstance(entry, dict) and "p_best" in entry:
            return float(entry["p_best"])
        return fallback

    # ATAMA: "hata <= %5" bandı — etiketi göstermek için makul denge.
    assign = cut("error<=5%", PosteriorPolicy.assign)

    # ÖĞRENME: en sıkı band, AMA kapsaması anlamlı olmak zorunda. Ölçtüğümüz
    # veride "hata <= %2" kesimi 0.986'ya (kapsama %0.1) düşüyor; onu öğrenme
    # kapısı yapmak rezervuarları tamamen dondururdu — profiller bir daha hiç
    # güncellenmez, sistem ilk günkü haliyle kalırdı.
    learn = assign
    for name, entry in sorted(cuts.items()):
        if not isinstance(entry, dict) or "p_best" not in entry:
            continue
        if float(entry.get("coverage", 0.0)) < MIN_LEARN_COVERAGE:
            continue
        learn = max(learn, float(entry["p_best"]))

    creation = PosteriorPolicy.new_speaker
    if new_speaker is not None:
        creation = float(new_speaker)
    elif "new_speaker" in payload:
        creation = float(payload["new_speaker"])

    policy = PosteriorPolicy(assign=assign, learn=learn, new_speaker=creation)
    return PosteriorSettings(config=config, policy=policy)
