"""SpeakerTracker — embedding tabanlı konuşmacı kimlik takibi.

ai_worker.py'den ayrıldı; ai_worker bunu geri export eder; `from
src.core.ai_worker import SpeakerTracker` ve subclass'lar (EvalSpeakerTracker)
aynen çalışmaya devam eder.

Rezervuar modeli
----------------
Konuşmacı başına TEK centroid, ses tonu/kanal değişimlerine karşı kırılgandı:
centroid "ortalama bir sese" yakınsadıkça ya herkes ona benziyor (over-collapse)
ya da aynı kişinin farklı konuşma tarzları eşiğin altında kalıyordu. Artık her
konuşmacı için son K kaliteli embedding bir REZERVUARDA tutulur; benzerlik
skoru, rezervuar + centroid adaylarının en iyi ikisinin ortalamasıdır. Centroid
rezervuarın normalize ortalaması olarak korunur (eski API'ler — known_speakers
dict'i — birebir çalışır; EvalSpeakerTracker gibi subclass'lar rezervuarsız
konuşmacı yazarsa skor otomatik centroid-only'ye düşer).
"""

import torch

from ..config import (
    DIARIZATION_EMBEDDING_THRESHOLD, DIARIZATION_WARMUP_MS,
    CANDIDATE_CONFIRMATIONS_NEEDED, CANDIDATE_TTL, CANDIDATE_SELF_SIMILARITY,
    MIN_NEW_SPEAKER_DURATION, DIARIZATION_COHORT_NORM,
)
from .speaker_posterior import speaker_posterior


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def _normalized_mean(embeddings: list) -> torch.Tensor:
    centroid = torch.stack(embeddings).mean(dim=0)
    norm = torch.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


class SpeakerTracker:
    """
    Embedding-based konuşmacı takip sistemi (warm-up destekli).

    Faz 1 (Warm-up): Kaliteli embedding'ler toplanır, kümelenir.
    Faz 2 (Aktif):   Yeni embedding'ler konuşmacı rezervuarlarıyla karşılaştırılır;
                      güçlü eşleşmeler rezervuara eklenir (centroid türetilir).

    Yeni konuşmacı ekleme: Onay tamponu (confirmation buffer) sistemi.
    - Bilinmeyen bir ses ilk görüldüğünde hemen konuşmacı oluşturulmaz.
    - Embedding "aday" olarak tamponlanır.
    - Aynı bilinmeyen ses birden fazla kez tutarlı şekilde görülürse
      (embedding'ler kendi aralarında benzer) yeni konuşmacı oluşturulur.
    - Tek seferlik sesler (oyun efektleri, müzik) otomatik filtrelenir.
    """

    # Aday tamponu sabitleri
    CANDIDATE_CONFIRMATIONS_NEEDED = CANDIDATE_CONFIRMATIONS_NEEDED   # Yeni konuşmacı için gereken min gözlem
    CANDIDATE_TTL = CANDIDATE_TTL                                     # Onaylanmayan adaylar kaç chunk sonra silinir
    CANDIDATE_SELF_SIMILARITY = CANDIDATE_SELF_SIMILARITY             # Aday embedding'lerin kendi aralarındaki min benzerlik

    # Rezervuar sabitleri
    RESERVOIR_SIZE = 8            # konuşmacı başına saklanan max embedding
    RESERVOIR_ADD_THRESHOLD = 0.85  # rezervuara ekleme için min eşleşme skoru (drift önleme)

    # --- Karar kalitesi sabitleri -------------------------------------- #
    # 1) MARGIN: en iyi ile ikinci en iyi konuşmacı arasındaki fark. İki profil
    #    neredeyse berabereyse eşiği geçmek tek başına bir şey ifade etmez —
    #    atama yazı-tura olur. Bu durumda etiket yine verilir (en iyi tahmin)
    #    ama profil GÜNCELLENMEZ; belirsiz ses profili kirletirse sonraki
    #    chunk'larda karışıklık (DER confusion) katlanarak büyür.
    MIN_DECISION_MARGIN = 0.06

    # 2) SÜRE KALİTESİ: kısa sesten çıkan embedding gürültülüdür. Sert eşikle
    #    atmak yerine (bilgi kaybı) eşleşme eşiğini geçici olarak yükseltiriz.
    #    quality: 0.25 (çok kısa) .. 1.0 (>= QUALITY_FULL_SEC)
    QUALITY_MIN_SEC = 0.35        # bu sürenin altı zaten embedding'e girmez
    QUALITY_FULL_SEC = 2.2        # bu süreden sonrası tam kalite
    QUALITY_THRESHOLD_PENALTY = 0.08   # düşük kalitede eşiğe eklenen max ceza

    # 3) OLGUNLUK: az konuşmuş (genç) bir profil, yeni sesi sahiplenmek için
    #    daha güçlü kanıt istemelidir. Hayalet konuşmacıların kendilerini
    #    beslemesini engeller; ses olgun/gerçek konuşmacıya gitmeye meyleder.
    MATURITY_FULL_SEC = 8.0            # bu kadar konuşmadan sonra tam olgun
    MATURITY_BASE = 0.45               # sıfır konuşmalı profilin olgunluğu
    # Ablasyon ölçümü (IS1009a): cezayı 0.0'a çekmek TÜM metrikleri kötüleştirdi
    # (cpWER 59.40 → 59.95, DER 40.96 → 41.22, confusion 15.29 → 15.50) ve
    # konuşmacı sayısı 9 → 10'a ÇIKTI. Yani olgunluk cezası konuşmacı şişmesinin
    # sebebi değil, küçük ama net bir katkı. Bu yüzden geri açıldı; eşik artık
    # konuşmacı-başına uygulanıyor (bkz. map_speakers) — genç top-1 elenince ses
    # olgun ikinciye düşüyor, aday tamponuna gitmiyor.
    MATURITY_THRESHOLD_PENALTY = 0.06  # genç profile eklenen max eşik cezası

    # --- Warm-up kapıları ----------------------------------------------- #
    # Kalibrasyonun bitmesi için İKİ koşul da gerekir: yeterli SES ve yeterli
    # EMBEDDING. Yalnız süreye bakmak, uzun ama az konuşmacılı chunk'larda
    # kümelemeyi 2-3 örnekle kurabiliyordu (10 sn'lik iki chunk = 20 sn ses,
    # 2 embedding). Üstelik _finalize_warmup'ın gürültü filtresi ancak n >= 6'da
    # devreye giriyor; altında her tekil embedding ayrı konuşmacı oluyor.
    WARMUP_MIN_EMBEDDINGS = 6
    # Tavan: embedding kapısı kalibrasyonu sonsuza kadar erteleyemesin. Bu süreye
    # ulaşılınca eldeki embedding sayısı ne olursa olsun kümeleme yapılır.
    WARMUP_MAX_AUDIO_FACTOR = 2.0

    # --- Kohort (AS-norm) normalizasyonu -------------------------------- #
    # Anlamlı bir impostor istatistiği için gereken min profil sayısı. 2 profille
    # "diğerleri" tek örnek demek; ortalaması gürültüden ibarettir.
    COHORT_MIN_PROFILES = 3
    # Referans EMA hızı. Referans, "normal koşulda tipik impostor benzerliği"dir;
    # sabit bir sayı yerine oturum içinde öğrenilir — mikrofon/oda/model değişince
    # tüm benzerlik dağılımı kayar ve sabit bir referans yanlış olurdu.
    COHORT_REFERENCE_MOMENTUM = 0.05

    def __init__(self, threshold=None, warmup_ms=None, cohort_norm=None, posterior=None):
        self.threshold = threshold if threshold is not None else DIARIZATION_EMBEDDING_THRESHOLD
        self.warmup_ms = warmup_ms if warmup_ms is not None else DIARIZATION_WARMUP_MS
        self.cohort_norm = cohort_norm if cohort_norm is not None else DIARIZATION_COHORT_NORM
        # Öğrenilen impostor referansı (ilk gözlemde tohumlanır).
        self._cohort_reference = None
        # Kalibre posterior ayarları (PosteriorSettings) ya da None.
        # None ise KARAR YOLU BİREBİR ESKİSİ GİBİ kalır — A/B tek anahtarla.
        self.posterior = posterior

        # Bilinen konuşmacılar (warm-up sonrası dolu olur)
        self.known_speakers = {}  # {global_label: centroid_tensor}
        self._reservoirs = {}     # {global_label: [embedding_tensor, ...]} (en yeni sonda)
        # Konuşmacı başına biriken temiz konuşma süresi (sn) — olgunluk ölçüsü.
        self._speech_seconds = {}
        self._next_id = 0

        # Warm-up state
        self._warmup_buffer = []  # list of embedding tensors
        # Her warm-up embedding'inin ÇAĞIRANA ait kaynağı (aynı sırada).
        # Kalibrasyon bitince "[Calibrating]" segmentlerini geriye dönük
        # etiketlemek için kullanılır.
        self._warmup_sources = []
        self._warmup_audio_ms = 0  # toplam işlenen ses süresi
        self._warmup_complete = False
        # {kaynak_anahtari: konusmaci_etiketi} — _finalize_warmup doldurur.
        self.warmup_assignments = {}

        # Yeni konuşmacı aday tamponu
        # Her aday: {"embeddings": [tensor, ...], "created_at": int}
        self._candidates = []
        self._chunk_counter = 0

        # --- Teşhis (embedding görünümü) --------------------------------- #
        # Kapalıyken maliyet SIFIR olmalı: skorlar zaten hesaplanıyor, ama
        # rezervuar/embedding kopyalamak bedava değil. Arayüz pencereyi açınca
        # açılır.
        self.debug_enabled = False
        self.last_trace = None
        # Warm-up izi her zaman tutulur (oturumda bir kez, küçük): pencere
        # sonradan açılsa da kalibrasyonun nasıl gittiği görülebilsin.
        self.last_warmup_trace = None

    def reset(self):
        """Tracker durumunu sıfırlayarak yeni bir dosya için hazır hale getirir."""
        self.known_speakers = {}
        self._reservoirs = {}
        self._speech_seconds = {}
        self._next_id = 0
        self._warmup_buffer = []
        self._warmup_sources = []
        self.warmup_assignments = {}
        self._warmup_audio_ms = 0
        self._warmup_complete = False
        self._candidates = []
        self._chunk_counter = 0
        self._cohort_reference = None
        self.last_trace = None
        self.last_warmup_trace = None

    # ------------------------------------------------------------------ #
    # Teşhis yardımcıları
    # ------------------------------------------------------------------ #
    # Warm-up izinde saklanacak max embedding sayısı (n×n matris büyümesin).
    WARMUP_TRACE_MAX_N = 200

    def _record_warmup_trace(self, sim_matrix, cluster_labels: dict, n: int) -> None:
        """Kalibrasyon sonucunu teşhis için saklar.

        `debug_enabled`'dan bağımsızdır: warm-up oturumda bir kez olur ve pencere
        sonradan açıldığında da "kaç embedding'le kalibre oldu" sorusu
        cevaplanabilmelidir. Bu soru, kalibrasyonun erken bitip bitmediğini
        gösteren tek doğrudan kanıttır.
        """
        matrix = None
        if sim_matrix is not None and n <= self.WARMUP_TRACE_MAX_N:
            matrix = sim_matrix.detach().cpu().numpy().copy()
        self.last_warmup_trace = {
            "embedding_count": n,
            "audio_ms": self._warmup_audio_ms,
            "similarity_matrix": matrix,
            "clusters": {label: list(idx) for label, idx in cluster_labels.items()},
            "threshold": self.threshold,
            "filtered": n - sum(len(idx) for idx in cluster_labels.values()),
        }

    def _speaker_snapshot(self) -> dict:
        """Konuşmacı profillerinin donmuş numpy görüntüsü (teşhis penceresi için)."""
        snapshot = {}
        for label, centroid in self.known_speakers.items():
            snapshot[label] = {
                "centroid": centroid.detach().cpu().numpy().copy(),
                "reservoir": [
                    e.detach().cpu().numpy().copy()
                    for e in self._reservoirs.get(label, [])
                ],
                "speech_seconds": float(self._speech_seconds.get(label, 0.0)),
            }
        return snapshot

    def _next_label(self):
        label = f"SPEAKER_{self._next_id:02d}"
        self._next_id += 1
        return label

    @property
    def is_warming_up(self):
        return not self._warmup_complete

    @property
    def warmup_remaining_ms(self) -> int:
        """Kalibrasyonun süre kapısı için kalan ms (0 olabilir, bkz. embedding kapısı).

        0 dönmesi warm-up'ın bittiği anlamına GELMEZ: süre dolmuş ama henüz
        WARMUP_MIN_EMBEDDINGS toplanmamış olabilir. Bitip bitmediği için
        `is_warming_up` kullanılmalı.
        """
        return max(0, self.warmup_ms - self._warmup_audio_ms)

    # ------------------------------------------------------------------ #
    # Rezervuar yardımcıları
    # ------------------------------------------------------------------ #
    def _register_speaker(self, embeddings: list, seed_seconds: float | None = None) -> str:
        """Yeni konuşmacı oluşturur: rezervuar + normalize centroid.

        seed_seconds: profilin başlangıç "olgunluk" kredisi. Verilmezse gözlem
        başına 1 sn sayılır — böylece warm-up'ta çok gözlemle kurulan profiller
        olgun, aday kapısından 2 gözlemle geçen yeni profiller GENÇ başlar
        (genç profil yeni sesi sahiplenmek için daha güçlü kanıt ister).
        """
        label = self._next_label()
        reservoir = [e.cpu() for e in embeddings[-self.RESERVOIR_SIZE:]]
        self._reservoirs[label] = reservoir
        self.known_speakers[label] = _normalized_mean(reservoir)
        if seed_seconds is None:
            seed_seconds = float(len(embeddings))
        self._speech_seconds[label] = max(0.0, seed_seconds)
        return label

    # ------------------------------------------------------------------ #
    # Karar kalitesi yardımcıları (margin / süre kalitesi / olgunluk)
    # ------------------------------------------------------------------ #
    def _duration_quality(self, duration: float) -> float:
        """Süreden [0.25, 1.0] aralığında kalite katsayısı.

        Süre bilinmiyorsa (None/inf) cezalandırmadan 1.0 döner.
        """
        if duration is None or duration == float("inf"):
            return 1.0
        span = self.QUALITY_FULL_SEC - self.QUALITY_MIN_SEC
        if span <= 0:
            return 1.0
        return max(0.25, min(1.0, (duration - self.QUALITY_MIN_SEC) / span))

    def _maturity(self, label: str) -> float:
        """Profilin olgunluğu [MATURITY_BASE, 1.0] — biriken konuşma süresine göre."""
        if not label:
            return 1.0
        seconds = self._speech_seconds.get(label, 0.0)
        grown = (1.0 - self.MATURITY_BASE) * (seconds / self.MATURITY_FULL_SEC)
        return min(1.0, self.MATURITY_BASE + grown)

    def _effective_threshold(self, quality: float, maturity: float) -> float:
        """Kısa ses ve genç profil için eşiği geçici olarak yükseltir."""
        return (
            self.threshold
            + (1.0 - quality) * self.QUALITY_THRESHOLD_PENALTY
            + (1.0 - maturity) * self.MATURITY_THRESHOLD_PENALTY
        )

    def _accumulate_speech(self, label: str, duration: float) -> None:
        """Atanan sesin süresini profilin olgunluk hanesine yazar."""
        if not label or label == "Unknown":
            return
        if duration is None or duration == float("inf"):
            return
        self._speech_seconds[label] = self._speech_seconds.get(label, 0.0) + max(0.0, duration)

    def _add_observation(self, label: str, emb: torch.Tensor) -> None:
        """Güçlü eşleşen embedding'i rezervuara ekler, centroid'i türetir."""
        reservoir = self._reservoirs.get(label)
        if reservoir is None:
            # Subclass/dış kod konuşmacıyı doğrudan known_speakers'a yazmış
            # olabilir — mevcut centroid'i tohum olarak koru.
            existing = self.known_speakers.get(label)
            reservoir = [existing.clone()] if existing is not None else []
            self._reservoirs[label] = reservoir
            # Dış kod yazdıysa olgunluk kaydı da yoktur — nötr başlat.
            self._speech_seconds.setdefault(label, 0.0)
        reservoir.append(emb.clone())
        if len(reservoir) > self.RESERVOIR_SIZE:
            reservoir.pop(0)
        self.known_speakers[label] = _normalized_mean(reservoir)

    def _similarity(self, emb: torch.Tensor, label: str) -> float:
        """Konuşmacı skoru: rezervuar+centroid benzerliklerinin en iyi ikisinin
        ortalaması. Tek centroid'e göre konuşmacı-içi varyasyona çok daha
        dayanıklı; tek örnekli konuşmacıda centroid-only'ye düşer.
        """
        candidates = list(self._reservoirs.get(label, []))
        centroid = self.known_speakers.get(label)
        if centroid is not None:
            candidates.append(centroid)
        if not candidates:
            return -1.0
        sims = sorted((_cosine(emb, c) for c in candidates), reverse=True)
        if len(sims) >= 2:
            return (sims[0] + sims[1]) / 2.0
        return sims[0]

    # ------------------------------------------------------------------ #
    # Kohort (AS-norm) normalizasyonu
    # ------------------------------------------------------------------ #
    def _cohort_normalize(self, raw_scores: dict) -> dict:
        """Utterance-seviyesi skor kaymasını düzeltir.

        SORUN: ham kosinüs benzerliği utterance'lar arasında kıyaslanabilir
        değildir. Kısa/gürültülü bir embedding TÜM profillere yakın çıkar
        (hepsine ~0.6), temiz bir embedding hepsinden uzak durur (~0.3 + doğru
        olanına 0.8). Tek bir global eşik bu ikisi için birden doğru olamaz;
        birincide herkes eşiği geçer, ikincide kimse geçemez.

        DÜZELTME: her skordan, o profilin gördüğü "impostor" ortalamasının
        referanstan sapması çıkarılır:

            s'_i = s_i - alpha * (ortalama_{j != i}(s_j) - referans)

        Böylece soru "s_i eşiği geçti mi" olmaktan çıkıp "s_i kendi kohortuna
        göre ne kadar sıra dışı" haline gelir. Dönüşüm s_i'de monotondur —
        SIRALAMA DEĞİŞMEZ, yalnızca eşiğe göre konum ve margin ölçeklenir. Bu
        bilinçli: sıralama en-yakın-komşu mantığının kalbi, ona dokunmuyoruz.

        Referans sabit bir sayı değil, oturum içinde öğrenilen bir EMA'dır:
        mikrofon/oda/model değişince tüm benzerlik dağılımı kayar.

        YAN ETKİ (bilinçli): dönüşüm s_i'de doğrusal olduğundan skorlar arası
        mesafe (1 + alpha/(K-1)) katına çıkar. Yani MIN_DECISION_MARGIN kapısı
        normalizasyon AÇIKKEN daha az ateşler; margin eşiği bu ölçekte yeniden
        okunmalıdır (AMI taramasının cevaplaması gereken sorulardan biri).

        alpha = 0 (varsayılan) → ham skorlar aynen döner.
        """
        if self.cohort_norm <= 0.0 or len(raw_scores) < self.COHORT_MIN_PROFILES:
            return raw_scores

        labels = list(raw_scores)
        total = sum(raw_scores.values())
        count = len(labels)

        # Referans için impostor istatistiği: EN İYİ eşleşme hariç ortalama.
        # (En iyi eşleşme muhtemelen gerçek konuşmacıdır; onu impostor
        #  havuzuna katmak referansı yukarı çeker ve düzeltmeyi zayıflatır.)
        best_score = max(raw_scores.values())
        impostor_mean = (total - best_score) / (count - 1)

        if self._cohort_reference is None:
            # İlk gözlem referansı TOHUMLAR → bu chunk düzeltilmeden geçer.
            self._cohort_reference = impostor_mean
        reference = self._cohort_reference

        normalized = {}
        for label in labels:
            cohort_mean = (total - raw_scores[label]) / (count - 1)
            normalized[label] = raw_scores[label] - self.cohort_norm * (cohort_mean - reference)

        # Referansı gözlemden sonra güncelle (bu chunk kendi düzeltmesini etkilemesin).
        momentum = self.COHORT_REFERENCE_MOMENTUM
        self._cohort_reference = (1.0 - momentum) * reference + momentum * impostor_mean
        return normalized

    # ------------------------------------------------------------------ #
    # Warm-up
    # ------------------------------------------------------------------ #
    def add_warmup_chunk(self, embeddings, chunk_duration_ms, sources=None):
        """Warm-up fazında BİR CHUNK'ın embedding'lerini toplar.

        API bilerek chunk seviyesindedir. Önceki embedding-seviyesi sürüm
        (`add_warmup_embedding`) çağıranın her embedding için ayrı çağırmasını
        gerektiriyordu ve chunk süresi HER çağrıda sayıldığı için N konuşmacılı
        chunk'ta süre N katına çıkıyordu: "20 sn kalibrasyon" 4 konuşmacılı bir
        toplantıda ~5 sn'de bitiyordu. Yani kalibrasyon, en çok veriye ihtiyaç
        duyulan durumda (kalabalık toplantı) en az veriyle kapanıyordu — ve
        warm-up kümelemesi tüm oturumun konuşmacı kümesini belirlediği için hata
        oturum boyunca yayılıyordu. Chunk seviyesi imza bu yanlışı imkânsız kılar.

        Embedding üretmeyen chunk kalibrasyona katkı yapmadığı için süresi de
        sayılmaz — aksi halde süre kapısı boş chunk'larla dolup kümeleme
        birkaç örnekle kurulabilirdi.

        Args:
            embeddings: bu chunk'tan çıkan embedding tensörleri (iterable).
            chunk_duration_ms: chunk'ın süresi — TOPLAMA BİR KEZ eklenir.
            sources: embedding'lerle AYNI SIRADA, çağırana ait opak anahtarlar
                (örn. (segment_index, local_label)). Verilirse kalibrasyon
                bitince `warmup_assignments` bu anahtarları oluşan konuşmacı
                etiketlerine eşler — böylece warm-up sırasında "[Calibrating]"
                diye geçilen segmentler GERİYE DÖNÜK olarak etiketlenebilir.

        Returns:
            bool: True ise warm-up bu çağrıda bitti (baseline hazır).
        """
        collected = [emb for emb in embeddings if emb is not None]
        if not collected:
            return False

        keys = list(sources) if sources is not None else []
        for index, embedding in enumerate(collected):
            self._warmup_buffer.append(embedding.cpu())
            self._warmup_sources.append(keys[index] if index < len(keys) else None)
        self._warmup_audio_ms += chunk_duration_ms

        # Kapı 1: yeterli ses.
        if self._warmup_audio_ms < self.warmup_ms:
            return False
        # Kapı 2: yeterli embedding — tavana ulaşılmadıysa beklenir.
        if (len(self._warmup_buffer) < self.WARMUP_MIN_EMBEDDINGS
                and self._warmup_audio_ms < self.warmup_ms * self.WARMUP_MAX_AUDIO_FACTOR):
            return False

        self._finalize_warmup()
        return True

    # --- Warm-up gürültü filtresi -------------------------------------- #
    # Tek üyeli küme "gürültü" sayılmadan önce, konuşmacı başına BİRDEN ÇOK
    # gözlem beklenebilecek kadar veri olmalı. 4-6 konuşmacılı bir toplantıda
    # 12 embedding ≈ kişi başına 2-3 gözlem demektir; ancak o zaman yalnız
    # kalmış bir embedding gerçekten sıra dışıdır.
    WARMUP_SINGLETON_FILTER_MIN_EMBEDDINGS = 12

    def _filter_noise_clusters(self, clusters: dict, n: int) -> dict:
        """Gürültü kümelerini eler — ama YALNIZCA elemenin anlamlı olduğu yerde.

        ÖLÇÜLDÜ (AMI, kaydedilmiş warm-up izleri):

            IS1009a  6 embedding -> kümeler [3,1,1,1]   gerçek 4 konuşmacı
            ES2004a  8 embedding -> [2,2,1,1,1,1]       gerçek 4 konuşmacı
            TS3003a  7 embedding -> [2,2,1,1,1]         gerçek 4 konuşmacı

        Küme-içi benzerlik 0.62-0.74, küme-arası 0.16-0.33 — yani ayrım TEMİZ,
        kümeleme doğru çalışıyor. Tek üyeli kümeler gürültü DEĞİL: warm-up
        penceresinde yalnızca bir kez konuşmuş GERÇEK insanlar. Onları elemek
        insanı elemek olur; eski kural (`n >= 6` ise tek üyelileri at) bu üç
        toplantıyı 4 konuşmacıdan 1-2'ye çöktürüyordu.

        İki koşul birden aranır:
          1. Yeterli veri — kişi başına birden çok gözlem beklenebilmeli.
          2. Tek üyeliler AZINLIK olmalı. Kümelerin çoğu tek üyeliyse bu,
             gürültü değil yetersiz örnekleme demektir.

        Ayrıca eleme gereksizdir: `_register_speaker` tek gözlemli profile zaten
        düşük olgunluk kredisi verir (gözlem başına 1 sn), yani o profil yeni
        sesi sahiplenmek için daha güçlü kanıt ister. "Az kanıtlı profile daha
        az güven" mekanizması zaten var.
        """
        singletons = [key for key, members in clusters.items() if len(members) < 2]
        if not singletons:
            return dict(clusters)

        enough_data = n >= self.WARMUP_SINGLETON_FILTER_MIN_EMBEDDINGS
        singletons_are_minority = len(singletons) * 2 <= len(clusters)
        if not (enough_data and singletons_are_minority):
            return dict(clusters)

        valid = {key: members for key, members in clusters.items() if len(members) >= 2}
        if not valid:  # her ihtimale karşı: hepsi elendiyse en büyüğü kurtar
            largest = max(clusters.items(), key=lambda item: len(item[1]))
            return {largest[0]: largest[1]}
        return valid

    def _finalize_warmup(self):
        """
        İki-aşamalı warm-up clustering:
        1. Pairwise similarity matrix ile agglomerative clustering
        2. Gürültü kümelerini filtrele (yalnızca yeterli veri varken — bkz.
           _filter_noise_clusters)
        """
        if not self._warmup_buffer:
            self._warmup_complete = True
            return

        n = len(self._warmup_buffer)
        print(f"\n[Warm-up] Clustering {n} embeddings...")

        if n == 1:
            # Tek embedding varsa direkt konuşmacı oluştur
            label = self._register_speaker([self._warmup_buffer[0]])
            self._record_warmup_trace(None, {label: [0]}, n)
            self._assign_warmup_sources({label: [0]})
            self._warmup_complete = True
            self._warmup_buffer = []
            self._warmup_sources = []
            print(f"[Warm-up Complete] 1 speaker detected: {label}")
            print(f"   ({self._warmup_audio_ms / 1000:.1f}s audio)\n")
            return

        # Pairwise similarity matrix
        emb_stack = torch.stack(self._warmup_buffer)  # (n, dim)
        sim_matrix = torch.nn.functional.cosine_similarity(
            emb_stack.unsqueeze(0), emb_stack.unsqueeze(1), dim=2
        )  # (n, n)

        # Agglomerative clustering — her embedding kendi kümesi olarak başlar
        clusters = {i: [i] for i in range(n)}

        # Merge: en yüksek similarity'den başla
        while True:
            best_i, best_j, best_sim = -1, -1, -1.0

            active_clusters = list(clusters.keys())
            for ci_idx in range(len(active_clusters)):
                for cj_idx in range(ci_idx + 1, len(active_clusters)):
                    ci = active_clusters[ci_idx]
                    cj = active_clusters[cj_idx]

                    # Average linkage: kümelerdeki tüm çiftlerin ortalama similarity'si
                    total_sim = 0.0
                    count = 0
                    for mi in clusters[ci]:
                        for mj in clusters[cj]:
                            total_sim += sim_matrix[mi, mj].item()
                            count += 1
                    avg_sim = total_sim / count if count > 0 else 0.0

                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_i = ci
                        best_j = cj

            # Threshold'un altındaysa dur
            if best_sim < self.threshold or best_i < 0:
                break

            # Merge clusters
            clusters[best_i].extend(clusters[best_j])
            del clusters[best_j]

        # Tek üyeli kümeleri "gürültü" diye eleme kararı.
        valid_clusters = self._filter_noise_clusters(clusters, n)

        # Her kümeden konuşmacı oluştur (rezervuar = küme üyeleri)
        cluster_labels = {}
        for member_indices in valid_clusters.values():
            member_embs = [self._warmup_buffer[i] for i in member_indices]
            cluster_labels[self._register_speaker(member_embs)] = list(member_indices)

        # Kalibrasyonun ne kadar sağlam olduğunu görünür kıl: kaç embedding'le
        # bitti, hangi kümeler oluştu, çiftler arası benzerlik neydi.
        self._record_warmup_trace(sim_matrix, cluster_labels, n)
        # Geriye dönük etiketleme haritası: hangi warm-up embedding'i hangi
        # konuşmacıya gitti. Elenen kümelerin kaynakları haritaya GİRMEZ.
        self._assign_warmup_sources(cluster_labels)

        self._warmup_complete = True

        # Filtrelenen embedding sayısı
        total_used = sum(len(v) for v in valid_clusters.values())
        filtered_count = n - total_used

        speaker_list = ", ".join(self.known_speakers.keys())
        print(f"[Warm-up Complete] {len(self.known_speakers)} speaker(s) detected: {speaker_list}")
        if filtered_count > 0:
            print(f"   (filtered {filtered_count} noisy embedding(s))")
        print(f"   ({self._warmup_audio_ms / 1000:.1f}s audio processed)\n")

        self._warmup_buffer = []

    # ------------------------------------------------------------------ #
    # Aday tamponu
    # ------------------------------------------------------------------ #
    def _find_matching_candidate(self, emb):
        """
        Aday tamponunda bu embedding'e benzer bir aday var mı?
        Varsa adayın index'ini döndürür, yoksa -1.
        """
        for idx, cand in enumerate(self._candidates):
            centroid = torch.stack(cand["embeddings"]).mean(dim=0)
            if _cosine(emb, centroid) >= self.CANDIDATE_SELF_SIMILARITY:
                return idx
        return -1

    def _try_promote_candidate(self, candidate):
        """
        Aday yeterli onay aldıysa ve embedding'ler kendi aralarında
        tutarlıysa gerçek konuşmacıya yükseltir.

        Returns:
            str veya None: Yeni konuşmacı etiketi, veya None
        """
        if len(candidate["embeddings"]) < self.CANDIDATE_CONFIRMATIONS_NEEDED:
            return None

        embs = candidate["embeddings"]

        # Kendi aralarında tutarlılık kontrolü
        if len(embs) >= 2:
            pair_scores = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    pair_scores.append(_cosine(embs[i], embs[j]))
            avg_self_sim = sum(pair_scores) / len(pair_scores)
            if avg_self_sim < self.CANDIDATE_SELF_SIMILARITY:
                return None

        # Onaylandı — yeni konuşmacı oluştur (aday gözlemleri rezervuarı tohumlar)
        return self._register_speaker(embs)

    def _expire_old_candidates(self):
        """TTL süresi dolan adayları temizler."""
        self._candidates = [
            c for c in self._candidates
            if (self._chunk_counter - c["created_at"]) < self.CANDIDATE_TTL
        ]

    def _merge_similar_speakers(self, merge_threshold=0.85):
        """
        Birbirine çok benzeyen bilinen konuşmacıları birleştirir.

        Warm-up clustering (veya canlı faz) aynı kişiyi yanlışlıkla iki ayrı
        konuşmacıya bölmüş olabilir; üstelik warm-up'ta belirlenen sayı kalıcı.
        Bu güvenlik ağı, centroid'leri merge_threshold'u aşan konuşmacıları
        tek etikete indirir (düşük id'li korunur), böylece şişen sayı zamanla
        kendiliğinden düzelir. Rezervuarlar da birleştirilir.

        Returns:
            dict: {silinen_etiket: korunan_etiket} — çağıran, mevcut chunk'ın
                  eşlemesini de bu remap'le güncelleyebilir.
        """
        remap = {}
        labels = sorted(self.known_speakers.keys())  # SPEAKER_00 < SPEAKER_01 ...
        i = 0
        while i < len(labels):
            keep = labels[i]
            j = i + 1
            while j < len(labels):
                drop = labels[j]
                if keep in self.known_speakers and drop in self.known_speakers:
                    sim = _cosine(self.known_speakers[keep], self.known_speakers[drop])
                    if sim >= merge_threshold:
                        merged_reservoir = (
                            self._reservoirs.get(keep, [self.known_speakers[keep]])
                            + self._reservoirs.get(drop, [self.known_speakers[drop]])
                        )[-self.RESERVOIR_SIZE:]
                        self._reservoirs[keep] = merged_reservoir
                        self.known_speakers[keep] = _normalized_mean(merged_reservoir)
                        del self.known_speakers[drop]
                        self._reservoirs.pop(drop, None)
                        # Olgunluk da birleşir: iki profil aynı kişiyse konuşma
                        # süreleri de aynı kişiye aittir.
                        self._speech_seconds[keep] = (
                            self._speech_seconds.get(keep, 0.0)
                            + self._speech_seconds.pop(drop, 0.0)
                        )
                        remap[drop] = keep
                        labels.pop(j)
                        print(f"  [Merge] {drop} -> {keep} (sim: {sim:.3f})")
                        continue
                j += 1
            i += 1
        return remap

    # ------------------------------------------------------------------ #
    # Aktif faz eşleme
    # ------------------------------------------------------------------ #
    def map_speakers(self, embeddings_dict, quality_dict=None):
        """
        Embedding'lere göre konuşmacıları eşler (warm-up sonrası).
        Güçlü eşleşmeler konuşmacı rezervuarına eklenir.
        Bilinmeyen sesler için onay tamponu kullanılır.

        Args:
            embeddings_dict: {local_label: embedding_tensor}
            quality_dict: {local_label: temiz_konuşma_süresi_sn} (opsiyonel).
                Yalnızca süresi MIN_NEW_SPEAKER_DURATION'ı aşan "güvenilir"
                embedding'ler yeni konuşmacı (aday) oluşturabilir. Kısa sesler
                ("evet", "aynen öyle") en yakın mevcut konuşmacıya yapışır,
                aday tamponuna hiç girmez.

        Returns:
            dict: {local_label: global_label}
        """
        quality_dict = quality_dict or {}
        self._chunk_counter += 1
        self._expire_old_candidates()
        mapping = {}
        probes = []

        for local_label, emb in embeddings_dict.items():
            emb = emb.cpu()
            duration = quality_dict.get(local_label, float("inf"))
            is_reliable = duration >= MIN_NEW_SPEAKER_DURATION

            # Bilinen konuşmacıları skorla ve SIRALA — yalnız en iyi skor değil,
            # en iyi ile ikinci arasındaki MARGIN de karara giriyor.
            # Kohort normalizasyonu (kapalıysa ham skorlar aynen geçer) skorları
            # utterance'lar arası kıyaslanabilir hale getirir; sıralamayı değiştirmez.
            raw_scores = {label: self._similarity(emb, label) for label in self.known_speakers}
            scores = self._cohort_normalize(raw_scores)
            ranked = sorted(
                ((score, label) for label, score in scores.items()),
                key=lambda item: item[0],
                reverse=True,
            )
            # En yakın konuşmacı (ham skor) — sticky/raporlama için referans.
            if ranked:
                best_score, best_match = ranked[0]
            else:
                best_score, best_match = -1.0, None
            # Margin HAM benzerlikler üzerinden: "iki konuşmacı akustik olarak
            # neredeyse berabere mi?" sorusu eşiklerden bağımsızdır.
            # Tek konuşmacı varsa rekabet yok → margin sonsuz kabul edilir.
            margin = (best_score - ranked[1][0]) if len(ranked) > 1 else float("inf")
            has_margin = margin >= self.MIN_DECISION_MARGIN

            quality = self._duration_quality(duration)
            maturity = self._maturity(best_match) if best_match else 1.0
            effective_threshold = self._effective_threshold(quality, maturity)

            # SADECE en iyi eşleşme (top-1) eşiğe karşı sınanır.
            #
            # ÖLÇÜLMÜŞ UYARI — burada sıralı listede aşağı inip "kendi eşiğini
            # geçen İLK konuşmacıyı" almak DENENDİ ve diarization'ı ciddi biçimde
            # bozdu (IS1009a: confusion 15.29 → 23.23, cpWER 59.40 → 69.12).
            # Sebep: 10 konuşmacı varken yeni/belirsiz bir ses, listenin
            # aşağısında eşiği geçen BİR profil neredeyse her zaman bulur; böylece
            # "eşleşmedi → ertele" davranışı "tutana kadar ara" davranışına
            # dönüşüp yanlış atama üretir. En-yakın-komşu mantığı gereği karar
            # DAİMA en iyi adaya göre verilmeli; en iyi aday yeterli değilse
            # daha kötü bir aday hiç değildir.
            passes_threshold = best_match is not None and best_score >= effective_threshold

            # --- Karar kapıları -------------------------------------------- #
            # Posterior KAPALIYKEN (varsayılan) üç kapı da eskisiyle birebir
            # aynıdır; A/B tek anahtarla yapılabilsin diye tek yerde toplandı.
            posterior_result = None
            if self.posterior is not None:
                posterior_result = speaker_posterior(
                    scores, self.posterior.config, duration=duration
                )
                policy = self.posterior.policy
                accept = policy.is_confident(posterior_result)
                should_learn = policy.may_learn(posterior_result) and quality >= 0.35
                # Yeni konuşmacı için "hiçbiri değil" kütlesi GEREKLİ. Ölçüldü:
                # candidate_promoted kararlarının çoğu referansta karşılık
                # bulmuyor — hayalet üretiminin kapısı burası.
                can_create = policy.may_create_speaker(posterior_result) and is_reliable
            else:
                accept = passes_threshold
                should_learn = (has_margin
                                and best_score > self.RESERVOIR_ADD_THRESHOLD
                                and quality >= 0.35)
                can_create = is_reliable

            # Teşhis: hangi kural ateşledi + profil güncellendi mi.
            decision = "unknown"
            reservoir_updated = False

            if accept:
                # Etiket her iki durumda da verilir (en iyi tahmin), ama profil
                # yalnızca KARARLI eşleşmede güncellenir.
                mapping[local_label] = best_match
                self._accumulate_speech(best_match, duration)
                decision = ("matched" if (posterior_result is not None or has_margin)
                            else "matched_ambiguous")

                # Drift'i önle: rezervuara YALNIZCA yüksek güvenli, kararlı ve
                # yeterli kaliteli eşleşmeler eklenir. Borderline ya da iki
                # konuşmacı arasında kararsız sesler rezervuarı kirletip baskın
                # konuşmacının herkese benzemesine yol açabilir.
                if should_learn:
                    self._add_observation(best_match, emb)
                    reservoir_updated = True
                elif posterior_result is None and not has_margin:
                    runner_up = ranked[1][1]
                    print(f"  [Ambiguous] {best_match} vs {runner_up} "
                          f"(margin: {margin:.3f} < {self.MIN_DECISION_MARGIN}) "
                          f"— etiket verildi, profil güncellenmedi")

            elif not can_create:
                # Eşleşme güvenli değil ve yeni konuşmacı açmaya da yetmiyor.
                # Etiket yine en yakın konuşmacıdır (ekranda bir şey görünmeli).
                mapping[local_label] = best_match if best_match else "Unknown"
                if posterior_result is not None:
                    # POSTERIOR YOLU FARKI: düşük güvenli atamanın süresi profile
                    # YAZILMAZ. Ölçüldü: bu kararların ~%46'sı yanlış; bugün
                    # süreleri yine de yanlış profilin olgunluk hanesini besliyor.
                    # Ekrana çıkan etiket değişmez (DER riski yok), profiller
                    # zamanla temizlenir.
                    decision = "low_confidence" if best_match else "unknown"
                else:
                    self._accumulate_speech(mapping[local_label], duration)
                    decision = "sticky_short" if best_match else "unknown"
                    if best_match:
                        print(f"  [Short utterance] sticky -> {best_match} "
                              f"(score: {best_score:.3f}, {duration:.1f}s)")

            else:
                # Bilinmeyen VE güvenilir (yeterince uzun) ses
                # — aday tamponuna ekle veya mevcut adayı onayla.
                cand_idx = self._find_matching_candidate(emb)

                if cand_idx >= 0:
                    # Mevcut adaya yeni gözlem ekle
                    self._candidates[cand_idx]["embeddings"].append(emb.clone())
                    promoted_label = self._try_promote_candidate(self._candidates[cand_idx])

                    if promoted_label:
                        # Aday onaylandı, gerçek konuşmacı oldu
                        mapping[local_label] = promoted_label
                        self._accumulate_speech(promoted_label, duration)
                        decision = "candidate_promoted"
                        closest_info = f" (closest: {best_match}, score: {best_score:.3f})" if best_match else ""
                        confirms = len(self._candidates[cand_idx]["embeddings"])
                        # ASCII-only: emoji cp1254/cp1252 Windows konsollarında
                        # UnicodeEncodeError ile canlı akışı düşürüyordu.
                        print(f"  [New speaker] {promoted_label}{closest_info} [confirmed after {confirms} observations]")
                        self._candidates.pop(cand_idx)
                    else:
                        # Henüz yeterli onay yok — en yakın bilinen konuşmacıya ata
                        mapping[local_label] = best_match if best_match else "Unknown"
                        decision = "candidate_pending"
                        confirms = len(self._candidates[cand_idx]["embeddings"])
                        needed = self.CANDIDATE_CONFIRMATIONS_NEEDED
                        print(f"  [Candidate] pending ({confirms}/{needed}), mapped to {mapping[local_label]}")
                else:
                    # Yeni aday oluştur
                    self._candidates.append({
                        "embeddings": [emb.clone()],
                        "created_at": self._chunk_counter,
                    })
                    mapping[local_label] = best_match if best_match else "Unknown"
                    decision = "candidate_new"
                    closest_info = f" (closest: {best_match}, score: {best_score:.3f})" if best_match else ""
                    print(f"  [Candidate] new candidate registered{closest_info}, mapped to {mapping[local_label]}")

            if self.debug_enabled:
                probes.append({
                    "local_label": local_label,
                    "embedding": emb.detach().numpy().copy(),
                    "duration": None if duration == float("inf") else float(duration),
                    "quality": float(quality),
                    "maturity": float(maturity),
                    "effective_threshold": float(effective_threshold),
                    "scores": {label: float(score) for score, label in ranked},
                    # Ham (normalizasyon öncesi) skorlar — düzeltmenin etkisi
                    # teşhis penceresinde/analizde görülebilsin.
                    "raw_scores": {label: float(v) for label, v in raw_scores.items()},
                    "cohort_reference": (None if self._cohort_reference is None
                                         else float(self._cohort_reference)),
                    "best": best_match,
                    "best_score": float(best_score),
                    # inf (tek konuşmacı → rekabet yok) JSON/arayüz için None.
                    "margin": None if margin == float("inf") else float(margin),
                    "has_margin": bool(has_margin),
                    "passed_threshold": bool(passes_threshold),
                    "reliable_duration": bool(is_reliable),
                    "decision": decision,
                    "reservoir_updated": reservoir_updated,
                    # Posterior açıksa güven değerleri de ize girer (analiz ve
                    # teşhis penceresi bunları okur).
                    "p_best": (None if posterior_result is None
                               else float(posterior_result.best_probability)),
                    "p_unknown": (None if posterior_result is None
                                  else float(posterior_result.unknown_probability)),
                })

        # Şişen konuşmacı sayısını düzelt: birbirine çok benzeyen bilinenleri
        # birleştir ve bu chunk'ın eşlemesini de remap'le güncelle.
        remap = self._merge_similar_speakers()
        if remap:
            for local_label, glabel in mapping.items():
                if glabel in remap:
                    mapping[local_label] = remap[glabel]

        if self.debug_enabled:
            # 'assigned' NİHAİ eşlemeden okunur: merge remap'i kararı değiştirmiş
            # olabilir.
            for probe in probes:
                probe["assigned"] = mapping.get(probe["local_label"])
            self.last_trace = {
                "chunk_index": self._chunk_counter,
                "probes": probes,
                "speakers": self._speaker_snapshot(),
                "merged": dict(remap),
                "candidate_count": len(self._candidates),
                "merge_threshold": 0.85,
                "reservoir_add_threshold": self.RESERVOIR_ADD_THRESHOLD,
                "min_decision_margin": self.MIN_DECISION_MARGIN,
            }

        return mapping

    def map_speakers_fallback(self, local_labels):
        """Embedding çıkarılamadığında (çok kısa/sessiz chunk) fallback.

        Eski davranış her local label'a YENİ bir global etiket uyduruyordu;
        embedding'i olmayan bu 'hayalet' konuşmacılar hiçbir zaman eşlenemediği
        için konuşmacı sayısını şişiriyordu (CHiME-6 over-count'un ana kaynağı).
        Artık kimliklendirilemeyen ses dürüstçe 'Unknown' etiketlenir.
        """
        return {label: "Unknown" for label in local_labels}
