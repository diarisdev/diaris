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
    MIN_NEW_SPEAKER_DURATION,
)


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

    def __init__(self, threshold=None, warmup_ms=None):
        self.threshold = threshold if threshold is not None else DIARIZATION_EMBEDDING_THRESHOLD
        self.warmup_ms = warmup_ms if warmup_ms is not None else DIARIZATION_WARMUP_MS

        # Bilinen konuşmacılar (warm-up sonrası dolu olur)
        self.known_speakers = {}  # {global_label: centroid_tensor}
        self._reservoirs = {}     # {global_label: [embedding_tensor, ...]} (en yeni sonda)
        # Konuşmacı başına biriken temiz konuşma süresi (sn) — olgunluk ölçüsü.
        self._speech_seconds = {}
        self._next_id = 0

        # Warm-up state
        self._warmup_buffer = []  # list of embedding tensors
        self._warmup_audio_ms = 0  # toplam işlenen ses süresi
        self._warmup_complete = False

        # Yeni konuşmacı aday tamponu
        # Her aday: {"embeddings": [tensor, ...], "created_at": int}
        self._candidates = []
        self._chunk_counter = 0

    def reset(self):
        """Tracker durumunu sıfırlayarak yeni bir dosya için hazır hale getirir."""
        self.known_speakers = {}
        self._reservoirs = {}
        self._speech_seconds = {}
        self._next_id = 0
        self._warmup_buffer = []
        self._warmup_audio_ms = 0
        self._warmup_complete = False
        self._candidates = []
        self._chunk_counter = 0

    def _next_label(self):
        label = f"SPEAKER_{self._next_id:02d}"
        self._next_id += 1
        return label

    @property
    def is_warming_up(self):
        return not self._warmup_complete

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
    # Warm-up
    # ------------------------------------------------------------------ #
    def add_warmup_embedding(self, embedding, chunk_duration_ms):
        """
        Warm-up fazında embedding toplar.
        Yeterli ses birikince warm-up'ı sonlandırır.

        Returns:
            bool: True ise warm-up bitti (baseline hazır)
        """
        self._warmup_buffer.append(embedding.cpu())
        self._warmup_audio_ms += chunk_duration_ms

        if self._warmup_audio_ms >= self.warmup_ms:
            self._finalize_warmup()
            return True
        return False

    def _finalize_warmup(self):
        """
        İki-aşamalı warm-up clustering:
        1. Pairwise similarity matrix ile agglomerative clustering
        2. Küçük kümeleri (< 2 embedding) filtrele (gürültü)
        """
        if not self._warmup_buffer:
            self._warmup_complete = True
            return

        n = len(self._warmup_buffer)
        print(f"\n[Warm-up] Clustering {n} embeddings...")

        if n == 1:
            # Tek embedding varsa direkt konuşmacı oluştur
            label = self._register_speaker([self._warmup_buffer[0]])
            self._warmup_complete = True
            self._warmup_buffer = []
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

        # Küçük kümeleri filtrele (gürültü olma ihtimali yüksek)
        min_cluster_size = 2 if n >= 6 else 1
        valid_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

        # Eğer filtreleme sonrası hiçbir küme kalmadıysa, en büyük kümeyi al
        if not valid_clusters:
            largest = max(clusters.items(), key=lambda x: len(x[1]))
            valid_clusters = {largest[0]: largest[1]}

        # Her kümeden konuşmacı oluştur (rezervuar = küme üyeleri)
        for member_indices in valid_clusters.values():
            member_embs = [self._warmup_buffer[i] for i in member_indices]
            self._register_speaker(member_embs)

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

        for local_label, emb in embeddings_dict.items():
            emb = emb.cpu()
            duration = quality_dict.get(local_label, float("inf"))
            is_reliable = duration >= MIN_NEW_SPEAKER_DURATION

            # Bilinen konuşmacıları skorla ve SIRALA — yalnız en iyi skor değil,
            # en iyi ile ikinci arasındaki MARGIN de karara giriyor.
            ranked = sorted(
                ((self._similarity(emb, label), label) for label in self.known_speakers),
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

            if passes_threshold:
                # Etiket her iki durumda da verilir (en iyi tahmin), ama profil
                # yalnızca KARARLI eşleşmede güncellenir.
                mapping[local_label] = best_match
                self._accumulate_speech(best_match, duration)

                # Drift'i önle: rezervuara YALNIZCA yüksek güvenli, kararlı ve
                # yeterli kaliteli eşleşmeler eklenir. Borderline ya da iki
                # konuşmacı arasında kararsız sesler rezervuarı kirletip baskın
                # konuşmacının herkese benzemesine yol açabilir.
                if has_margin and best_score > self.RESERVOIR_ADD_THRESHOLD and quality >= 0.35:
                    self._add_observation(best_match, emb)
                elif not has_margin:
                    runner_up = ranked[1][1]
                    print(f"  [Ambiguous] {best_match} vs {runner_up} "
                          f"(margin: {margin:.3f} < {self.MIN_DECISION_MARGIN}) "
                          f"— etiket verildi, profil güncellenmedi")

            elif not is_reliable:
                # Bilinmeyen AMA kısa/güvenilmez ses → yeni konuşmacı YARATMA.
                # En yakın mevcut konuşmacıya yapıştır (sticky). Bu, kısa
                # cümlelerin ("evet") yeni konuşmacı doğurmasını engeller.
                mapping[local_label] = best_match if best_match else "Unknown"
                self._accumulate_speech(mapping[local_label], duration)
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
                        closest_info = f" (closest: {best_match}, score: {best_score:.3f})" if best_match else ""
                        confirms = len(self._candidates[cand_idx]["embeddings"])
                        # ASCII-only: emoji cp1254/cp1252 Windows konsollarında
                        # UnicodeEncodeError ile canlı akışı düşürüyordu.
                        print(f"  [New speaker] {promoted_label}{closest_info} [confirmed after {confirms} observations]")
                        self._candidates.pop(cand_idx)
                    else:
                        # Henüz yeterli onay yok — en yakın bilinen konuşmacıya ata
                        mapping[local_label] = best_match if best_match else "Unknown"
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
                    closest_info = f" (closest: {best_match}, score: {best_score:.3f})" if best_match else ""
                    print(f"  [Candidate] new candidate registered{closest_info}, mapped to {mapping[local_label]}")

        # Şişen konuşmacı sayısını düzelt: birbirine çok benzeyen bilinenleri
        # birleştir ve bu chunk'ın eşlemesini de remap'le güncelle.
        remap = self._merge_similar_speakers()
        if remap:
            for local_label, glabel in mapping.items():
                if glabel in remap:
                    mapping[local_label] = remap[glabel]

        return mapping

    def map_speakers_fallback(self, local_labels):
        """Embedding çıkarılamadığında (çok kısa/sessiz chunk) fallback.

        Eski davranış her local label'a YENİ bir global etiket uyduruyordu;
        embedding'i olmayan bu 'hayalet' konuşmacılar hiçbir zaman eşlenemediği
        için konuşmacı sayısını şişiriyordu (CHiME-6 over-count'un ana kaynağı).
        Artık kimliklendirilemeyen ses dürüstçe 'Unknown' etiketlenir.
        """
        return {label: "Unknown" for label in local_labels}
