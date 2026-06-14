"""SpeakerTracker — embedding tabanlı konuşmacı kimlik takibi.

ai_worker.py'den ayrıldı (saf taşıma; davranış birebir aynı). ai_worker bunu geri
export eder; `from src.core.ai_worker import SpeakerTracker` ve subclass'lar
(EvalSpeakerTracker) aynen çalışmaya devam eder.
"""

import torch

from ..config import (
    DIARIZATION_EMBEDDING_THRESHOLD, DIARIZATION_WARMUP_MS,
    CANDIDATE_CONFIRMATIONS_NEEDED, CANDIDATE_TTL, CANDIDATE_SELF_SIMILARITY,
    MIN_NEW_SPEAKER_DURATION,
)


class SpeakerTracker:
    """
    Embedding-based konuşmacı takip sistemi (warm-up destekli).
    
    Faz 1 (Warm-up): Kaliteli embedding'ler toplanır, kümelenir.
    Faz 2 (Aktif):   Yeni embedding'ler baseline'larla karşılaştırılır.
                      Güncelleme confidence-weighted yapılır.

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

    def __init__(self, threshold=None, warmup_ms=None):
        self.threshold = threshold if threshold is not None else DIARIZATION_EMBEDDING_THRESHOLD
        self.warmup_ms = warmup_ms if warmup_ms is not None else DIARIZATION_WARMUP_MS

        # Bilinen konuşmacılar (warm-up sonrası dolu olur)
        self.known_speakers = {}  # {global_label: embedding_tensor}
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
            label = self._next_label()
            self.known_speakers[label] = self._warmup_buffer[0]
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
        cluster_ids = list(range(n))
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

        # Her kümeden konuşmacı oluştur
        for cluster_id, member_indices in valid_clusters.items():
            member_embs = [self._warmup_buffer[i] for i in member_indices]
            centroid = torch.stack(member_embs).mean(dim=0)
            label = self._next_label()
            self.known_speakers[label] = centroid

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

    def _find_matching_candidate(self, emb):
        """
        Aday tamponunda bu embedding'e benzer bir aday var mı?
        Varsa adayın index'ini döndürür, yoksa -1.
        """
        for idx, cand in enumerate(self._candidates):
            centroid = torch.stack(cand["embeddings"]).mean(dim=0)
            score = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), centroid.unsqueeze(0)
            ).item()
            if score >= self.CANDIDATE_SELF_SIMILARITY:
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
                    s = torch.nn.functional.cosine_similarity(
                        embs[i].unsqueeze(0), embs[j].unsqueeze(0)
                    ).item()
                    pair_scores.append(s)
            avg_self_sim = sum(pair_scores) / len(pair_scores)
            if avg_self_sim < self.CANDIDATE_SELF_SIMILARITY:
                return None

        # Onaylandı — yeni konuşmacı oluştur
        centroid = torch.stack(embs).mean(dim=0)
        norm = torch.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        new_label = self._next_label()
        self.known_speakers[new_label] = centroid
        return new_label

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
        kendiliğinden düzelir.

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
                    sim = torch.nn.functional.cosine_similarity(
                        self.known_speakers[keep].unsqueeze(0),
                        self.known_speakers[drop].unsqueeze(0),
                    ).item()
                    if sim >= merge_threshold:
                        centroid = (self.known_speakers[keep] + self.known_speakers[drop]) / 2.0
                        norm = torch.norm(centroid)
                        if norm > 0:
                            centroid = centroid / norm
                        self.known_speakers[keep] = centroid
                        del self.known_speakers[drop]
                        remap[drop] = keep
                        labels.pop(j)
                        print(f"  [Merge] {drop} → {keep} (sim: {sim:.3f})")
                        continue
                j += 1
            i += 1
        return remap

    def map_speakers(self, embeddings_dict, quality_dict=None):
        """
        Embedding'lere göre konuşmacıları eşler (warm-up sonrası).
        Confidence-weighted baseline güncelleme yapar.
        Bilinmeyen sesler için onay tamponu kullanır.

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

            # Bilinen konuşmacılarla karşılaştır
            best_match = None
            best_score = -1.0

            for global_label, known_emb in self.known_speakers.items():
                score = torch.nn.functional.cosine_similarity(
                    emb.unsqueeze(0), known_emb.unsqueeze(0)
                ).item()
                if score > best_score:
                    best_score = score
                    best_match = global_label

            if best_match and best_score >= self.threshold:
                mapping[local_label] = best_match

                # Centroid drift'i önle: baseline'ı YALNIZCA yüksek güvende
                # güncelle. Borderline eşleşmeler (0.70-0.85) centroid'i yavaşça
                # "ortalama bir ses"e kaydırıp baskın konuşmacının herkese
                # benzemesine yol açıyordu (AMI over-collapse). Artık sadece
                # >0.85 skorlu, kesin eşleşmeler küçük bir adımla günceller.
                if best_score > 0.85:
                    alpha = 0.8  # eski centroid ağırlığı (küçük hareket)
                    self.known_speakers[best_match] = (
                        alpha * self.known_speakers[best_match] + (1 - alpha) * emb
                    )
                    # Re-normalize baseline to prevent magnitude decay
                    norm = torch.norm(self.known_speakers[best_match])
                    if norm > 0:
                        self.known_speakers[best_match] /= norm

            elif not is_reliable:
                # Bilinmeyen AMA kısa/güvenilmez ses → yeni konuşmacı YARATMA.
                # En yakın mevcut konuşmacıya yapıştır (sticky). Bu, kısa
                # cümlelerin ("evet") yeni konuşmacı doğurmasını engeller.
                mapping[local_label] = best_match if best_match else "Unknown"
                if best_match:
                    print(f"  [Short utterance] sticky → {best_match} "
                          f"(score: {best_score:.3f}, {duration:.1f}s)")

            else:
                # Bilinmeyen VE güvenilir (yeterince uzun) ses
                # — aday tamponuna ekle veya mevcut adayı onayla.
                # NOT: Eski "belirsizlik bandı" (threshold-0.15 ile threshold arası)
                # bu sesleri zorla en yakın baskın konuşmacıya atıyordu ve farklı
                # gerçek konuşmacıları yutuyordu. O dal kaldırıldı; eşik altındaki
                # güvenilir sesler artık aday tamponundan geçip kendi konuşmacı
                # etiketini oluşturabilir. Aday kapısı (CONFIRMATIONS=4,
                # SELF_SIMILARITY=0.78) hayalet konuşmacıları yine de bastırır.
                cand_idx = self._find_matching_candidate(emb)

                if cand_idx >= 0:
                    # Mevcut adaya yeni gözlem ekle
                    self._candidates[cand_idx]["embeddings"].append(emb.clone())
                    promoted_label = self._try_promote_candidate(self._candidates[cand_idx])

                    if promoted_label:
                        # Aday onaylandı, gerçek konuşmacı oldu
                        mapping[local_label] = promoted_label
                        closest_info = f" (closest: {best_match}, score: {best_score:.3f})" if best_match else ""
                        confirms = len(self._candidates[cand_idx]["embeddings"])
                        print(f"  🆕 New speaker: {promoted_label}{closest_info} [confirmed after {confirms} observations]")
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
        """Embedding yoksa fallback — her label'a yeni isim atar."""
        mapping = {}
        for label in local_labels:
            mapping[label] = self._next_label()
        return mapping
