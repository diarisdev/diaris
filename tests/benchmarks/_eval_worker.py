"""AMI-eval davranışı — production src/core/ai_worker.py'ye DOKUNMADAN.

Canlı YouTube yolu HEAD'de bozulmadan kalsın diye, AMI'ye özel agresif
heuristic'ler burada subclass ile enjekte edilir; `src/core/ai_worker.py`
hiç değiştirilmez (eval ↔ prod ayrımı). Önceki denemede bu davranışlar paylaşılan
dosyaya konunca YouTube konuşmacı ayrımı bozulmuştu; subclass bunu önler.

Eval davranışları (HEAD/live'a göre fark):
  * map_speakers_fallback: embedding çıkarılamayan kısa chunk'ta yeni hayalet
    konuşmacı YARATMA → "Unknown" bırak. (AMI over-count'un asıl kaynağı buydu:
    tek başına "Okay."/"Yeah." söyleyen her kısa chunk ayrı SPEAKER üretiyordu.)
  * _smooth_speaker_labels: < 1s segmenti baskın (uzun) komşunun konuşmacısına
    koşulsuz devret (agresif) — kısa kalıntılar ayrı turn oluşturmasın.

NOT: İlk denemedeki Whisper halüsinasyon filtresi (no_speech_prob/avg_logprob)
bilerek dahil EDİLMEDİ; o, ai_worker.py'nin process_chunk'ını değiştirmeyi
gerektiriyordu (bu skorlar results'a yazılmıyor) ve amaç ai_worker'a hiç
dokunmamak. Fallback→Unknown + replay.py'deki sayımdan Unknown/CALIBRATING
çıkarma over-count'u büyük ölçüde zaten kapatıyor. Gerekirse ileride ai_worker'a
*additive* bir alanla (davranışı değiştirmeden) geri eklenebilir.
"""

import torch

from src.core.ai_worker import AIWorker, SpeakerTracker


class EvalSpeakerTracker(SpeakerTracker):
    """Eval varyantı: fallback→Unknown + warm-up singleton kümelerini ELEME."""

    def map_speakers_fallback(self, local_labels):
        return {label: "Unknown" for label in local_labels}

    def _finalize_warmup(self):
        """Base ile aynı agglomerative clustering, ama küçük-küme (singleton)
        filtresi YOK.

        Base sürüm n>=6'da <2 üyeli kümeleri 'gürültü' diye atıyor; AMI'de ilk
        ~35s'de bir kez konuşan gerçek konuşmacılar tek embedding verince bu
        filtre onları siliyor ve tüm toplantı TEK konuşmacıya çöküyor
        (TS3003a: 7 embedding → 5 filtrelendi → 1 konuşmacı). Burada tüm kümeler
        korunur: over-collapse, hafif over-count'tan çok daha pahalı (DER conf /
        cpWER). ai_worker.py'ye dokunmadan, yalnızca eval yolunda geçerli.
        """
        buf = self._warmup_buffer
        if not buf:
            self._warmup_complete = True
            return

        n = len(buf)
        if n == 1:
            self.known_speakers[self._next_label()] = buf[0]
            self._warmup_complete = True
            self._warmup_buffer = []
            print("[Warm-up Complete] 1 speaker (eval) ")
            return

        emb = torch.stack(buf)
        sim = torch.nn.functional.cosine_similarity(
            emb.unsqueeze(0), emb.unsqueeze(1), dim=2
        )
        clusters = {i: [i] for i in range(n)}
        while len(clusters) > 1:
            best_i, best_j, best = -1, -1, -1.0
            keys = list(clusters)
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    ci, cj = keys[a], keys[b]
                    tot = sum(sim[mi, mj].item()
                              for mi in clusters[ci] for mj in clusters[cj])
                    avg = tot / (len(clusters[ci]) * len(clusters[cj]))
                    if avg > best:
                        best, best_i, best_j = avg, ci, cj
            if best < self.threshold or best_i < 0:
                break
            clusters[best_i].extend(clusters[best_j])
            del clusters[best_j]

        # Singleton filtreleme YOK — her küme bir konuşmacı
        for members in clusters.values():
            centroid = torch.stack([buf[i] for i in members]).mean(dim=0)
            norm = torch.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.known_speakers[self._next_label()] = centroid

        self._warmup_complete = True
        self._warmup_buffer = []
        print(f"[Warm-up Complete] {len(self.known_speakers)} speaker(s) detected "
              f"(eval, filtresiz): {', '.join(self.known_speakers)}")


class EvalAIWorker(AIWorker):
    """Eval için agresif diarization davranışı (AMI replay + CHiME-6 benchmark).

    Over-count düzeltmeleri: fallback→Unknown + agresif smoothing.
    ai_worker.py'ye (canlı/YouTube yolu) dokunmaz.
    """

    def __init__(self, rate=None, channels=None, embedding_threshold=None):
        super().__init__(rate=rate, channels=channels)
        # Tracker'ı eval varyantıyla değiştir. embedding_threshold None ise
        # config/.env varsayılanı kullanılır; verilirse eval'e özel (canlı etkilenmez).
        self.speaker_tracker = EvalSpeakerTracker(threshold=embedding_threshold)

    def _smooth_speaker_labels(self, results):
        """Agresif: < 1s segmenti baskın komşunun konuşmacısına devret.

        Komşulardan biriyle zaten aynıysa dokunmaz; değilse daha uzun komşunun
        konuşmacısına (eşitse öncekine) atar.
        """
        if len(results) < 3:
            return results

        smoothed = [dict(r) for r in results]
        for i in range(1, len(smoothed) - 1):
            seg = smoothed[i]
            if seg["end"] - seg["start"] >= 1.0:
                continue
            prev_seg = smoothed[i - 1]
            next_seg = smoothed[i + 1]
            if seg["speaker"] in (prev_seg["speaker"], next_seg["speaker"]):
                continue
            prev_dur = prev_seg["end"] - prev_seg["start"]
            next_dur = next_seg["end"] - next_seg["start"]
            seg["speaker"] = (
                prev_seg["speaker"] if prev_dur >= next_dur else next_seg["speaker"]
            )
        return smoothed
