"""Oturum boyu transkript birikimi + oturum-sonu konuşmacı düzeltmesi.

Canlı takip (SpeakerTracker) kararlarını GERİ ALAMAZ: bir chunk'a verilen etiket
kalıcıdır. speaker_refinement bu kararların bir kısmını düzeltebilir ama doğası
gereği OFFLINE'dır — bir etiketin hatalı olduğunu ancak SONRAKİ konuşmayı da
görünce anlar. Bu yüzden canlı akışta çalıştırılamaz.

Bu sınıf ikisini birleştirir: diarization thread'i her chunk'ın konuşmacı-etiketli
segmentlerini buraya biriktirir; oturum bitince `refine()` tüm zaman çizelgesine
tek seferde refinement uygular ve YALNIZCA etiketi değişen chunk'lar için yeniden
biçimlendirilmiş metni döner. Böylece tüketici (UI log'u / CLI çıktısı) tüm
transkripti değil, sadece düzelen satırları yeniden yazar.

Ölçülmüş kazanç (AMI, 8 toplantı — bkz. RefinementConfig docstring'i):
DER 34.45 -> 33.78, cpWER 50.87 -> 49.59, konuşmacı sayısı 75 -> 55.

Saf Python (torch/model/IO yok) — birim testleri modelsiz koşar.
"""

from __future__ import annotations

import threading

from .formatting import format_results
from .speaker_refinement import RefinementConfig, refine_speakers


class SessionTranscript:
    """Chunk bazlı konuşmacı segmentlerini biriktirir, oturum sonunda düzeltir."""

    def __init__(self):
        # {segment_index: [segment dict, ...]} — chunk içi sıra korunur.
        self._chunks: dict[int, list[dict]] = {}
        # add_chunk diarization thread'inden, refine kapanış yolundan çağrılır.
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Yeni bir oturum için geçmişi temizler."""
        with self._lock:
            self._chunks = {}

    def add_chunk(self, segment_index: int, segments) -> None:
        """Bir chunk'ın konuşmacı-etiketli segmentlerini biriktirir.

        Segmentler kopyalanır: çağıran listeyi sonradan değiştirse bile oturum
        geçmişi bozulmaz.

        Zaman damgaları CHUNK-GÖRELİ kalabilir — refinement kuralları yalnızca
        segment SIRASI ve SÜRESİ (end - start) üzerinden çalışır, mutlak zamana
        hiç bakmaz. Sıra, segment_index'ten (chunk'lar arası) ve liste
        konumundan (chunk içi) gelir.
        """
        if not segments:
            return
        with self._lock:
            self._chunks[segment_index] = [dict(s) for s in segments]

    def refine(self, cfg: RefinementConfig | None = None) -> tuple[dict[int, str], dict]:
        """Tüm oturuma refinement uygular.

        Args:
            cfg: refinement eşikleri (None → ölçülmüş varsayılanlar).

        Returns:
            (updates, stats):
                updates: {segment_index: yeniden biçimlendirilmiş metin} —
                    YALNIZCA en az bir segmentinin etiketi değişen chunk'lar.
                    Hiçbir şey değişmediyse boş dict.
                stats: refine_speakers istatistikleri (kural başına sayaçlar).
        """
        with self._lock:
            ordered_indexes = sorted(self._chunks)
            flat: list[dict] = []
            owners: list[int] = []
            for index in ordered_indexes:
                for segment in self._chunks[index]:
                    flat.append(dict(segment))
                    owners.append(index)

        if not flat:
            return {}, {"total": 0, "passes_run": 0}

        refined, stats = refine_speakers(flat, cfg)
        if not stats.get("total"):
            return {}, stats

        # Yalnızca etiketi DEĞİŞEN chunk'lar yeniden yazılır: değişmemiş satırı
        # yeniden biçimlendirmek tüketiciye gereksiz iş ve titreme yaratır.
        touched = {
            owner
            for owner, before, after in zip(owners, flat, refined)
            if before.get("speaker") != after.get("speaker")
        }
        if not touched:
            return {}, stats

        grouped: dict[int, list[dict]] = {}
        for owner, segment in zip(owners, refined):
            if owner in touched:
                grouped.setdefault(owner, []).append(segment)

        updates = {}
        for index, segments in grouped.items():
            text = format_results(segments, return_str=True)
            if text:
                updates[index] = text
        return updates, stats
