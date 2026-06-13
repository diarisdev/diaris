"""
Diarization yardımcıları — ağır bağımlılık (torch/pyannote) içermez.

Bu modül saf-Python mantığı barındırır ki birim testleri model/torch
kurulumu olmadan da çalışabilsin.
"""


def assign_words_to_speakers(transcribed_segments, turns):
    """
    Kelime-seviyesi konuşmacı ataması + sınır bölme.

    Her kelime, orta noktasının düştüğü diarization turn'üne atanır. Böylece
    Whisper'ın A'nın kuyruğu ile B'nin başını birleştirdiği segmentler,
    gerçek konuşmacı değişim noktasından bölünür ("A'nın son cümlesi B'ye
    geçiyor" sorununun çözümü).

    Kelime zaman damgası yoksa segment-seviyesi overlap'e geri düşer.

    Args:
        transcribed_segments: [{"start","end","text","words"?}] — words opsiyonel,
            her biri {"start","end","word"}.
        turns: [{"start","end","speaker"}] — global etiketli diarization turn'leri.

    Returns:
        list[{"speaker","start","end","text"}] — konuşmacı değişiminde bölünmüş,
            ardışık aynı-konuşmacı kelimeler birleştirilmiş segmentler.
    """
    def speaker_at(t_mid):
        """Verilen zaman noktasını içeren turn'ün konuşmacısı; yoksa en yakını."""
        # 1) Noktayı kapsayan turn
        for turn in turns:
            if turn["start"] <= t_mid <= turn["end"]:
                return turn["speaker"]
        # 2) Kapsayan yoksa zamanca en yakın turn
        best_speaker = None
        best_dist = float("inf")
        for turn in turns:
            if t_mid < turn["start"]:
                dist = turn["start"] - t_mid
            elif t_mid > turn["end"]:
                dist = t_mid - turn["end"]
            else:
                dist = 0.0
            if dist < best_dist:
                best_dist = dist
                best_speaker = turn["speaker"]
        return best_speaker if best_speaker is not None else "Unknown"

    def segment_speaker_by_overlap(seg):
        """Kelime yoksa segmenti en çok örtüşen konuşmacıya ata (eski yöntem)."""
        best_speaker, best_overlap = "Unknown", 0.0
        for turn in turns:
            overlap = max(0.0, min(seg["end"], turn["end"]) - max(seg["start"], turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        return best_speaker

    if not turns:
        return [
            {"speaker": "Unknown", "start": s["start"], "end": s["end"], "text": s["text"]}
            for s in transcribed_segments
        ]

    out = []
    for seg in transcribed_segments:
        words = seg.get("words") or []
        if not words:
            # Kelime damgası yok → segment-seviyesi atama
            out.append({
                "speaker": segment_speaker_by_overlap(seg),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
            continue

        # Her kelimeyi orta noktasına göre bir konuşmacıya ata
        for w in words:
            mid = (w["start"] + w["end"]) / 2.0
            spk = speaker_at(mid)
            text = w["word"]
            if out and out[-1]["speaker"] == spk and out[-1].get("_open"):
                out[-1]["end"] = w["end"]
                out[-1]["text"] += text
            else:
                # Önceki segmenti kapat
                if out:
                    out[-1].pop("_open", None)
                out.append({
                    "speaker": spk,
                    "start": w["start"],
                    "end": w["end"],
                    "text": text,
                    "_open": True,
                })

    for seg in out:
        seg.pop("_open", None)
        seg["text"] = seg["text"].strip()

    # Boş metinli segmentleri at
    return [s for s in out if s["text"]]
