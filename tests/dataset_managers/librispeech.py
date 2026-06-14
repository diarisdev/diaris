"""LibriSpeech test-clean — İngilizce temiz konuşma (ASR/transkripsiyon benchmark).

Kullanım:
    manager = DatasetManager()
    manager.download()
    samples = manager.get_samples(limit=10)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import DATASETS_ROOT, download_targz, sf

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
# Arşivin iç yapısı: LibriSpeech/test-clean/...
LIBRISPEECH_INNER = os.path.join("LibriSpeech", "test-clean")


def _parse_transcription_file(trans_path):
    """LibriSpeech .trans.txt -> {utterance_id: transcript_text}."""
    transcriptions = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                utt_id, transcript = parts
                transcriptions[utt_id] = transcript.strip()
    return transcriptions


class DatasetManager:
    """LibriSpeech test-clean indir/aç + (ses, transkripsiyon) çiftleri."""

    def __init__(self, data_dir=None):
        # data_dir = datasets/librispeech ; içine arşiv LibriSpeech/test-clean açar
        self.data_dir = data_dir or os.path.join(DATASETS_ROOT, "librispeech")
        self.archive_path = os.path.join(self.data_dir, "test-clean.tar.gz")
        self.dataset_root = os.path.join(self.data_dir, LIBRISPEECH_INNER)

    def is_downloaded(self):
        return os.path.isdir(self.dataset_root)

    def download(self, force=False):
        if self.is_downloaded() and not force:
            print("✅ LibriSpeech zaten mevcut, indirme atlanıyor.")
            return True
        return download_targz(
            LIBRISPEECH_URL, self.archive_path, self.data_dir,
            label="LibriSpeech test-clean",
            size_hint="Boyut ~350 MB, internet hızınıza bağlı olarak biraz sürebilir.",
        )

    def get_samples(self, limit=None, min_duration=1.0, max_duration=30.0):
        if not self.is_downloaded():
            print("⚠️  Veri seti bulunamadı. Önce download() çağırın.")
            return []
        if sf is None:
            raise ImportError("LibriSpeech örnekleri için 'soundfile' gerekli: pip install soundfile")

        samples = []
        for speaker_dir in sorted(Path(self.dataset_root).iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            for chapter_dir in sorted(speaker_dir.iterdir()):
                if not chapter_dir.is_dir():
                    continue
                chapter_id = chapter_dir.name
                trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                if not trans_file.exists():
                    continue
                transcriptions = _parse_transcription_file(trans_file)
                for utt_id, transcript in transcriptions.items():
                    audio_path = chapter_dir / f"{utt_id}.flac"
                    if not audio_path.exists():
                        continue
                    try:
                        duration = sf.info(str(audio_path)).duration
                    except Exception:
                        continue
                    if duration < min_duration or duration > max_duration:
                        continue
                    samples.append({
                        "audio_path": str(audio_path),
                        "transcript": transcript,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utt_id,
                        "duration": round(duration, 2),
                    })
                    if limit and len(samples) >= limit:
                        return samples
        return samples

    def get_summary(self):
        samples = self.get_samples()
        if not samples:
            return {"status": "not_ready", "count": 0}
        durations = [s["duration"] for s in samples]
        speakers = set(s["speaker_id"] for s in samples)
        return {
            "status": "ready",
            "count": len(samples),
            "total_duration_min": round(sum(durations) / 60, 1),
            "avg_duration_sec": round(sum(durations) / len(durations), 1),
            "min_duration_sec": round(min(durations), 1),
            "max_duration_sec": round(max(durations), 1),
            "unique_speakers": len(speakers),
        }

    def cleanup(self):
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
            print("🗑️  LibriSpeech silindi.")
