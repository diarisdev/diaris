import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

class SegmentStatus(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"
    REVISED = "revised"

@dataclass
class TranscriptSegment:
    id: str = field(default_factory=lambda: str(uuid4()))
    stream_id: str = "default"
    start_time: float = 0.0
    end_time: float = 0.0
    text: str = ""
    speaker_tag: str | None = None
    speaker_confidence: float | None = None
    source_language: str = "auto"
    target_language: str = "tr"
    translated_text: str | None = None
    translation_status: SegmentStatus = SegmentStatus.PARTIAL
    status: SegmentStatus = SegmentStatus.PARTIAL
    stt_confidence: float | None = None
    revision: int = 0
    stt_latency_ms: float | None = None
    total_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stream_id": self.stream_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speaker_tag": self.speaker_tag,
            "speaker_confidence": self.speaker_confidence,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "translated_text": self.translated_text,
            "translation_status": self.translation_status.value if isinstance(self.translation_status, SegmentStatus) else self.translation_status,
            "status": self.status.value if isinstance(self.status, SegmentStatus) else self.status,
            "stt_confidence": self.stt_confidence,
            "revision": self.revision,
            "stt_latency_ms": self.stt_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class SpeakerTurn:
    start_time: float
    end_time: float
    speaker_tag: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker_tag": self.speaker_tag,
            "confidence": self.confidence,
        }

class PipelineEventType(str, Enum):
    AUDIO_FRAME_CAPTURED = "AudioFrameCaptured"
    SPEECH_STATE_CHANGED = "SpeechStateChanged"
    CHUNK_READY_FOR_STT = "ChunkReadyForSTT"
    CHUNK_READY_FOR_DIARIZATION = "ChunkReadyForDiarization"
    TRANSCRIPT_SEGMENT_CREATED = "TranscriptSegmentCreated"
    TRANSCRIPT_SEGMENT_FINALIZED = "TranscriptSegmentFinalized"
    TRANSLATION_UPDATED = "TranslationUpdated"
    SPEAKER_TURNS_READY = "SpeakerTurnsReady"
    TRANSCRIPT_SEGMENT_REVISED = "TranscriptSegmentRevised"
    PIPELINE_STATUS_CHANGED = "PipelineStatusChanged"

@dataclass
class PipelineEvent:
    event_type: PipelineEventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
