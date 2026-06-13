#!/usr/bin/env python3
"""
CHiME-6 Benchmark & Evaluation Tool.

All-in-one script for evaluating ASR and diarization performance on CHiME-6
dataset sessions. Computes WER, CER, cpWER, DER and per-speaker breakdowns.

Usage examples:
    # 5 minutes of S02, raw Whisper, channel 1
    python tests/chime6_benchmark_new.py --session S02 --limit-minutes 5

    # Full session, AIWorker mode
    python tests/chime6_benchmark_new.py --session S02 --mode aiworker

    # All sessions, save JSON report
    python tests/chime6_benchmark_new.py --save-report results.json
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Silence noisy third-party warnings before any imports
# ---------------------------------------------------------------------------
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=r".*std\(\).*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import gc
import itertools
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Project bootstrap — add root to path and configure CUDA DLLs
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import configure_cuda_dll_paths, WHISPER_PATH, DEVICE, COMPUTE_TYPE

configure_cuda_dll_paths()


# ===================================================================
#  Section 1 — Data Loading
# ===================================================================

def _parse_timestamp(ts) -> float:
    """Convert HH:MM:SS.ss or numeric value to float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


def _clean_text(text: str) -> str:
    """Normalize text for fair WER/CER comparison."""
    if not text:
        return ""
    # Remove bracketed annotations like [laughs], [noise]
    text = re.sub(r"\[[^\]]*\]", "", text)
    # Hyphens → spaces
    text = text.replace("-", " ")
    # Keep only word chars, spaces and apostrophes
    text = re.sub(r"[^\w\s']", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


@dataclass
class Segment:
    """A single speech segment with speaker label and text."""
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_transcript(json_path: str | Path) -> list[Segment]:
    """Load and parse a CHiME-6 session transcript JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    segments: list[Segment] = []
    for item in raw:
        try:
            start = _parse_timestamp(item["start_time"])
            end = _parse_timestamp(item["end_time"])
        except (KeyError, ValueError):
            continue

        speaker = item.get("speaker", "UNK")
        text = _clean_text(item.get("words") or item.get("text") or "")
        if not text or end <= start:
            continue

        segments.append(Segment(speaker=speaker, start=start, end=end, text=text))

    segments.sort(key=lambda s: s.start)
    return segments


def apply_time_limit(segments: list[Segment], limit_sec: float) -> list[Segment]:
    """Truncate segments to a maximum time window."""
    out: list[Segment] = []
    for seg in segments:
        if seg.start >= limit_sec:
            continue
        end = min(seg.end, limit_sec)
        if end > seg.start:
            out.append(Segment(speaker=seg.speaker, start=seg.start, end=end, text=seg.text))
    return out


# ===================================================================
#  Section 2 — Audio Loading
# ===================================================================

def _read_audio_segment(path: Path, start: float, end: float) -> tuple[np.ndarray | None, int]:
    """Read a segment of audio, return (float32_mono_16k, 16000) or (None, 0)."""
    if not path.exists():
        return None, 0
    try:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            total = len(f)
            s_frame = max(0, int(round(start * sr)))
            e_frame = min(total, int(round(end * sr)))
            if s_frame >= e_frame:
                return None, 0
            f.seek(s_frame)
            data = f.read(e_frame - s_frame, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
    except Exception:
        return None, 0

    # Resample to 16 kHz if needed
    if sr != 16000:
        tensor = torch.from_numpy(data).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=16000)
        data = tensor.squeeze(0).numpy()
    return data, 16000


def resolve_audio_path(data_dir: Path, session: str, array: str) -> Path:
    """Return the path to channel-1 audio for the given session and array."""
    return data_dir / "audio" / f"{session}_{array}.CH1.wav"


# ===================================================================
#  Section 3 — ASR Engines (Raw Whisper & AIWorker)
# ===================================================================

class RawWhisperEngine:
    """Lightweight wrapper around faster-whisper for direct transcription."""

    def __init__(self):
        from faster_whisper import WhisperModel
        print(f"   Loading Whisper model: {WHISPER_PATH}")
        self.model = WhisperModel(WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"   Device: {DEVICE} | Compute: {COMPUTE_TYPE}")

    def transcribe(self, audio_f32: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio_f32, beam_size=3, language="en", condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments)

    def transcribe_and_diarize(self, audio_f32: np.ndarray) -> list[dict]:
        """Return list of {speaker, start, end, text} with dummy speaker."""
        text = self.transcribe(audio_f32)
        if not text.strip():
            return []
        duration = len(audio_f32) / 16000.0
        return [{
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": duration,
            "text": text,
        }]


class AIWorkerEngine:
    """Uses the full AIWorker pipeline (Whisper + Pyannote diarization)."""

    def __init__(self):
        from src.core.ai_worker import AIWorker
        self._worker = AIWorker(rate=16000, channels=1)
        if not self._worker.load_models():
            raise RuntimeError("AIWorker failed to load models")

    def transcribe(self, audio_f32: np.ndarray) -> str:
        # AIWorker expects int16 bytes
        int16 = (audio_f32 * 32768).clip(-32768, 32767).astype(np.int16)
        output = self._worker.process_chunk(int16.tobytes(), is_final=True, language="en")
        if not output:
            return ""
        results = output.get("results", [])
        return " ".join(r["text"].strip() for r in results)

    def transcribe_and_diarize(self, audio_f32: np.ndarray) -> list[dict]:
        """Return list of {speaker, start, end, text} with diarization."""
        int16 = (audio_f32 * 32768).clip(-32768, 32767).astype(np.int16)
        output = self._worker.process_chunk(int16.tobytes(), is_final=True, language="en")
        if not output:
            return []
        results = output.get("results", [])
        waveform = output.get("waveform_16k")
        sr = output.get("sample_rate", 16000)
        dur_ms = output.get("chunk_duration_ms", 0)
        if waveform is not None and results:
            results = self._worker.run_diarization(waveform, sr, dur_ms, results)
        return results


# ===================================================================
#  Section 4 — Metrics
# ===================================================================

def _levenshtein_ops(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple[int, int, int]:
    """Compute substitutions, insertions, deletions via fast rapidfuzz or exact Levenshtein DP."""
    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0:
        return 0, m, 0
    if m == 0:
        return 0, 0, n

    # Try rapidfuzz first
    try:
        from rapidfuzz.distance import Levenshtein
        ops = Levenshtein.editops(ref_tokens, hyp_tokens)
        subs = sum(1 for op in ops if op.tag == 'replace')
        ins = sum(1 for op in ops if op.tag == 'insert')
        dels = sum(1 for op in ops if op.tag == 'delete')
        return subs, ins, dels
    except ImportError:
        pass

    # Fallback to exact Levenshtein DP (since SciPy linear assignment reduces calls to only 32 times, exact DP is extremely fast)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1]
                ins = dp[i][j - 1]
                dl = dp[i - 1][j]
                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),       # substitution
                    (ins[0] + 1, ins[1], ins[2] + 1, ins[3]),       # insertion
                    (dl[0] + 1, dl[1], dl[2], dl[3] + 1),           # deletion
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])
    _, subs, ins, dels = dp[n][m]
    return subs, ins, dels


@dataclass
class WERResult:
    """Word Error Rate breakdown."""
    ref_words: int
    hyp_words: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def wer(self) -> float:
        return self.errors / max(self.ref_words, 1) * 100.0


@dataclass
class CERResult:
    """Character Error Rate breakdown."""
    ref_chars: int
    hyp_chars: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def cer(self) -> float:
        return self.errors / max(self.ref_chars, 1) * 100.0


def compute_wer(ref: str, hyp: str) -> WERResult:
    """Compute WER between reference and hypothesis strings."""
    ref_tokens = ref.lower().split()
    hyp_tokens = hyp.lower().split()
    if not ref_tokens:
        return WERResult(0, len(hyp_tokens), 0, len(hyp_tokens), 0)
    subs, ins, dels = _levenshtein_ops(ref_tokens, hyp_tokens)
    return WERResult(len(ref_tokens), len(hyp_tokens), subs, ins, dels)


def compute_cer(ref: str, hyp: str) -> CERResult:
    """Compute CER between reference and hypothesis strings."""
    ref_chars = list(ref.lower().replace(" ", ""))
    hyp_chars = list(hyp.lower().replace(" ", ""))
    if not ref_chars:
        return CERResult(0, len(hyp_chars), 0, len(hyp_chars), 0)
    subs, ins, dels = _levenshtein_ops(ref_chars, hyp_chars)
    return CERResult(len(ref_chars), len(hyp_chars), subs, ins, dels)


def compute_cpwer(
    ref_segments: list[Segment],
    hyp_segments: list[Segment],
) -> dict:
    """
    Concatenated minimum-permutation WER (cpWER).

    Finds the optimal mapping of hypothesis speakers to reference speakers
    that minimizes the total word error rate.
    """
    ref_speakers = sorted(set(s.speaker for s in ref_segments))
    hyp_speakers = sorted(set(s.speaker for s in hyp_segments))

    # Build per-speaker concatenated text
    ref_text_by_spk = {}
    for spk in ref_speakers:
        texts = [s.text for s in ref_segments if s.speaker == spk]
        ref_text_by_spk[spk] = " ".join(texts)

    hyp_text_by_spk = {}
    for spk in hyp_speakers:
        texts = [s.text for s in hyp_segments if s.speaker == spk]
        hyp_text_by_spk[spk] = " ".join(texts)

    if not ref_speakers:
        total_hyp = " ".join(hyp_text_by_spk.values())
        w = compute_wer("", total_hyp)
        return {
            "cpwer": w.wer,
            "mapping": {},
            "wer_result": w,
            "per_speaker_details": {}
        }

    best_mapping = {}

    try:
        from scipy.optimize import linear_sum_assignment
        N_ref = len(ref_speakers)
        
        # Construct cost matrix: row=ref, col=hyp
        # If we have fewer hypothesis speakers, we can pad with empty speakers
        # so that every reference speaker can map to something.
        padded_hyp_speakers = list(hyp_speakers)
        while len(padded_hyp_speakers) < N_ref:
            padded_hyp_speakers.append(f"__EMPTY_{len(padded_hyp_speakers)}__")
            
        N_hyp_pad = len(padded_hyp_speakers)
        cost_matrix = np.zeros((N_ref, N_hyp_pad))
        
        for i, ref_spk in enumerate(ref_speakers):
            ref_tokens = ref_text_by_spk[ref_spk].lower().split()
            for j, hyp_spk in enumerate(padded_hyp_speakers):
                hyp_tokens = hyp_text_by_spk.get(hyp_spk, "").lower().split()
                subs, ins, dels = _levenshtein_ops(ref_tokens, hyp_tokens)
                cost_matrix[i, j] = subs + ins + dels
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            best_mapping[ref_speakers[r]] = padded_hyp_speakers[c]
            
    except Exception:
        # Fallback to permutations if scipy is not available
        best_wer_val = float("inf")
        hyp_list = list(hyp_speakers)
        while len(hyp_list) < len(ref_speakers):
            hyp_list.append(f"__EMPTY_{len(hyp_list)}__")

        for perm in itertools.permutations(hyp_list, len(ref_speakers)):
            mapping = dict(zip(ref_speakers, perm))
            total_ref_tokens = []
            total_hyp_tokens = []

            for ref_spk, hyp_spk in mapping.items():
                ref_t = ref_text_by_spk.get(ref_spk, "").lower().split()
                hyp_t = hyp_text_by_spk.get(hyp_spk, "").lower().split()
                total_ref_tokens.extend(ref_t)
                total_hyp_tokens.extend(hyp_t)

            if not total_ref_tokens:
                continue

            subs, ins, dels = _levenshtein_ops(total_ref_tokens, total_hyp_tokens)
            wr = WERResult(len(total_ref_tokens), len(total_hyp_tokens), subs, ins, dels)
            if wr.wer < best_wer_val:
                best_wer_val = wr.wer
                best_mapping = mapping

    # Now calculate global WER result using the best mapping
    total_ref_tokens = []
    total_hyp_tokens = []
    for ref_spk, hyp_spk in best_mapping.items():
        ref_t = ref_text_by_spk.get(ref_spk, "").lower().split()
        hyp_t = hyp_text_by_spk.get(hyp_spk, "").lower().split()
        total_ref_tokens.extend(ref_t)
        total_hyp_tokens.extend(hyp_t)

    if total_ref_tokens:
        subs, ins, dels = _levenshtein_ops(total_ref_tokens, total_hyp_tokens)
        best_result = WERResult(len(total_ref_tokens), len(total_hyp_tokens), subs, ins, dels)
    else:
        best_result = WERResult(0, 0, 0, 0, 0)

    per_speaker_details = {}
    for ref_spk, hyp_spk in best_mapping.items():
        ref_t = ref_text_by_spk.get(ref_spk, "")
        hyp_t = hyp_text_by_spk.get(hyp_spk, "")
        w = compute_wer(ref_t, hyp_t)
        c = compute_cer(ref_t, hyp_t)
        per_speaker_details[ref_spk] = {
            "mapped_to": hyp_spk,
            "wer": w,
            "cer": c,
            "ref_segments": len([s for s in ref_segments if s.speaker == ref_spk]),
        }

    return {
        "cpwer": best_result.wer,
        "mapping": best_mapping,
        "wer_result": best_result,
        "per_speaker_details": per_speaker_details,
    }


def compute_der(
    ref_segments: list[Segment],
    hyp_segments: list[Segment],
    collar: float = 0.25,
) -> dict | None:
    """Compute DER using pyannote.metrics if available."""
    try:
        from pyannote.core import Annotation, Segment as PSegment
        from pyannote.metrics.diarization import DiarizationErrorRate
    except ImportError:
        return None

    ref_ann = Annotation()
    for seg in ref_segments:
        ref_ann[PSegment(seg.start, seg.end)] = seg.speaker

    hyp_ann = Annotation()
    for seg in hyp_segments:
        hyp_ann[PSegment(seg.start, seg.end)] = seg.speaker

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    der_value = metric(ref_ann, hyp_ann)
    detail = metric.report()

    components = metric(ref_ann, hyp_ann, detailed=True)
    total = components.get("total", 0.0)
    if total == 0:
        total = 1e-8

    return {
        "der": der_value * 100.0,
        "false_alarm": (components.get("false alarm", 0.0) / total) * 100.0,
        "missed": (components.get("missed detection", 0.0) / total) * 100.0,
        "confusion": (components.get("confusion", 0.0) / total) * 100.0,
    }


# ===================================================================
#  Section 5 — Evaluation Runner
# ===================================================================

@dataclass
class SessionResult:
    session: str
    array: str
    mode: str
    duration_sec: float
    processing_time_sec: float
    num_ref_segments: int
    num_hyp_segments: int
    ref_speakers: list[str]
    hyp_speakers: list[str]
    global_wer: WERResult
    global_cer: CERResult
    cpwer_result: dict
    der_result: dict | None
    per_speaker: dict = field(default_factory=dict)


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:.1f}%)"


def evaluate_session(
    data_dir: Path,
    session: str,
    array: str,
    engine,
    segmentation: str = "oracle",
    limit_minutes: float | None = None,
    log_fn=print,
) -> SessionResult:
    """Run evaluation on a single CHiME-6 session (oracle or streaming)."""

    # --- Load ground truth ---
    transcript_path = data_dir / "transcriptions" / f"{session}.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    ref_segments = load_transcript(transcript_path)
    if limit_minutes:
        ref_segments = apply_time_limit(ref_segments, limit_minutes * 60.0)

    if not ref_segments:
        raise ValueError(f"No reference segments for {session} within time limit")

    audio_path = resolve_audio_path(data_dir, session, array)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    total_duration = ref_segments[-1].end
    ref_speakers = sorted(set(s.speaker for s in ref_segments))

    # --- Run processing ---
    hyp_segments: list[Segment] = []
    t0 = time.time()

    if segmentation == "oracle":
        log_fn(f"\n{'=' * 70}")
        log_fn(f"  SESSION: {session} | Array: {array} | Oracle Segments: {len(ref_segments)}")
        log_fn(f"  Duration: {total_duration:.1f}s ({total_duration / 60:.1f} min)")
        log_fn(f"  Speakers: {', '.join(ref_speakers)}")
        log_fn(f"{'=' * 70}")

        all_ref_text = []
        all_hyp_text = []
        per_speaker_ref: dict[str, list[str]] = {spk: [] for spk in ref_speakers}
        per_speaker_hyp: dict[str, list[str]] = {spk: [] for spk in ref_speakers}

        for idx, seg in enumerate(ref_segments):
            audio, sr = _read_audio_segment(audio_path, seg.start, seg.end)

            hyp_text = ""
            if audio is not None and len(audio) > 0:
                try:
                    raw = engine.transcribe(audio)
                    hyp_text = _clean_text(raw)
                except Exception as e:
                    log_fn(f"  WARNING: Transcription failed for segment {idx + 1}: {e}")

            hyp_segments.append(Segment(
                speaker=seg.speaker, start=seg.start, end=seg.end, text=hyp_text,
            ))

            all_ref_text.append(seg.text)
            all_hyp_text.append(hyp_text)
            per_speaker_ref[seg.speaker].append(seg.text)
            per_speaker_hyp[seg.speaker].append(hyp_text)

            # Progress
            done = idx + 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(ref_segments) - done) / rate if rate > 0 else 0
            bar = _progress_bar(done, len(ref_segments))
            print(f"\r  {bar} | {rate:.1f} seg/s | ETA: {eta:.0f}s", end="", flush=True)

            # Periodic memory cleanup
            if done % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print()  # newline after progress bar

    elif segmentation == "streaming":
        from src.audio.vad import VADEngine
        from src.config import (
            FRAME_DURATION_MS, SILENCE_LIMIT, SHORT_SILENCE_LIMIT,
            SOFT_CHUNK_DURATION_MS, MAX_CHUNK_DURATION_MS,
        )

        log_fn(f"\n{'=' * 70}")
        log_fn(f"  SESSION: {session} | Array: {array} | Continuous Streaming")
        log_fn(f"  Audio Path: {audio_path.name}")
        log_fn(f"{'=' * 70}")

        # Reset engine tracker if available
        if hasattr(engine, "_worker") and hasattr(engine._worker, "speaker_tracker"):
            engine._worker.speaker_tracker.reset()

        vad_engine = VADEngine()

        # Open source file for streaming
        with sf.SoundFile(str(audio_path)) as sf_file:
            sr = sf_file.samplerate
            frames_per_frame = int(sr * (FRAME_DURATION_MS / 1000.0))
            
            # Determine limit in frames
            limit_frames = int(sr * limit_minutes * 60.0) if limit_minutes else None
            total_duration = limit_minutes * 60.0 if limit_minutes else sf_file.frames / sr

            chunk_buffer_bytes = []
            silence_counter = 0
            has_spoken = False
            
            global_time_s = 0.0
            chunk_start_s = 0.0
            total_frames_read = 0

            def process_current_chunk():
                nonlocal chunk_buffer_bytes, silence_counter, has_spoken
                if not chunk_buffer_bytes:
                    return
                chunk_bytes = b''.join(chunk_buffer_bytes)
                chunk_np_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                chunk_f32 = chunk_np_int16.astype(np.float32) / 32768.0
                
                if sr != 16000:
                    tensor = torch.from_numpy(chunk_f32).unsqueeze(0)
                    tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=16000)
                    chunk_f32 = tensor.squeeze(0).numpy()
                
                try:
                    chunk_segs = engine.transcribe_and_diarize(chunk_f32)
                    for s_item in chunk_segs:
                        # Skip calibrating label
                        if "Calibrating" in s_item["speaker"]:
                            continue
                        global_start = chunk_start_s + s_item["start"]
                        global_end = chunk_start_s + s_item["end"]
                        # Clip boundaries
                        if limit_minutes and global_start >= limit_minutes * 60.0:
                            continue
                        if limit_minutes and global_end > limit_minutes * 60.0:
                            global_end = limit_minutes * 60.0
                        if global_end > global_start:
                            hyp_segments.append(Segment(
                                speaker=s_item["speaker"],
                                start=global_start,
                                end=global_end,
                                text=_clean_text(s_item["text"])
                            ))
                except Exception as e_chunk:
                    log_fn(f"\n  WARNING: Chunk transcription failed: {e_chunk}")
                
                chunk_buffer_bytes = []
                silence_counter = 0
                has_spoken = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            while True:
                if limit_frames and total_frames_read >= limit_frames:
                    if has_spoken and chunk_buffer_bytes:
                        process_current_chunk()
                    break
                
                data = sf_file.read(frames_per_frame, dtype='int16')
                if len(data) == 0:
                    if has_spoken and chunk_buffer_bytes:
                        process_current_chunk()
                    break
                
                total_frames_read += len(data)
                frame_bytes = data.tobytes()
                
                is_speech, conf = vad_engine.check_speech(frame_bytes, sr, 1)
                
                if is_speech:
                    chunk_buffer_bytes.append(frame_bytes)
                    silence_counter = 0
                    if not has_spoken:
                        has_spoken = True
                        chunk_start_s = global_time_s
                else:
                    silence_bytes = b'\x00' * (frames_per_frame * 2)
                    if has_spoken:
                        chunk_buffer_bytes.append(silence_bytes)
                        silence_counter += 1
                        
                global_time_s += (FRAME_DURATION_MS / 1000.0)
                
                current_duration_ms = len(chunk_buffer_bytes) * FRAME_DURATION_MS
                active_silence_limit = SHORT_SILENCE_LIMIT if current_duration_ms > SOFT_CHUNK_DURATION_MS else SILENCE_LIMIT
                
                if has_spoken and (silence_counter > active_silence_limit or current_duration_ms >= MAX_CHUNK_DURATION_MS):
                    process_current_chunk()
                
                # Periodic progress printing
                if total_frames_read % (frames_per_frame * 100) == 0:
                    progress_sec = total_frames_read / sr
                    pct = (progress_sec / total_duration) * 100
                    bar = _progress_bar(int(progress_sec), int(total_duration))
                    print(f"\r  {bar} | {progress_sec:.1f}s / {total_duration:.1f}s | hyp segs: {len(hyp_segments)}", end="", flush=True)

            print()  # newline after progress bar

    processing_time = time.time() - t0

    # --- Compute global metrics ---
    global_ref = " ".join(s.text for s in ref_segments)
    global_hyp = " ".join(s.text for s in hyp_segments)
    global_wer = compute_wer(global_ref, global_hyp)
    global_cer = compute_cer(global_ref, global_hyp)

    # --- cpWER ---
    cpwer_result = compute_cpwer(ref_segments, hyp_segments)

    # --- DER (requires pyannote.metrics) ---
    der_result = compute_der(ref_segments, hyp_segments)

    # --- Per-speaker WER ---
    per_speaker = {}
    if segmentation == "oracle":
        for spk in ref_speakers:
            spk_ref = " ".join(per_speaker_ref[spk])
            spk_hyp = " ".join(per_speaker_hyp[spk])
            spk_wer = compute_wer(spk_ref, spk_hyp)
            spk_cer = compute_cer(spk_ref, spk_hyp)
            per_speaker[spk] = {
                "wer": spk_wer,
                "cer": spk_cer,
                "ref_segments": len(per_speaker_ref[spk]),
            }
    else:
        details = cpwer_result.get("per_speaker_details", {})
        for ref_spk in ref_speakers:
            spk_detail = details.get(ref_spk)
            if spk_detail:
                per_speaker[ref_spk] = {
                    "wer": spk_detail["wer"],
                    "cer": spk_detail["cer"],
                    "ref_segments": spk_detail["ref_segments"],
                    "mapped_to": spk_detail["mapped_to"],
                }
            else:
                per_speaker[ref_spk] = {
                    "wer": WERResult(0, 0, 0, 0, 0),
                    "cer": CERResult(0, 0, 0, 0, 0),
                    "ref_segments": len([s for s in ref_segments if s.speaker == ref_spk]),
                    "mapped_to": "[None]",
                }

    result = SessionResult(
        session=session,
        array=array,
        mode=engine.__class__.__name__,
        duration_sec=total_duration,
        processing_time_sec=processing_time,
        num_ref_segments=len(ref_segments),
        num_hyp_segments=len(hyp_segments),
        ref_speakers=ref_speakers,
        hyp_speakers=sorted(set(s.speaker for s in hyp_segments)),
        global_wer=global_wer,
        global_cer=global_cer,
        cpwer_result=cpwer_result,
        der_result=der_result,
        per_speaker=per_speaker,
    )

    return result



# ===================================================================
#  Section 6 — Report Printing
# ===================================================================

def print_session_report(result: SessionResult, log_fn=print):
    """Print a formatted report for a single session."""
    rtf = result.processing_time_sec / max(result.duration_sec, 1)

    log_fn(f"\n{'=' * 70}")
    log_fn(f"  REPORT: {result.session} | Array: {result.array}")
    log_fn(f"{'=' * 70}")
    log_fn(f"  Mode:           {result.mode}")
    log_fn(f"  Duration:       {result.duration_sec:.1f}s ({result.duration_sec / 60:.1f} min)")
    log_fn(f"  Processing:     {result.processing_time_sec:.1f}s (RTF: {rtf:.3f}x)")
    log_fn(f"  Ref Segments:   {result.num_ref_segments}")
    log_fn(f"  Ref Speakers:   {', '.join(result.ref_speakers)}")

    log_fn(f"\n  {'─' * 50}")
    log_fn(f"  ASR Metrics")
    log_fn(f"  {'─' * 50}")
    w = result.global_wer
    c = result.global_cer
    log_fn(f"  Global WER:     {w.wer:.2f}%  (S={w.substitutions} I={w.insertions} D={w.deletions} / {w.ref_words} words)")
    log_fn(f"  Global CER:     {c.cer:.2f}%  ({c.ref_chars} chars)")

    cp = result.cpwer_result
    log_fn(f"  cpWER:          {cp['cpwer']:.2f}%")
    if cp.get("mapping"):
        log_fn(f"  Speaker Map:    {cp['mapping']}")

    if result.der_result:
        d = result.der_result
        log_fn(f"\n  {'─' * 50}")
        log_fn(f"  Diarization Metrics")
        log_fn(f"  {'─' * 50}")
        log_fn(f"  DER:            {d['der']:.2f}%")
        log_fn(f"    False Alarm:  {d['false_alarm']:.2f}%")
        log_fn(f"    Missed:       {d['missed']:.2f}%")
        log_fn(f"    Confusion:    {d['confusion']:.2f}%")

    log_fn(f"\n  {'─' * 55}")
    log_fn(f"  Per-Speaker Breakdown")
    log_fn(f"  {'─' * 55}")
    log_fn(f"  {'Speaker':<15} {'WER':>8} {'CER':>8} {'Sub':>6} {'Ins':>6} {'Del':>6} {'Segs':>6}")
    log_fn(f"  {'─' * 57}")

    for spk in sorted(result.per_speaker.keys()):
        info = result.per_speaker[spk]
        sw = info["wer"]
        sc = info["cer"]
        spk_display = spk
        if "mapped_to" in info and info["mapped_to"] != spk:
            spk_display = f"{spk}->{info['mapped_to']}"
        log_fn(
            f"  {spk_display:<15} {sw.wer:>7.1f}% {sc.cer:>7.1f}% "
            f"{sw.substitutions:>6} {sw.insertions:>6} {sw.deletions:>6} "
            f"{info['ref_segments']:>6}"
        )

    log_fn(f"{'=' * 70}\n")


def print_summary(results: list[SessionResult], log_fn=print):
    """Print a combined summary across all sessions."""
    total_duration = sum(r.duration_sec for r in results)
    total_proc = sum(r.processing_time_sec for r in results)
    total_ref_words = sum(r.global_wer.ref_words for r in results)
    total_errors = sum(r.global_wer.errors for r in results)
    total_ref_chars = sum(r.global_cer.ref_chars for r in results)
    total_char_errors = sum(r.global_cer.errors for r in results)

    avg_wer = total_errors / max(total_ref_words, 1) * 100.0
    avg_cer = total_char_errors / max(total_ref_chars, 1) * 100.0
    avg_cpwer = np.mean([r.cpwer_result["cpwer"] for r in results])
    avg_rtf = total_proc / max(total_duration, 1)

    log_fn(f"\n{'=' * 70}")
    log_fn(f"  GLOBAL SUMMARY ({len(results)} session(s))")
    log_fn(f"{'=' * 70}")
    log_fn(f"  Total Duration:     {total_duration:.1f}s ({total_duration / 60:.1f} min)")
    log_fn(f"  Total Processing:   {total_proc:.1f}s (RTF: {avg_rtf:.3f}x)")
    log_fn(f"  Aggregate WER:      {avg_wer:.2f}%")
    log_fn(f"  Aggregate CER:      {avg_cer:.2f}%")
    log_fn(f"  Average cpWER:      {avg_cpwer:.2f}%")

    der_results = [r.der_result for r in results if r.der_result]
    if der_results:
        avg_der = np.mean([d["der"] for d in der_results])
        avg_fa = np.mean([d["false_alarm"] for d in der_results])
        avg_miss = np.mean([d["missed"] for d in der_results])
        avg_conf = np.mean([d["confusion"] for d in der_results])
        log_fn(f"  Average DER:        {avg_der:.2f}%")
        log_fn(f"    False Alarm:      {avg_fa:.2f}%")
        log_fn(f"    Missed:           {avg_miss:.2f}%")
        log_fn(f"    Confusion:        {avg_conf:.2f}%")

    log_fn(f"{'=' * 70}\n")


# ===================================================================
#  Section 7 — Result Serialization
# ===================================================================

def _wer_to_dict(w: WERResult) -> dict:
    return {
        "wer": round(w.wer, 4),
        "ref_words": w.ref_words,
        "hyp_words": w.hyp_words,
        "substitutions": w.substitutions,
        "insertions": w.insertions,
        "deletions": w.deletions,
    }


def _cer_to_dict(c: CERResult) -> dict:
    return {
        "cer": round(c.cer, 4),
        "ref_chars": c.ref_chars,
        "hyp_chars": c.hyp_chars,
        "substitutions": c.substitutions,
        "insertions": c.insertions,
        "deletions": c.deletions,
    }


def results_to_json(results: list[SessionResult]) -> dict:
    """Serialize all session results to a JSON-compatible dict."""
    sessions = []
    for r in results:
        per_spk = {}
        for spk, info in r.per_speaker.items():
            per_spk[spk] = {
                "wer": _wer_to_dict(info["wer"]),
                "cer": _cer_to_dict(info["cer"]),
                "ref_segments": info["ref_segments"],
            }

        entry = {
            "session": r.session,
            "array": r.array,
            "mode": r.mode,
            "duration_sec": round(r.duration_sec, 2),
            "processing_time_sec": round(r.processing_time_sec, 2),
            "rtf": round(r.processing_time_sec / max(r.duration_sec, 1), 4),
            "num_ref_segments": r.num_ref_segments,
            "ref_speakers": r.ref_speakers,
            "global_wer": _wer_to_dict(r.global_wer),
            "global_cer": _cer_to_dict(r.global_cer),
            "cpwer": round(r.cpwer_result["cpwer"], 4),
            "cpwer_mapping": {k: v for k, v in r.cpwer_result.get("mapping", {}).items()},
            "der": r.der_result,
            "per_speaker": per_spk,
        }
        sessions.append(entry)

    # Compute summary
    total_ref_words = sum(r.global_wer.ref_words for r in results)
    total_errors = sum(r.global_wer.errors for r in results)

    summary = {
        "num_sessions": len(results),
        "total_duration_sec": round(sum(r.duration_sec for r in results), 2),
        "aggregate_wer": round(total_errors / max(total_ref_words, 1) * 100, 4),
        "average_cpwer": round(float(np.mean([r.cpwer_result["cpwer"] for r in results])), 4),
    }

    der_vals = [r.der_result for r in results if r.der_result]
    if der_vals:
        summary["average_der"] = round(float(np.mean([d["der"] for d in der_vals])), 4)

    return {"summary": summary, "sessions": sessions}


def save_report(results: list[SessionResult], path: str):
    """Save full results to a JSON file."""
    data = results_to_json(results)
    data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {out_path.resolve()}")


# ===================================================================
#  Section 8 — CLI
# ===================================================================

def discover_sessions(data_dir: Path) -> list[str]:
    """Find available session IDs from transcription files."""
    trans_dir = data_dir / "transcriptions"
    if not trans_dir.exists():
        return []
    return sorted(
        p.stem for p in trans_dir.glob("S*.json")
    )


def main():
    parser = argparse.ArgumentParser(
        description="CHiME-6 Benchmark & Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/chime6_benchmark_new.py --session S02 --limit-minutes 5
  python tests/chime6_benchmark_new.py --session S02 --mode aiworker
  python tests/chime6_benchmark_new.py --save-report output/results.json
  python tests/chime6_benchmark_new.py --session S02 --array U01,U02
        """,
    )

    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to CHiME-6 data directory (default: tests/chime6_data)",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Session ID to evaluate (e.g., S02). Default: all available sessions.",
    )
    parser.add_argument(
        "--array", type=str, default="U01",
        help="Microphone array(s) to use (default: U01). Comma-separated for multiple.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["raw", "aiworker"], default="raw",
        help="ASR engine: 'raw' (direct Whisper) or 'aiworker' (full pipeline). Default: raw.",
    )
    parser.add_argument(
        "--limit-minutes", type=float, default=None,
        help="Limit evaluation to first N minutes of each session.",
    )
    parser.add_argument(
        "--save-report", type=str, default=None,
        help="Save JSON report to the specified path.",
    )
    parser.add_argument(
        "--collar", type=float, default=0.25,
        help="DER collar in seconds (default: 0.25).",
    )
    parser.add_argument(
        "--segmentation", type=str, choices=["oracle", "streaming"], default="oracle",
        help="Segmentation mode: 'oracle' (ground-truth segment boundaries) or 'streaming' (continuous VAD/diarization). Default: oracle.",
    )

    args = parser.parse_args()

    # Resolve data directory
    data_dir = Path(args.data_dir) if args.data_dir else _PROJECT_ROOT / "tests" / "chime6_data"
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    # Discover sessions
    available = discover_sessions(data_dir)
    if not available:
        print(f"ERROR: No transcription files found in {data_dir / 'transcriptions'}")
        return 1

    if args.session:
        sessions = [s.strip() for s in args.session.split(",")]
        for s in sessions:
            if s not in available:
                print(f"ERROR: Session {s} not found. Available: {available}")
                return 1
    else:
        sessions = available

    arrays = [a.strip() for a in args.array.split(",")]

    # Print header
    print(f"\n{'=' * 70}")
    print(f"  CHiME-6 BENCHMARK & EVALUATION")
    print(f"{'=' * 70}")
    print(f"  Mode:          {args.mode}")
    print(f"  Segmentation:  {args.segmentation}")
    print(f"  Sessions:      {', '.join(sessions)}")
    print(f"  Arrays:        {', '.join(arrays)}")
    if args.limit_minutes:
        print(f"  Time Limit:    {args.limit_minutes:.1f} minutes per session")
    print(f"  DER Collar:    {args.collar:.2f}s")
    print(f"{'=' * 70}")

    # Initialize engine
    print(f"\n  Loading ASR engine ({args.mode})...")
    if args.mode == "aiworker":
        engine = AIWorkerEngine()
    else:
        engine = RawWhisperEngine()

    # Run evaluation
    all_results: list[SessionResult] = []
    total_t0 = time.time()

    for session in sessions:
        for array in arrays:
            audio_path = resolve_audio_path(data_dir, session, array)
            if not audio_path.exists():
                print(f"\n  SKIP: {session}_{array} — audio not found")
                continue

            try:
                result = evaluate_session(
                    data_dir=data_dir,
                    session=session,
                    array=array,
                    engine=engine,
                    segmentation=args.segmentation,
                    limit_minutes=args.limit_minutes,
                )
                all_results.append(result)
                print_session_report(result)
            except Exception as e:
                print(f"\n  ERROR evaluating {session}_{array}: {e}")
                import traceback
                traceback.print_exc()

    if not all_results:
        print("\n  No sessions were evaluated successfully.")
        return 1

    # Summary
    print_summary(all_results)

    total_elapsed = time.time() - total_t0
    print(f"  Total wall time: {total_elapsed:.1f}s")

    # Save report
    if args.save_report:
        save_report(all_results, args.save_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
