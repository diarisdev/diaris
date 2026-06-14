"""
CHiME6 Dataset Evaluation and Benchmarking Tool.

This script runs the audio-process system on CHiME6 sessions, comparing outputs
with ground-truth transcripts to calculate key conversational speech metrics:
  - WER (Word Error Rate) & CER (Character Error Rate)
  - DER (Diarization Error Rate) using pyannote.metrics
  - cpWER (concatenated minimum-permutation Word Error Rate)

Supports multiple modes:
  - Close-talk (worn) microphones evaluation (ideal for system sanity checks)
  - Far-field Kinect array evaluation (single channels or multi-channel combinations)
  - Oracle segmentation (Track 1) vs Continuous Streaming (Track 2)
  - Multi-channel enhancement (Average combination, Delay-and-Sum beamform, GSS)

==============================================================================
HOW TO RUN (terminal examples)
==============================================================================
Run as a module from the project root. Data is expected under
datasets/chime6/audio/ and datasets/chime6/transcriptions/ by default.

  # Fastest sanity check — 1 minute, close-talk, oracle segmentation:
  python -m tests.benchmarks.chime6 --sanity-check

  # Full close-talk (worn) Track-1 run over ALL sessions, oracle boundaries:
  python -m tests.benchmarks.chime6 --mode worn --segmentation oracle

  # One session only, first 5 minutes (quick iteration while developing):
  python -m tests.benchmarks.chime6 --session S02 --limit-minutes 5

  # A single worn speaker in one session:
  python -m tests.benchmarks.chime6 --session S02 --mode worn --worn-speaker P05

  # Far-field, single Kinect array U01, single channel (no enhancement):
  python -m tests.benchmarks.chime6 --mode far-field --array U01 --enhance none

  # Far-field, multi-channel averaging of array U01, streaming (Track 2):
  python -m tests.benchmarks.chime6 --mode far-field --array U01 \
      --enhance average --segmentation streaming

  # Far-field, Delay-and-Sum beamforming over two arrays:
  python -m tests.benchmarks.chime6 --mode far-field --array U01,U02 --enhance beamform

  # Far-field with pre-generated GSS audio (recommended for paper-comparable
  # far-field numbers — see github.com/desh2608/gss):
  python -m tests.benchmarks.chime6 --mode far-field --array U01 --enhance gss \
      --segmentation streaming

  # Paper-comparable cpWER via the meeteval reference implementation, save JSON:
  python -m tests.benchmarks.chime6 --session S02 --meeteval --output output/s02.json

  # Custom data locations + explicit log file + precise WER (jiwer):
  python -m tests.benchmarks.chime6 --audio-dir D:/chime6/audio \
      --trans-dir D:/chime6/transcriptions --jiwer --log output/run.log

------------------------------------------------------------------------------
ARGUMENT REFERENCE
------------------------------------------------------------------------------
  --session, -s        Session ID to evaluate (e.g. S02). Omit = all sessions.
  --mode, -m           'worn' (close-talk mics) | 'far-field' (Kinect arrays).
  --array              Far-field array(s): 'U01', 'U01,U02', or 'all'.
  --enhance            'none' (single CH) | 'average' (multi-CH mean) |
                       'beamform' (delay-and-sum) | 'gss' (pre-enhanced GSS).
  --segmentation       'oracle' (Track 1, ground-truth boundaries) |
                       'streaming' (Track 2, VAD + diarization + ASR).
  --worn-speaker       Limit worn eval to one speaker ID (e.g. P05).
  --limit-minutes, -l  Process only first N minutes of each session.
  --jiwer              Use jiwer for precise WER (if installed).
  --meeteval           Use meeteval's reference cpWER (pip install meeteval) so
                       the cpWER number is directly comparable to CHiME-6/7/8.
  --output, -o         Save full evaluation report to a JSON file.
  --log                Path for the execution text log (default: output/...).
  --audio-dir, -a      Override CHiME6 audio directory.
  --trans-dir, -t      Override CHiME6 JSON transcripts directory.
  --sanity-check       Shorthand for worn + oracle + 1 minute.

Notes:
  * Track 1 (oracle) headline metric is WER; Track 2 (streaming) is cpWER.
  * Far-field without GSS ('none'/'average'/'beamform') does NOT separate
    overlapping speakers and scores far worse than the ~51% WER CHiME-6
    baseline; use '--enhance gss' for results comparable to the literature.
  * For Track 2, run FULL sessions so the diarization warm-up is negligible;
    short clips inflate cpWER/DER.
"""
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=r".*std\(\).*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")

import gc
import os
import sys
import time
import json
import argparse
import re
import itertools
import numpy as np
import soundfile as sf
import torch
import torchaudio
import concurrent.futures
from pathlib import Path

# Add project root to sys.path (tests/benchmarks/chime6.py -> 3 seviye yukarı)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure CUDA DLLs are discoverable before any model loading
from src.config import configure_cuda_dll_paths
configure_cuda_dll_paths()

from tests.metrics import TranscriptionEvaluator, DiarizationEvaluator, cpwer_from_segments
from tests.dataset_managers import (
    Chime6DatasetManager,
    parse_time_to_seconds,
    clean_chime6_text,
)
from src.audio.vad import VADEngine
from src.core.ai_worker import AIWorker
from src.config import (
    FRAME_DURATION_MS,
    SILENCE_LIMIT,
    SHORT_SILENCE_LIMIT,
    SOFT_CHUNK_DURATION_MS,
    MAX_CHUNK_DURATION_MS,
)


def print_and_log(msg, log_file=None):
    """Prints to console (ASCII-safe) and logs to file if provided."""
    # Clean non-ASCII characters for console print if encoding is not UTF-8
    try:
        print(msg)
    except UnicodeEncodeError:
        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg)
        
    if log_file:
        log_file.write(str(msg) + "\n")
        log_file.flush()


def _get_process_memory_mb():
    """Returns current process RSS memory in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for Windows without psutil
        try:
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {os.getpid()}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            # Parse CSV: "python.exe","1234","Console","1","123,456 K"
            parts = result.stdout.strip().split(',')
            if len(parts) >= 5:
                mem_str = parts[-1].strip().strip('"').replace(',', '').replace('.', '').split()[0]
                return int(mem_str) / 1024  # KB to MB
        except Exception:
            pass
    return 0.0


def delay_and_sum_beamforming(waveforms, sr=16000):
    """
    Applies Delay-and-Sum Beamforming to align and sum multi-channel waveforms.
    Uses cross-correlation to estimate time delay of arrival (TDOA) relative to channel 0.
    """
    if len(waveforms) <= 1:
        return waveforms[0]
        
    ref = waveforms[0]
    aligned_waveforms = [ref]
    
    # Max delay search range (e.g., 5ms, which is 80 samples at 16kHz)
    max_delay = int(0.005 * sr)
    
    for i in range(1, len(waveforms)):
        sig = waveforms[i]
        
        # Calculate cross-correlation using a representative 1-second high-energy slice
        # if the chunk is too long, to speed up computation.
        if len(ref) > sr:
            win_size = int(sr)
            step = int(sr // 2)
            best_energy = -1
            best_idx = 0
            for start in range(0, len(ref) - win_size, step):
                energy = np.sum(ref[start:start+win_size]**2)
                if energy > best_energy:
                    best_energy = energy
                    best_idx = start
            ref_seg = ref[best_idx:best_idx+win_size]
            sig_seg = sig[best_idx:best_idx+win_size]
        else:
            ref_seg = ref
            sig_seg = sig
            
        if len(ref_seg) == 0 or len(sig_seg) == 0:
            aligned_waveforms.append(sig)
            continue
            
        correlation = np.correlate(ref_seg, sig_seg, mode='same')
        center = len(correlation) // 2
        search_start = max(0, center - max_delay)
        search_end = min(len(correlation), center + max_delay + 1)
        
        if search_start >= search_end:
            aligned_waveforms.append(sig)
            continue
            
        best_shift = np.argmax(correlation[search_start:search_end]) - (center - search_start)
        
        if best_shift > 0:
            shifted = np.pad(sig[best_shift:], (0, best_shift), mode='constant')
        elif best_shift < 0:
            shifted = np.pad(sig[:best_shift], (-best_shift, 0), mode='constant')
        else:
            shifted = sig
            
        aligned_waveforms.append(shifted)
        
    return np.mean(aligned_waveforms, axis=0).astype(np.int16)


def load_chime6_transcript(json_path):
    """
    Loads CHiME6 ground truth transcript from JSON.
    Expected format: a list of utterances, each containing speaker, start_time, end_time, and words.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    segments = []
    for item in data:
        start_raw = item.get("start_time")
        end_raw = item.get("end_time")
        if start_raw is None or end_raw is None:
            continue
            
        try:
            start = parse_time_to_seconds(start_raw)
            end = parse_time_to_seconds(end_raw)
        except Exception:
            continue
            
        speaker = item.get("speaker", "Unknown")
        raw_text = item.get("words") or item.get("text") or ""
        clean_text = clean_chime6_text(raw_text)
        
        # Skip purely non-speech/empty segments
        if not clean_text:
            continue
            
        segments.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": clean_text
        })
        
    # Sort chronologically by start time
    segments.sort(key=lambda x: x["start"])
    return segments


def truncate_segments(segments, limit_seconds):
    """Truncates reference segments to a maximum duration for partial evaluation."""
    truncated = []
    for seg in segments:
        if seg["start"] >= limit_seconds:
            continue
        new_seg = dict(seg)
        if new_seg["end"] > limit_seconds:
            new_seg["end"] = limit_seconds
        if new_seg["start"] < new_seg["end"]:
            truncated.append(new_seg)
    return truncated


def get_audio_paths(audio_dir, session_id, mode, array="U01", enhance="none", worn_speaker=None):
    """
    Finds and returns paths to audio files matching the requested configuration.
    """
    audio_dir = Path(audio_dir)
    
    if mode == "worn":
        if worn_speaker:
            # Look for a specific speaker file: e.g. S01_P03.wav
            path = audio_dir / f"{session_id}_{worn_speaker}.wav"
            if path.exists():
                return [path]
            candidates = list(audio_dir.glob(f"**/{session_id}_{worn_speaker}.wav"))
            if candidates:
                return [candidates[0]]
            raise FileNotFoundError(f"Worn audio file not found for speaker {worn_speaker} in session {session_id} in {audio_dir}")
        else:
            # Find all worn speaker files: S01_P*.wav
            candidates = sorted(list(audio_dir.glob(f"**/{session_id}_P*.wav")))
            if not candidates:
                # Direct check
                for i in range(1, 100):
                    p = audio_dir / f"{session_id}_P{i:02d}.wav"
                    if p.exists():
                        candidates.append(p)
            if not candidates:
                raise FileNotFoundError(f"No worn audio files found for session {session_id} in {audio_dir}")
            return candidates
            
    else:  # far-field
        # Parse arrays
        arrays = []
        if array == "all":
            # Find all arrays for this session in the audio_dir
            pattern = f"{session_id}_U*.CH1.wav"
            matches = list(audio_dir.glob(pattern))
            if not matches:
                matches = list(audio_dir.glob(f"**/{pattern}"))
            for m in matches:
                part = m.name.split('_')[1]  # e.g. "U01.CH1.wav"
                arr_id = part.split('.')[0]  # e.g. "U01"
                if arr_id.startswith('U') and len(arr_id) == 3:
                    arrays.append(arr_id)
            arrays = sorted(list(set(arrays)))
            if not arrays:
                arrays = ["U01"]
        else:
            arrays = [a.strip() for a in array.split(',') if a.strip()]
            
        if enhance == "gss":
            # First look for GSS directory containing enhanced segment files (generated by desh2608/gss)
            gss_dir = audio_dir / "gss"
            if gss_dir.exists() and gss_dir.is_dir():
                return [gss_dir]
                
            # Try to look for pre-enhanced single GSS files: e.g. S01_U01.GSS.wav or S01_GSS.wav
            for arr in arrays:
                candidates = [
                    audio_dir / f"{session_id}_{arr}.GSS.wav",
                    audio_dir / f"{session_id}_{arr}_GSS.wav",
                    audio_dir / f"{session_id}_GSS.wav"
                ]
                for c in candidates:
                    if c.exists():
                        return [c]
                # Recursively find any GSS wav
                found = list(audio_dir.glob(f"**/{session_id}*{arr}*GSS*.wav")) + list(audio_dir.glob(f"**/{session_id}*GSS*.wav"))
                if found:
                    return [found[0]]
            print(f"WARNING: Pre-enhanced GSS audio not found for {session_id} {arrays}. Falling back to 'beamform' combination.")
            enhance = "beamform"
            
        if enhance in ("average", "beamform"):
            # Multi-channel combination: return CH1, CH2, CH3, CH4 of the array(s)
            channels = []
            for arr in arrays:
                for ch in range(1, 5):
                    path = audio_dir / f"{session_id}_{arr}.CH{ch}.wav"
                    if not path.exists():
                        candidates = list(audio_dir.glob(f"**/{session_id}_{arr}.CH{ch}.wav"))
                        if candidates:
                            path = candidates[0]
                    channels.append(path)
                
            existing = [p for p in channels if p.exists()]
            if not existing:
                raise FileNotFoundError(f"No multi-channel array files found for arrays {arrays} in session {session_id} in {audio_dir}")
            return existing
            
        else:  # none - single channel
            first_arr = arrays[0]
            path = audio_dir / f"{session_id}_{first_arr}.CH1.wav"
            if not path.exists():
                candidates = []
                for ch in range(1, 5):
                    candidates.extend(list(audio_dir.glob(f"**/{session_id}_{first_arr}.CH{ch}.wav")))
                if candidates:
                    path = candidates[0]
            if not path.exists():
                raise FileNotFoundError(f"Single channel far-field audio file not found for {session_id} {first_arr} in {audio_dir}")
            return [path]


def process_audio_segment(audio_paths, start_time, end_time, mode, enhance, speaker_id=None):
    """
    Extracts and combines an audio segment [start_time, end_time] from source file(s)
    and returns a resampled mono float32 numpy array at 16kHz.
    """
    waveforms = []
    sr = None
    
    # If using GSS directory of segment files, find the matching segment wav file
    if enhance == "gss" and speaker_id and len(audio_paths) == 1 and Path(audio_paths[0]).is_dir():
        gss_dir = Path(audio_paths[0])
        # Find files matching the speaker pattern
        pattern = f"*-{speaker_id}-*.wav"
        candidates = list(gss_dir.glob(pattern))
        if not candidates:
            # Fallback to general matching
            candidates = list(gss_dir.glob(f"*{speaker_id}*.wav"))
            
        best_file = None
        min_diff = float('inf')
        for cand in candidates:
            # Example filename: S01_U01-P01-002118_002530.wav or similar
            # Extract start time (last part, split by underscore/dash)
            name_parts = cand.stem.split('-')
            if len(name_parts) >= 3:
                time_part = name_parts[-1]
                time_subparts = time_part.split('_')
                if len(time_subparts) == 2:
                    try:
                        cand_start = float(time_subparts[0]) / 100.0
                        diff = abs(cand_start - start_time)
                        if diff < min_diff and diff < 1.0: # threshold of 1.0s
                            min_diff = diff
                            best_file = cand
                    except ValueError:
                        pass
        
        if best_file:
            try:
                with sf.SoundFile(str(best_file)) as f:
                    sr = f.samplerate
                    data = f.read(dtype='int16')
                    if len(data) > 0:
                        if data.ndim > 1:
                            data = data.mean(axis=1).astype(np.int16)
                        waveforms.append(data)
            except Exception as e:
                print(f"WARNING: Error reading GSS segment file {best_file}: {e}")
                
    # In worn mode, if we are evaluating a specific segment, we only read from the speaker's worn mic
    target_paths = audio_paths
    if not waveforms:
        if mode == "worn":
            if speaker_id and len(audio_paths) > 1:
                # Filter audio paths matching the speaker (e.g. S01_P01.wav matches P01)
                filtered = [p for p in audio_paths if speaker_id in p.name]
                if filtered:
                    target_paths = filtered
                else:
                    # Fallback to the first path instead of averaging all worn microphones
                    target_paths = [audio_paths[0]]
            else:
                target_paths = [audio_paths[0]]
                
        for path in target_paths:
            try:
                with sf.SoundFile(str(path)) as f:
                    sr = f.samplerate
                    total_frames = len(f)
                    start_frame = int(start_time * sr)
                    end_frame = int(end_time * sr)
                    
                    # Boundary checks
                    if start_frame < 0:
                        start_frame = 0
                    if start_frame >= total_frames:
                        continue
                        
                    num_frames = end_frame - start_frame
                    if num_frames <= 0:
                        continue
                    if start_frame + num_frames > total_frames:
                        num_frames = total_frames - start_frame
                        
                    f.seek(start_frame)
                    data = f.read(num_frames, dtype='int16')
                    if len(data) == 0:
                        continue
                    if data.ndim > 1:
                        data = data.mean(axis=1).astype(np.int16)
                    waveforms.append(data)
            except Exception:
                # Fail silently for this path
                continue
            
    if not waveforms:
        return None, None
        
    if len(waveforms) > 1:
        # Multi-channel combination
        # Pad shorter chunks to match size
        max_len = max(len(w) for w in waveforms)
        padded = []
        for w in waveforms:
            if len(w) < max_len:
                padded.append(np.pad(w, (0, max_len - len(w))))
            else:
                padded.append(w)
                
        if enhance == "beamform":
            combined = delay_and_sum_beamforming(padded, sr=sr)
        else:
            combined = np.mean(padded, axis=0).astype(np.int16)
    else:
        combined = waveforms[0]
        
    # Convert to mono float32
    mono_f32 = combined.astype(np.float32) / 32768.0
    del combined, waveforms  # Free intermediate arrays
    
    # Resample to 16kHz using torchaudio
    waveform_tensor = torch.from_numpy(mono_f32).unsqueeze(0)
    del mono_f32
    if sr != 16000:
        waveform_tensor = torchaudio.functional.resample(waveform_tensor, orig_freq=sr, new_freq=16000)
    
    result = waveform_tensor.squeeze(0).numpy()
    del waveform_tensor
    return result, 16000


def run_oracle_evaluation(ai_worker, ref_segments, audio_paths, mode, enhance, log_file=None):
    """
    Runs Oracle Segment-by-Segment evaluation (Track 1).
    Reads each utterance boundary from the transcript, transcribes it, and compares.
    Uses ThreadPoolExecutor to overlap CPU audio extraction with GPU inference.
    """
    print_and_log(f"\n[EVAL] Running Oracle Segmentation Evaluation on {len(ref_segments)} segments...", log_file)
    results = [None] * len(ref_segments)
    start_eval_time = time.time()
    
    def process_single_segment(idx, seg):
        spk = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        
        # 1. Process and load audio segment (CPU-bound)
        audio_segment, sr = process_audio_segment(
            audio_paths=audio_paths,
            start_time=start,
            end_time=end,
            mode=mode,
            enhance=enhance,
            speaker_id=spk
        )
        
        if audio_segment is None or len(audio_segment) == 0:
            return idx, {"start": start, "end": end, "speaker": spk, "text": ""}
            
        # 2. Run Whisper on the segment (GPU-bound)
        try:
            segments, _ = ai_worker.transcriber.transcribe(
                audio_segment,
                beam_size=3,
                language="en",
                condition_on_previous_text=False  # isolated segments to prevent context leaks
            )
            hyp_text = " ".join(s.text.strip() for s in segments)
            hyp_text = clean_chime6_text(hyp_text)
        except Exception as e:
            print(f"\nWARNING: Error transcribing segment {idx+1}: {e}")
            hyp_text = ""
            
        return idx, {"start": start, "end": end, "speaker": spk, "text": hyp_text}

    completed = 0
    # Use max_workers=8 to ensure the GPU is constantly fed with pre-processed audio chunks
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(process_single_segment, idx, seg): idx 
            for idx, seg in enumerate(ref_segments)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, result_seg = future.result()
                results[idx] = result_seg
            except Exception as e:
                print(f"\nWARNING: Unhandled exception in segment {idx+1}: {e}")
                results[idx] = {"start": ref_segments[idx]["start"], "end": ref_segments[idx]["end"], "speaker": ref_segments[idx]["speaker"], "text": ""}
                
            completed += 1
            
            # Periodic memory cleanup to prevent OOM on long sessions
            if completed % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Dynamic Progress Bar
            elapsed = time.time() - start_eval_time
            progress = completed / len(ref_segments)
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '-' * (bar_len - filled)
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(ref_segments) - completed) / rate if rate > 0 else 0
            mem_mb = _get_process_memory_mb()
            
            sys.stdout.write(f"\r   [{bar}] {completed}/{len(ref_segments)} ({progress*100:.1f}%) | {rate:.1f} seg/s | ETA: {eta:.0f}s | RAM: {mem_mb:.0f}MB")
            sys.stdout.flush()
            
    print()  # Move to the next line after the progress bar finishes
    return results


def run_streaming_evaluation(ai_worker, audio_paths, mode, enhance, limit_minutes=None, log_file=None):
    """
    Runs Streaming Continuous VAD + Diarization + ASR evaluation (Track 2).
    Streams audio chunk-by-chunk to simulate real-time processing.
    """
    print_and_log(f"\n[EVAL] Running Continuous Streaming Evaluation...", log_file)
    
    hyp_segments = []
    
    if mode == "worn":
        # Process each close-talk worn mic independently
        for path in audio_paths:
            match = re.search(r'P\d+', Path(path).name)
            worn_spk_id = match.group(0) if match else "Unknown"
            
            print_and_log(f"   Streaming close-talk mic for speaker {worn_spk_id} ({Path(path).name})...", log_file)
            
            # Reset worker state for this speaker
            ai_worker.speaker_tracker.reset()
            
            with sf.SoundFile(str(path)) as sf_file:
                rate = sf_file.samplerate
                ai_worker.rate = rate
                frames_per_chunk = int(rate * (FRAME_DURATION_MS / 1000.0))
                limit_frames = int(rate * limit_minutes * 60.0) if limit_minutes else None
                
                vad_engine = VADEngine()
                chunk_buffer = []
                silence_counter = 0
                has_spoken = False
                global_time_s = 0.0
                chunk_start_s = 0.0
                total_frames_read = 0
                
                while True:
                    if limit_frames and total_frames_read >= limit_frames:
                        break
                        
                    data = sf_file.read(frames_per_chunk, dtype='int16')
                    if len(data) == 0:
                        break
                    if data.ndim > 1:
                        data = data.mean(axis=1).astype(np.int16)
                    if len(data) < frames_per_chunk:
                        data = np.pad(data, (0, frames_per_chunk - len(data)))
                        
                    total_frames_read += len(data)
                    frame_bytes = data.tobytes()
                    
                    is_speech, conf = vad_engine.check_speech(frame_bytes, rate, 1)
                    
                    if is_speech:
                        chunk_buffer.append(frame_bytes)
                        silence_counter = 0
                        if not has_spoken:
                            has_spoken = True
                            chunk_start_s = global_time_s
                    else:
                        silence_bytes = b'\x00' * len(frame_bytes)
                        if has_spoken:
                            chunk_buffer.append(silence_bytes)
                            silence_counter += 1
                            
                    global_time_s += (FRAME_DURATION_MS / 1000.0)
                    current_duration_ms = len(chunk_buffer) * FRAME_DURATION_MS
                    active_silence_limit = SHORT_SILENCE_LIMIT if current_duration_ms > SOFT_CHUNK_DURATION_MS else SILENCE_LIMIT
                    
                    if has_spoken and (silence_counter > active_silence_limit or current_duration_ms >= MAX_CHUNK_DURATION_MS):
                        chunk_bytes_to_send = b''.join(chunk_buffer)
                        output = ai_worker.process_chunk(chunk_bytes_to_send, language="en")
                        results = output.get("results") if isinstance(output, dict) else output
                        
                        if results:
                            # Run background diarization
                            diarized_results = ai_worker.run_diarization(
                                output["waveform_16k"],
                                output["sample_rate"],
                                output["chunk_duration_ms"],
                                results
                            )
                            
                            # Convert to global timeline and collect
                            for r in diarized_results:
                                global_start = chunk_start_s + r["start"]
                                global_end = chunk_start_s + r["end"]
                                
                                # Tag hypothesis speaker name with worn speaker ID
                                hyp_segments.append({
                                    "start": global_start,
                                    "end": global_end,
                                    "speaker": f"{worn_spk_id}_{r['speaker']}",
                                    "text": clean_chime6_text(r["text"])
                                })
                                
                        chunk_buffer = []
                        silence_counter = 0
                        has_spoken = False

                        # Reset the Silero VAD RNN state between chunks. Each chunk
                        # is an independent utterance; leaking hidden state across
                        # boundaries skews VAD decisions. (Ported from chime6_benchmark_new.)
                        if hasattr(vad_engine.silero_model, "reset_states"):
                            vad_engine.silero_model.reset_states()

        # Sort all collected segments chronologically
        hyp_segments.sort(key=lambda x: x["start"])
        return hyp_segments
        
    else:  # far-field
        # Open all source files
        sound_files = [sf.SoundFile(str(p)) for p in audio_paths]
        rate = sound_files[0].samplerate
        
        # Reinitialize worker with active rate
        ai_worker.rate = rate
        ai_worker.speaker_tracker.reset()
        
        # Setup VAD and stream parameters
        vad_engine = VADEngine()
        frames_per_chunk = int(rate * (FRAME_DURATION_MS / 1000.0))
        limit_frames = int(rate * limit_minutes * 60.0) if limit_minutes else None
        
        chunk_buffer = []
        silence_counter = 0
        has_spoken = False
        
        global_time_s = 0.0
        chunk_start_s = 0.0
        total_frames_read = 0
        
        start_eval_time = time.time()
        
        def read_next_combined_frame():
            frame_waveforms = []
            for sf_file in sound_files:
                try:
                    data = sf_file.read(frames_per_chunk, dtype='int16')
                    if len(data) == 0:
                        continue
                    if data.ndim > 1:
                        data = data.mean(axis=1).astype(np.int16)
                    if len(data) < frames_per_chunk:
                        data = np.pad(data, (0, frames_per_chunk - len(data)))
                    frame_waveforms.append(data)
                except Exception:
                    continue
            if not frame_waveforms:
                return None
                
            if len(frame_waveforms) > 1:
                if enhance == "beamform":
                    return delay_and_sum_beamforming(frame_waveforms, sr=rate)
                else:
                    return np.mean(frame_waveforms, axis=0).astype(np.int16)
            else:
                return frame_waveforms[0]

        # Stream loop
        while True:
            if limit_frames and total_frames_read >= limit_frames:
                break
                
            combined_frame = read_next_combined_frame()
            if combined_frame is None:
                break
                
            total_frames_read += len(combined_frame)
            frame_bytes = combined_frame.tobytes()
            
            is_speech, conf = vad_engine.check_speech(frame_bytes, rate, 1)
            
            if is_speech:
                chunk_buffer.append(frame_bytes)
                silence_counter = 0
                if not has_spoken:
                    has_spoken = True
                    chunk_start_s = global_time_s
            else:
                silence_bytes = b'\x00' * len(frame_bytes)
                if has_spoken:
                    chunk_buffer.append(silence_bytes)
                    silence_counter += 1
                    
            global_time_s += (FRAME_DURATION_MS / 1000.0)
            
            current_duration_ms = len(chunk_buffer) * FRAME_DURATION_MS
            active_silence_limit = SHORT_SILENCE_LIMIT if current_duration_ms > SOFT_CHUNK_DURATION_MS else SILENCE_LIMIT
            
            if has_spoken and (silence_counter > active_silence_limit or current_duration_ms >= MAX_CHUNK_DURATION_MS):
                chunk_bytes_to_send = b''.join(chunk_buffer)
                output = ai_worker.process_chunk(chunk_bytes_to_send, language="en")
                results = output.get("results") if isinstance(output, dict) else output
                
                if results:
                    # Run background diarization
                    diarized_results = ai_worker.run_diarization(
                        output["waveform_16k"],
                        output["sample_rate"],
                        output["chunk_duration_ms"],
                        results
                    )
                    
                    # Convert to global timeline and collect
                    for r in diarized_results:
                        global_start = chunk_start_s + r["start"]
                        global_end = chunk_start_s + r["end"]
                        
                        hyp_segments.append({
                            "start": global_start,
                            "end": global_end,
                            "speaker": r["speaker"],
                            "text": clean_chime6_text(r["text"])
                        })
                        
                chunk_buffer = []
                silence_counter = 0
                has_spoken = False

                # Reset the Silero VAD RNN state between chunks (see worn path
                # above). Ported from chime6_benchmark_new.
                if hasattr(vad_engine.silero_model, "reset_states"):
                    vad_engine.silero_model.reset_states()

        # Close all files
        for sf_file in sound_files:
            sf_file.close()
            
        eval_elapsed = time.time() - start_eval_time
        actual_duration = global_time_s
        rtf = eval_elapsed / actual_duration if actual_duration > 0 else 0.0
        print_and_log(f"[OK] Processing complete in {eval_elapsed:.1f}s (RTF: {rtf:.3f}x)", log_file)
        
        return hyp_segments


def run_chime6_session(audio_dir, transcript_path, mode, array="U01", enhance="none",
                      segmentation="oracle", limit_minutes=None, use_jiwer=False,
                      worn_speaker=None, use_meeteval=False, log_file=None):
    """
    Evaluates a single CHiME6 session under selected parameters.
    """
    session_id = os.path.basename(transcript_path).split('.')[0]
    print_and_log(f"\n[SESSION] Session: {session_id}", log_file)
    print_and_log(f"   Mode:         {mode.upper()}", log_file)
    print_and_log(f"   Segmentation: {segmentation.upper()}", log_file)
    if mode == "far-field":
        print_and_log(f"   Array:        {array}", log_file)
        print_and_log(f"   Enhancement:  {enhance.upper()}", log_file)
        if enhance != "gss":
            print_and_log(
                "   [WARNING] Far-field without GSS enhancement. 'none', 'average' and "
                "'beamform' do NOT separate overlapping speakers and will score far worse "
                "than the CHiME-6 baseline (~51% WER), which uses GSS. Naive channel "
                "averaging can also be WORSE than a single channel. For results comparable "
                "to the literature, pre-generate GSS audio (e.g. github.com/desh2608/gss) "
                "and run with '--enhance gss'.", log_file
            )
        if array == "all" and enhance in ("average", "beamform"):
            print_and_log(
                "   [WARNING] Combining different arrays (different rooms) with naive "
                "average/beamform is acoustically invalid; use GSS for multi-array.", log_file
            )
    elif worn_speaker:
        print_and_log(f"   Worn Speaker: {worn_speaker}", log_file)
        
    # 1. Load ground truth transcript
    ref_segments = load_chime6_transcript(transcript_path)
    if not ref_segments:
        print_and_log(f"❌ Empty or invalid ground truth transcript at: {transcript_path}", log_file)
        return None
        
    # 2. Get audio paths
    try:
        audio_paths = get_audio_paths(audio_dir, session_id, mode, array, enhance, worn_speaker)
    except FileNotFoundError as e:
        print_and_log(f"❌ {e}", log_file)
        return None
        
    # Display info
    info = sf.info(str(audio_paths[0]))
    actual_duration = info.duration
    if limit_minutes:
        actual_duration = min(actual_duration, limit_minutes * 60.0)
        ref_segments = truncate_segments(ref_segments, actual_duration)
        
    print_and_log(f"   Audio Path(s): {[p.name for p in audio_paths]}", log_file)
    print_and_log(f"   Audio Info:    {info.samplerate} Hz | {info.channels} channels | {actual_duration / 60:.2f} mins evaluated", log_file)
    print_and_log(f"   Ground Truth:  {len(ref_segments)} speech segments loaded", log_file)

    # The speaker tracker has a calibration (warm-up) phase during which it emits
    # "[Calibrating...]" labels that get filtered out. On a full ~2.5h session this
    # is negligible, but on short clips it wastes a large fraction of the audio and
    # heavily inflates Track 2 metrics (cpWER/DER). Warn the user accordingly.
    if segmentation == "streaming":
        try:
            warmup_s = AIWorker().speaker_tracker.warmup_ms / 1000.0
        except Exception:
            warmup_s = 0.0
        if warmup_s > 0 and actual_duration < 5 * warmup_s:
            print_and_log(
                f"   [WARNING] Streaming warm-up is {warmup_s:.0f}s but only "
                f"{actual_duration:.0f}s of audio is evaluated. The calibration period "
                f"dominates and will inflate cpWER/DER. Run FULL sessions for Track 2 "
                f"(so warm-up is ~negligible), or lower DIARIZATION_WARMUP_MS for quick tests.",
                log_file
            )
    
    # 3. Load Models
    print_and_log("🧠 Loading AIWorker models...", log_file)
    ai_worker = AIWorker(rate=info.samplerate, channels=1)
    if not ai_worker.load_models():
        print_and_log("❌ Failed to load AIWorker models.", log_file)
        return None
        
    # 4. Processing
    start_eval_time = time.time()
    if segmentation == "oracle":
        hyp_segments = run_oracle_evaluation(
            ai_worker=ai_worker,
            ref_segments=ref_segments,
            audio_paths=audio_paths,
            mode=mode,
            enhance=enhance,
            log_file=log_file
        )
        rtf = (time.time() - start_eval_time) / actual_duration if actual_duration > 0 else 0.0
    else:  # streaming
        hyp_segments = run_streaming_evaluation(
            ai_worker=ai_worker,
            audio_paths=audio_paths,
            mode=mode,
            enhance=enhance,
            limit_minutes=limit_minutes,
            log_file=log_file
        )
        rtf = (time.time() - start_eval_time) / actual_duration if actual_duration > 0 else 0.0

    # Free GPU memory before heavy CPU-based evaluation
    del ai_worker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 5. Evaluate Performance
    print_and_log("[EVAL] Evaluating performance metrics...", log_file)
    trans_eval = TranscriptionEvaluator(use_jiwer=use_jiwer)
    
    # Global ASR Evaluation: Concatenate texts chronologically
    ref_all_text = " ".join(trans_eval.normalize_text(s["text"]) for s in ref_segments)
    
    # Ignore calibrating tags in hypothesis transcripts
    clean_hyp_segments = [s for s in hyp_segments if "Calibrating" not in s["speaker"]]
    hyp_all_text = " ".join(trans_eval.normalize_text(s["text"]) for s in clean_hyp_segments)
    
    asr_res = trans_eval.evaluate(
        reference=ref_all_text,
        hypothesis=hyp_all_text,
        audio_path=str(audio_paths[0]),
        duration=actual_duration
    )
    
    # cpWER Evaluation (ortak tests.metrics çekirdeği)
    _cp = cpwer_from_segments(ref_segments, clean_hyp_segments, use_meeteval=use_meeteval)
    cpwer, mapping, details = _cp.cpwer, _cp.mapping, _cp.details
    
    # Diarization DER Evaluation (Only applicable to streaming segmentation)
    der = 0.0
    false_alarm = 0.0
    missed_detection = 0.0
    confusion = 0.0
    
    if segmentation == "streaming":
        diar_eval = DiarizationEvaluator()

        # NOTE: CHiME-6 DER convention is collar=0 and overlap INCLUDED.
        # Make sure DiarizationEvaluator is configured that way (collar=0.0,
        # skip_overlap=False); otherwise these numbers are NOT comparable to
        # the CHiME-6 paper / dscore results.
        ref_diar = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in ref_segments]
        hyp_diar = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in clean_hyp_segments]

        # No warmup filtering: trimming reference speech makes DER non-standard
        # and not comparable to published CHiME-6 numbers. Score the full range.
        diar_res = diar_eval.evaluate(
            meeting_id=session_id,
            reference_intervals=ref_diar,
            hypothesis_intervals=hyp_diar,
            duration=max(0.1, actual_duration)
        )
        if diar_res:
            der = diar_res.der
            false_alarm = diar_res.false_alarm
            missed_detection = diar_res.missed_detection
            confusion = diar_res.confusion
            
    report = {
        "session_id": session_id,
        "audio_paths": [str(p) for p in audio_paths],
        "duration": actual_duration,
        "rtf": rtf,
        "mode": mode,
        "segmentation": segmentation,
        "enhance": enhance,
        "ref_speakers": sorted(list(set(s["speaker"] for s in ref_segments))),
        "hyp_speakers": sorted(list(set(s["speaker"] for s in clean_hyp_segments))),
        "wer": asr_res.wer,
        "cer": asr_res.cer,
        "der": der,
        "false_alarm": false_alarm,
        "missed_detection": missed_detection,
        "confusion": confusion,
        "cpwer": cpwer,
        "speaker_mapping": mapping,
        "speaker_details": details
    }
    
    return report


def print_session_report(report, log_file=None):
    """Prints a beautiful summary report for a single session."""
    if not report:
        return
        
    print_and_log("\n" + "=" * 70, log_file)
    print_and_log(f"[REPORT] CHiME-6 EVALUATION REPORT - SESSION: {report['session_id']}", log_file)
    print_and_log("=" * 70, log_file)
    print_and_log(f"   Config:          Mode: {report['mode'].upper()} | Segmentation: {report['segmentation'].upper()} | Enhance: {report['enhance'].upper()}", log_file)
    print_and_log(f"   Duration:        {report['duration']:.2f} seconds ({report['duration'] / 60:.1f} minutes)", log_file)
    print_and_log(f"   Processing RTF:  {report['rtf']:.3f}x", log_file)
    print_and_log(f"   Ref Speakers:    {len(report['ref_speakers'])} ({', '.join(report['ref_speakers'])})", log_file)
    print_and_log(f"   Hyp Speakers:    {len(report['hyp_speakers'])} ({', '.join(report['hyp_speakers'])})", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log("   ASR & Diarization Performance", log_file)
    print_and_log("-" * 70, log_file)
    if report['segmentation'] == "streaming":
        print_and_log(f"  Global WER:       {report['wer'] * 100:.2f}%  "
                      f"(speaker-agnostic; NOT comparable to the paper -> use cpWER)", log_file)
        print_and_log(f"  Global CER:       {report['cer'] * 100:.2f}%  "
                      f"(speaker-agnostic)", log_file)
    else:
        print_and_log(f"  Global WER:       {report['wer'] * 100:.2f}%", log_file)
        print_and_log(f"  Global CER:       {report['cer'] * 100:.2f}%", log_file)
    if report['segmentation'] == "streaming":
        print_and_log(f"  Diarization (DER):{report['der'] * 100:.2f}%", log_file)
        print_and_log(f"  |- Missed Detection: {report['missed_detection'] * 100:.2f}%", log_file)
        print_and_log(f"  |- False Alarm:      {report['false_alarm'] * 100:.2f}%", log_file)
        print_and_log(f"  |- Speaker Confusion:{report['confusion'] * 100:.2f}%", log_file)
    else:
        print_and_log(f"  Diarization (DER): N/A (Oracle Segmentation)", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log("   Speaker Mapping & cpWER (Concatenated Permutation WER)", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log(f"  cpWER:            {report['cpwer'] * 100:.2f}%", log_file)
    print_and_log("\n  Speaker-by-Speaker Breakdown:", log_file)
    print_and_log("  +-----------------+-----------------+--------+--------+--------+--------+", log_file)
    print_and_log("  | Ref Speaker     | Hyp Speaker     | WER    | Sub    | Ins    | Del    |", log_file)
    print_and_log("  | (Ground Truth)  | (Assigned ID)   |        |        |        |        |", log_file)
    print_and_log("  +-----------------+-----------------+--------+--------+--------+--------+", log_file)
    
    for r_spk, h_spk in report["speaker_mapping"].items():
        det = report["speaker_details"].get(r_spk)
        if det:
            total_err = det["sub"] + det["ins"] + det["del"]
            spk_wer = total_err / det["ref_count"] if det["ref_count"] > 0 else (1.0 if total_err > 0 else 0.0)
            print_and_log(f"  | {r_spk:<15} | {h_spk:<15} | {spk_wer * 100:>5.1f}% | {det['sub']:<6} | {det['ins']:<6} | {det['del']:<6} |", log_file)
        else:
            print_and_log(f"  | {r_spk:<15} | {h_spk:<15} |    N/A | -      | -      | -      |", log_file)
            
    print_and_log("  +-----------------+-----------------+--------+--------+--------+--------+", log_file)
    print_and_log("=" * 70 + "\n", log_file)


def main():
    _chime6 = Chime6DatasetManager()
    default_audio = _chime6.audio_dir
    default_trans = _chime6.transcriptions_dir

    parser = argparse.ArgumentParser(description="CHiME6 Evaluation & Benchmark Script")
    parser.add_argument(
        "--audio-dir", "-a", type=str, default=default_audio,
        help=f"Directory containing CHiME6 audio files (default: {default_audio})"
    )
    parser.add_argument(
        "--trans-dir", "-t", type=str, default=default_trans,
        help=f"Directory containing CHiME6 JSON transcripts (default: {default_trans})"
    )
    parser.add_argument(
        "--session", "-s", type=str, default=None,
        help="Evaluate only a specific session ID (e.g. S01). If omitted, evaluates all matching sessions."
    )
    parser.add_argument(
        "--mode", "-m", type=str, choices=["worn", "far-field"], default="worn",
        help="Audio mode to evaluate: close-talk 'worn' microphones or 'far-field' arrays (default: 'worn')"
    )
    parser.add_argument(
        "--array", type=str, default="U01",
        help="Kinect array ID for far-field mode (e.g. 'U01', 'U01,U02', or 'all' to combine all arrays) (default: 'U01')"
    )
    parser.add_argument(
        "--enhance", type=str, choices=["none", "average", "gss", "beamform"], default="none",
        help="Enhancement technique: 'none' (single channel), 'average' (multi-channel combined), 'gss' (GSS preprocessed), or 'beamform' (Delay-and-Sum beamforming) (default: 'none')"
    )
    parser.add_argument(
        "--segmentation", type=str, choices=["oracle", "streaming"], default="oracle",
        help="Segmentation technique: 'oracle' (ground-truth boundaries / Track 1) or 'streaming' (VAD + Diarization / Track 2) (default: 'oracle')"
    )
    parser.add_argument(
        "--worn-speaker", type=str, default=None,
        help="Limit worn evaluation to a specific speaker ID (e.g., P01). If omitted, evaluates all speakers."
    )
    parser.add_argument(
        "--limit-minutes", "-l", type=float, default=None,
        help="Process only first N minutes of each audio session (default: full length)"
    )
    parser.add_argument(
        "--jiwer", action="store_true",
        help="Use jiwer library for precise WER calculations if installed"
    )
    parser.add_argument(
        "--meeteval", action="store_true",
        help="Use meeteval's reference cpWER implementation (pip install meeteval) "
             "so the cpWER number is directly comparable to CHiME-6/7/8 results."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save evaluation report to a JSON file"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to save the execution text log file"
    )
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Shorthand helper flag for 1-minute close-talk oracle evaluation sanity check"
    )
    
    args = parser.parse_args()
    
    # Handle shorthand sanity check flag
    if args.sanity_check:
        args.mode = "worn"
        args.segmentation = "oracle"
        args.limit_minutes = 1.0
        print("[INFO] [Sanity Check Mode] Enabled shorthand config: worn mode, oracle segmentation, 1-minute limit.")
        
    # Set up log file
    if args.log:
        log_file_path = args.log
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(PROJECT_ROOT, "output", f"chime6_eval_{timestamp}.log")
        
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    print_and_log(f"[START] Initializing CHiME-6 Benchmark Tool", log_file)
    print_and_log(f"   Log File: {log_file_path}", log_file)
    
    # 1. Match sessions
    trans_dir = Path(args.trans_dir)
    audio_dir = Path(args.audio_dir)
    
    if not trans_dir.exists():
        print_and_log(f"❌ Transcriptions directory does not exist: {trans_dir}", log_file)
        sys.exit(1)
        
    if not audio_dir.exists():
        print_and_log(f"❌ Audio directory does not exist: {audio_dir}", log_file)
        sys.exit(1)
        
    json_files = list(trans_dir.glob("**/*.json"))
    if not json_files:
        print_and_log(f"❌ No JSON transcripts found in {args.trans_dir}", log_file)
        sys.exit(1)
        
    sessions_to_run = []
    for j_file in json_files:
        s_id = j_file.stem
        
        # Filter by specific session if requested
        if args.session and s_id != args.session:
            continue
            
        sessions_to_run.append({
            "session_id": s_id,
            "transcript_path": str(j_file)
        })
        
    if not sessions_to_run:
        print_and_log(f"❌ No matching sessions found with the specified parameters.", log_file)
        sys.exit(1)
        
    print_and_log(f"   Found {len(sessions_to_run)} session(s) to evaluate.", log_file)
    
    reports = []
    for sess in sessions_to_run:
        try:
            report = run_chime6_session(
                audio_dir=args.audio_dir,
                transcript_path=sess["transcript_path"],
                mode=args.mode,
                array=args.array,
                enhance=args.enhance,
                segmentation=args.segmentation,
                limit_minutes=args.limit_minutes,
                use_jiwer=args.jiwer,
                worn_speaker=args.worn_speaker,
                use_meeteval=args.meeteval,
                log_file=log_file
            )
            if report:
                reports.append(report)
                print_session_report(report, log_file)
        except Exception as e:
            print_and_log(f"❌ Error evaluating session {sess['session_id']}: {e}", log_file)
            import traceback
            traceback.print_exc(file=log_file)
            traceback.print_exc()
            
    if not reports:
        print_and_log("❌ No successful evaluations completed.", log_file)
        log_file.close()
        sys.exit(1)
        
    # Global Summary across all sessions
    total_duration = sum(r["duration"] for r in reports)
    avg_wer = sum(r["wer"] for r in reports) / len(reports)
    avg_cer = sum(r["cer"] for r in reports) / len(reports)
    avg_der = sum(r["der"] for r in reports) / len(reports)
    avg_cpwer = sum(r["cpwer"] for r in reports) / len(reports)
    avg_rtf = sum(r["rtf"] for r in reports) / len(reports)
    
    print_and_log("\n" + "=" * 70, log_file)
    print_and_log("🏆  GLOBAL EVALUATION SUMMARY", log_file)
    print_and_log("=" * 70, log_file)
    print_and_log(f"   Total Sessions:  {len(reports)}", log_file)
    print_and_log(f"   Total Duration:  {total_duration / 60:.2f} minutes", log_file)
    print_and_log(f"   Average RTF:     {avg_rtf:.3f}x", log_file)
    print_and_log(f"   Config:          Mode: {args.mode.upper()} | Segmentation: {args.segmentation.upper()} | Enhance: {args.enhance.upper()}", log_file)
    print_and_log("-" * 70, log_file)
    if args.segmentation == "streaming":
        print_and_log(f"   Average Global WER:    {avg_wer * 100:.2f}%  (speaker-agnostic, info only)", log_file)
        print_and_log(f"   Average Global CER:    {avg_cer * 100:.2f}%  (speaker-agnostic, info only)", log_file)
        print_and_log(f"   Average DER:           {avg_der * 100:.2f}%", log_file)
        print_and_log(f"   Average cpWER:         {avg_cpwer * 100:.2f}%   <-- PRIMARY Track 2 metric", log_file)
    else:
        print_and_log(f"   Average Global WER:    {avg_wer * 100:.2f}%   <-- PRIMARY Track 1 metric", log_file)
        print_and_log(f"   Average Global CER:    {avg_cer * 100:.2f}%", log_file)
        print_and_log(f"   Average cpWER:         {avg_cpwer * 100:.2f}%", log_file)
    print_and_log("=" * 70 + "\n", log_file)
    
    log_file.close()
    
    if args.output:
        output_data = {
            "summary": {
                "total_sessions": len(reports),
                "total_duration_seconds": total_duration,
                "avg_rtf": avg_rtf,
                "avg_wer": avg_wer,
                "avg_cer": avg_cer,
                "avg_der": avg_der,
                "avg_cpwer": avg_cpwer,
                "mode": args.mode,
                "segmentation": args.segmentation,
                "enhance": args.enhance
            },
            "sessions": reports
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Full JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()