"""
CHiME6 Dataset Evaluation and Benchmarking Tool.

This script runs the audio-process system on CHiME6 sessions, comparing outputs
with ground-truth transcripts to calculate key conversational speech metrics:
  - WER (Word Error Rate) & CER (Character Error Rate)
  - DER (Diarization Error Rate) using pyannote.metrics
  - cpWER (concatenated minimum-permutation Word Error Rate)

Usage:
    python -m tests.chime6_benchmark --audio-dir /path/to/chime6/audio --trans-dir /path/to/chime6/transcriptions
"""
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=r".*std\(\).*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")

import os
import sys
import time
import json
import argparse
import re
import itertools
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def print_and_log(msg, log_file=None):
    """Prints to console and logs to file if provided."""
    print(msg)
    if log_file:
        log_file.write(str(msg) + "\n")


from tests.evaluator import TranscriptionEvaluator, DiarizationEvaluator
from src.audio.vad import VADEngine
from src.core.ai_worker import AIWorker
from src.config import (
    FRAME_DURATION_MS,
    SILENCE_LIMIT,
    SHORT_SILENCE_LIMIT,
    SOFT_CHUNK_DURATION_MS,
    MAX_CHUNK_DURATION_MS,
)


def parse_time_to_seconds(t_val):
    """
    Parses timestamp to float seconds.
    Supports float/int directly, string float seconds, and HH:MM:SS.mmm format.
    """
    if isinstance(t_val, (int, float)):
        return float(t_val)
    
    t_str = str(t_val).strip()
    try:
        return float(t_str)
    except ValueError:
        pass
        
    parts = t_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    else:
        raise ValueError(f"Unknown time format: {t_val}")


def load_chime6_transcript(json_path):
    """
    Loads CHiME6 ground truth transcript from JSON.
    Expected format: a list of utterances, each containing speaker, start_time, end_time, and words.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    segments = []
    for item in data:
        # Check standard CHiME6 keys
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
        text = item.get("words") or item.get("text") or ""
        
        segments.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": text
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


def compute_cpwer(ref_segments, hyp_segments, evaluator):
    """
    Computes cpWER (concatenated minimum-permutation Word Error Rate).
    
    1. Concatenates all reference text per speaker.
    2. Concatenates all hypothesis text per speaker.
    3. Finds the permutation matching hypothesis speakers to reference speakers that minimizes edit distance.
    4. Calculates the word error rate over the matched pairs.
    """
    # Group reference texts by speaker
    ref_speaker_texts = {}
    for seg in ref_segments:
        spk = seg["speaker"]
        ref_speaker_texts.setdefault(spk, []).append(seg)
        
    # Group hypothesis texts by speaker
    hyp_speaker_texts = {}
    for seg in hyp_segments:
        spk = seg["speaker"]
        if "Calibrating" in spk:  # ignore calibrating labels
            continue
        hyp_speaker_texts.setdefault(spk, []).append(seg)
        
    # Concatenate texts for each speaker chronologically
    ref_concats = {}
    for spk, segs in ref_speaker_texts.items():
        sorted_segs = sorted(segs, key=lambda x: x["start"])
        text = " ".join(evaluator.normalize_text(s["text"]) for s in sorted_segs)
        ref_concats[spk] = evaluator.normalize_text(text)
        
    hyp_concats = {}
    for spk, segs in hyp_speaker_texts.items():
        sorted_segs = sorted(segs, key=lambda x: x["start"])
        text = " ".join(evaluator.normalize_text(s["text"]) for s in sorted_segs)
        hyp_concats[spk] = evaluator.normalize_text(text)
        
    ref_spks = list(ref_concats.keys())
    hyp_spks = list(hyp_concats.keys())
    
    # Edge case: no speakers at all
    if not ref_spks and not hyp_spks:
        return 0.0, {}, {}
        
    K = len(ref_spks)
    M = len(hyp_spks)
    N = max(K, M)
    
    # Pad lists to N elements
    ref_list = ref_spks + [None] * (N - K)
    hyp_list = hyp_spks + [None] * (N - M)
    
    # Cost matrix for edit distances between all speaker pairs
    cost_matrix = [[0] * N for _ in range(N)]
    details_matrix = [[None] * N for _ in range(N)]
    
    for i in range(N):
        ref_spk = ref_list[i]
        ref_text = ref_concats[ref_spk] if ref_spk else ""
        ref_words = ref_text.split()
        
        for j in range(N):
            hyp_spk = hyp_list[j]
            hyp_text = hyp_concats[hyp_spk] if hyp_spk else ""
            hyp_words = hyp_text.split()
            
            sub, ins, dlt = evaluator._levenshtein_ops(ref_words, hyp_words)
            cost = sub + ins + dlt
            cost_matrix[i][j] = cost
            details_matrix[i][j] = {
                "sub": sub,
                "ins": ins,
                "del": dlt,
                "ref_count": len(ref_words),
                "hyp_count": len(hyp_words)
            }
            
    # Find best permutation minimizing total cost
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_perm = [0] * N
        for r, c in zip(row_ind, col_ind):
            best_perm[r] = c
        best_cost = sum(cost_matrix[i][best_perm[i]] for i in range(N))
    except ImportError:
        # Fallback if scipy is not installed
        if N <= 8:
            # Brute-force is fast enough for N <= 8
            best_cost = float('inf')
            best_perm = None
            for perm in itertools.permutations(range(N)):
                cost = sum(cost_matrix[i][perm[i]] for i in range(N))
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
        else:
            # Greedy matching for N > 8 to prevent hanging
            best_perm = [-1] * N
            matched_cols = set()
            best_cost = 0
            for i in range(N):
                min_c = -1
                min_val = float('inf')
                for j in range(N):
                    if j not in matched_cols and cost_matrix[i][j] < min_val:
                        min_val = cost_matrix[i][j]
                        min_c = j
                best_perm[i] = min_c
                matched_cols.add(min_c)
                best_cost += min_val
            
    # Map speakers and compile details
    mapping = {}
    details = {}
    total_ref_words = sum(len(ref_concats[spk].split()) for spk in ref_spks)
    
    for i in range(N):
        ref_spk = ref_list[i]
        hyp_spk = hyp_list[best_perm[i]]
        
        if ref_spk or hyp_spk:
            r_name = ref_spk if ref_spk else "[None]"
            h_name = hyp_spk if hyp_spk else "[No Match]"
            mapping[r_name] = h_name
            details[r_name] = details_matrix[i][best_perm[i]]
            
    cpwer = best_cost / total_ref_words if total_ref_words > 0 else 0.0
    return cpwer, mapping, details


def run_chime6_session(audio_path, transcript_path, limit_minutes=None, use_jiwer=False):
    """
    Evaluates a single CHiME6 session.
    Simulates real-time streaming processing of the audio file and compares with transcript.
    """
    session_id = os.path.basename(transcript_path).split('.')[0]
    
    # 1. Load ground truth transcript
    ref_segments = load_chime6_transcript(transcript_path)
    if not ref_segments:
        print(f"❌ Empty or invalid ground truth transcript at: {transcript_path}")
        return None
        
    # 2. Get audio details and read audio
    info = sf.info(audio_path)
    rate = info.samplerate
    channels = info.channels
    duration = info.duration
    
    if limit_minutes:
        limit_seconds = limit_minutes * 60.0
        actual_duration = min(duration, limit_seconds)
        frames_to_read = int(rate * actual_duration)
        audio_data, _ = sf.read(audio_path, dtype="int16", frames=frames_to_read)
        ref_segments = truncate_segments(ref_segments, actual_duration)
    else:
        actual_duration = duration
        audio_data, _ = sf.read(audio_path, dtype="int16")
        
    print(f"\n📂 Session: {session_id}")
    print(f"   Audio File:    {audio_path}")
    print(f"   Audio Info:    {rate} Hz | {channels} channels | {actual_duration / 60:.2f} mins")
    print(f"   Ground Truth:  {len(ref_segments)} segments loaded")
    
    # Ensure audio is mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)
        
    audio_bytes = audio_data.tobytes()
    
    # 3. Initialize pipeline components
    print("🧠 Loading models...")
    vad_engine = VADEngine()
    ai_worker = AIWorker(rate=rate, channels=1)
    if not ai_worker.load_models():
        print("❌ Failed to load AIWorker models.")
        return None
        
    ai_worker.speaker_tracker.reset()
    warmup_s = ai_worker.speaker_tracker.warmup_ms / 1000.0
    
    # Stream simulation setup
    bytes_per_frame = int(rate * (FRAME_DURATION_MS / 1000.0) * 2)
    chunk_buffer = []
    silence_counter = 0
    has_spoken = False
    
    hyp_segments = []
    global_time_s = 0.0
    chunk_start_s = 0.0
    
    print("🔄 Processing audio stream...")
    start_eval_time = time.time()
    
    # Process frame-by-frame
    for offset in range(0, len(audio_bytes), bytes_per_frame):
        frame_bytes = audio_bytes[offset:offset+bytes_per_frame]
        if len(frame_bytes) < bytes_per_frame:
            break
            
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
            # Process final speech chunk
            chunk_bytes_to_send = b''.join(chunk_buffer)
            output = ai_worker.process_chunk(chunk_bytes_to_send, language="en")
            results = output.get("results") if isinstance(output, dict) else output
            
            if results:
                # Run background diarization simulation
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
                    
                    # Ignore calibration phase segments if desired, or map them as Speaker Tracker does
                    hyp_segments.append({
                        "start": global_start,
                        "end": global_end,
                        "speaker": r["speaker"],
                        "text": r["text"]
                    })
                    
            chunk_buffer = []
            silence_counter = 0
            has_spoken = False
            
    eval_elapsed = time.time() - start_eval_time
    rtf = eval_elapsed / actual_duration
    print(f"✅ Processing complete in {eval_elapsed:.1f}s (RTF: {rtf:.3f}x)")
    
    # 4. Evaluation
    print("📊 Evaluating performance metrics...")
    
    trans_eval = TranscriptionEvaluator(use_jiwer=use_jiwer)
    diar_eval = DiarizationEvaluator()
    
    # ASR Evaluation: Concatenate all texts chronologically
    ref_all_text = " ".join(trans_eval.normalize_text(s["text"]) for s in ref_segments)
    # Ignore calibrating tags in hypothesis transcripts
    clean_hyp_segments = [s for s in hyp_segments if "Calibrating" not in s["speaker"]]
    hyp_all_text = " ".join(trans_eval.normalize_text(s["text"]) for s in clean_hyp_segments)
    
    asr_res = trans_eval.evaluate(
        reference=ref_all_text,
        hypothesis=hyp_all_text,
        audio_path=audio_path,
        duration=actual_duration
    )
    
    # Diarization Evaluation
    ref_diar_intervals = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in ref_segments]
    # Filter out calibration segments from hypothesis for fair diarization evaluation
    hyp_diar_intervals = []
    for s in hyp_segments:
        if "Calibrating" in s["speaker"]:
            continue
        hyp_diar_intervals.append({
            "start": s["start"],
            "end": s["end"],
            "speaker": s["speaker"]
        })
        
    # We must also filter references/hypotheses to exceed warmup duration since system is calibrating
    filtered_refs = [r for r in ref_diar_intervals if r["start"] >= warmup_s]
    filtered_hyps = [h for h in hyp_diar_intervals if h["start"] >= warmup_s]
    
    diar_res = diar_eval.evaluate(
        meeting_id=session_id,
        reference_intervals=filtered_refs,
        hypothesis_intervals=filtered_hyps,
        duration=max(0.1, actual_duration - warmup_s)
    )
    
    # cpWER Evaluation
    cpwer, mapping, details = compute_cpwer(ref_segments, clean_hyp_segments, trans_eval)
    
    # Generate report dict
    report = {
        "session_id": session_id,
        "audio_path": audio_path,
        "duration": actual_duration,
        "rtf": rtf,
        "ref_speakers": list(set(s["speaker"] for s in ref_segments)),
        "hyp_speakers": list(set(s["speaker"] for s in clean_hyp_segments)),
        "wer": asr_res.wer,
        "cer": asr_res.cer,
        "der": diar_res.der if diar_res else 0.0,
        "false_alarm": diar_res.false_alarm if diar_res else 0.0,
        "missed_detection": diar_res.missed_detection if diar_res else 0.0,
        "confusion": diar_res.confusion if diar_res else 0.0,
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
    print_and_log(f"📊  CHiME-6 EVALUATION REPORT - SESSION: {report['session_id']}", log_file)
    print_and_log("=" * 70, log_file)
    print_and_log(f"📁 Audio File:     {report['audio_path']}", log_file)
    print_and_log(f"⏳ Duration:       {report['duration']:.2f} seconds ({report['duration'] / 60:.1f} minutes)", log_file)
    print_and_log(f"⚡ Processing RTF: {report['rtf']:.3f}x", log_file)
    print_and_log(f"👥 Ref Speakers:   {len(report['ref_speakers'])} ({', '.join(report['ref_speakers'])})", log_file)
    print_and_log(f"👥 Hyp Speakers:   {len(report['hyp_speakers'])} ({', '.join(report['hyp_speakers'])})", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log("📈 ASR & Diarization Performance", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log(f"  Global WER:       {report['wer'] * 100:.2f}%", log_file)
    print_and_log(f"  Global CER:       {report['cer'] * 100:.2f}%", log_file)
    print_and_log(f"  Diarization (DER):{report['der'] * 100:.2f}%", log_file)
    print_and_log(f"  ├─ Missed Detection: {report['false_alarm'] * 100:.2f}%", log_file)
    print_and_log(f"  ├─ False Alarm:      {report['missed_detection'] * 100:.2f}%", log_file)
    print_and_log(f"  └─ Speaker Confusion:{report['confusion'] * 100:.2f}%", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log("🗣️ Speaker Mapping & cpWER (Concatenated Permutation WER)", log_file)
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
    default_audio = os.path.join(PROJECT_ROOT, "tests", "chime6_data", "audio")
    default_trans = os.path.join(PROJECT_ROOT, "tests", "chime6_data", "transcriptions")

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
        help="Evaluate only a specific session ID (e.g. S02). If omitted, evaluates all matching sessions."
    )
    parser.add_argument(
        "--audio-pattern", "-p", type=str, default="U01.CH1",
        help="Filename pattern to select audio channel (default: 'U01.CH1')"
    )
    parser.add_argument(
        "--limit-minutes", "-l", type=float, default=None,
        help="Process only first N minutes of each audio file (default: full length)"
    )
    parser.add_argument(
        "--jiwer", action="store_true",
        help="Use jiwer library for precise WER calculations if installed"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save evaluation report to a JSON file"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to save the execution text log file (default: output/chime6_eval_YYYYMMDD_HHMMSS.log)"
    )
    
    args = parser.parse_args()
    
    # Set up log file with timestamp to prevent overwriting
    if args.log:
        log_file_path = args.log
        if os.path.exists(log_file_path):
            base, ext = os.path.splitext(log_file_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            candidate = f"{base}_{timestamp}{ext}"
            if os.path.exists(candidate):
                counter = 1
                while os.path.exists(f"{base}_{timestamp}_{counter}{ext}"):
                    counter += 1
                log_file_path = f"{base}_{timestamp}_{counter}{ext}"
            else:
                log_file_path = candidate
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(PROJECT_ROOT, "output", f"chime6_eval_{timestamp}.log")
        if os.path.exists(log_file_path):
            base, ext = os.path.splitext(log_file_path)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}"):
                counter += 1
            log_file_path = f"{base}_{counter}{ext}"
        
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    # 1. Match sessions
    trans_dir = Path(args.trans_dir)
    audio_dir = Path(args.audio_dir)
    
    if not trans_dir.exists():
        print(f"❌ Transcriptions directory does not exist: {trans_dir}")
        print(f"   Lütfen gerçek veri dizinini --trans-dir ile belirtin ya da verilerinizi '{default_trans}' altına yerleştirin.")
        sys.exit(1)
        
    if not audio_dir.exists():
        print(f"❌ Audio directory does not exist: {audio_dir}")
        print(f"   Lütfen gerçek veri dizinini --audio-dir ile belirtin ya da verilerinizi '{default_audio}' altına yerleştirin.")
        sys.exit(1)
        
    # Get json files
    json_files = list(trans_dir.glob("**/*.json"))
    if not json_files:
        print(f"❌ No JSON transcripts found in {args.trans_dir}")
        sys.exit(1)
        
    sessions_to_run = []
    
    for j_file in json_files:
        s_id = j_file.stem
        
        # Filter by specific session if requested
        if args.session and s_id != args.session:
            continue
            
        # Try to find corresponding audio file matching session ID and pattern
        audio_candidates = list(audio_dir.glob(f"**/{s_id}*.wav")) + list(audio_dir.glob(f"**/{s_id}*.flac"))
        if not audio_candidates:
            continue
            
        # Filter by pattern
        matched_audio = None
        for ac in audio_candidates:
            if args.audio_pattern in ac.name:
                matched_audio = ac
                break
                
        # If no pattern match, fallback to the first candidate
        if not matched_audio:
            matched_audio = audio_candidates[0]
            print(f"⚠️ Warning: No audio matched pattern '{args.audio_pattern}' for session {s_id}. Using '{matched_audio.name}'.")
            
        sessions_to_run.append({
            "session_id": s_id,
            "audio_path": str(matched_audio),
            "transcript_path": str(j_file)
        })
        
    if not sessions_to_run:
        print(f"❌ No matching sessions found with the specified parameters.")
        sys.exit(1)
        
    print(f"🚀 Found {len(sessions_to_run)} session(s) to evaluate.")
    
    reports = []
    for sess in sessions_to_run:
        try:
            report = run_chime6_session(
                audio_path=sess["audio_path"],
                transcript_path=sess["transcript_path"],
                limit_minutes=args.limit_minutes,
                use_jiwer=args.jiwer
            )
            if report:
                reports.append(report)
                print_session_report(report, log_file)
        except Exception as e:
            print(f"❌ Error evaluating session {sess['session_id']}: {e}")
            import traceback
            traceback.print_exc()
            
    if not reports:
        print("❌ No successful evaluations completed.")
        if log_file:
            log_file.close()
        sys.exit(1)
        
    # 5. Global Summary across all sessions
    total_duration = sum(r["duration"] for r in reports)
    avg_wer = sum(r["wer"] for r in reports) / len(reports)
    avg_cer = sum(r["cer"] for r in reports) / len(reports)
    avg_der = sum(r["der"] for r in reports) / len(reports)
    avg_cpwer = sum(r["cpwer"] for r in reports) / len(reports)
    avg_rtf = sum(r["rtf"] for r in reports) / len(reports)
    
    print_and_log("\n" + "=" * 70, log_file)
    print_and_log("🏆  GLOBAL EVALUATION SUMMARY", log_file)
    print_and_log("=" * 70, log_file)
    print_and_log(f"📋 Total Sessions:  {len(reports)}", log_file)
    print_and_log(f"⏳ Total Duration:  {total_duration / 60:.2f} minutes", log_file)
    print_and_log(f"⚡ Average RTF:     {avg_rtf:.3f}x", log_file)
    print_and_log("-" * 70, log_file)
    print_and_log(f"📈 Average Global WER:    {avg_wer * 100:.2f}%", log_file)
    print_and_log(f"📈 Average Global CER:    {avg_cer * 100:.2f}%", log_file)
    print_and_log(f"📈 Average DER:           {avg_der * 100:.2f}%", log_file)
    print_and_log(f"📈 Average cpWER:         {avg_cpwer * 100:.2f}%", log_file)
    print_and_log("=" * 70 + "\n", log_file)
    
    if log_file:
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
                "avg_cpwer": avg_cpwer
            },
            "sessions": reports
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Full JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
