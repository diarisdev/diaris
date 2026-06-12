import os
import json
import subprocess
import sys
from pathlib import Path
import soundfile as sf

import re

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from lhotse import Recording, AudioSource, RecordingSet, SupervisionSegment, SupervisionSet, CutSet
    from tqdm import tqdm
except ImportError:
    print("❌ Required packages are not installed! GSS requires Lhotse and tqdm.")
    print("Please run: pip install lhotse tqdm")
    sys.exit(1)


def parse_time_to_seconds(t_val):
    """Parses timestamp to float seconds."""
    if isinstance(t_val, dict):
        if "original" in t_val:
            t_val = t_val["original"]
        elif t_val:
            t_val = next(iter(t_val.values()))
        else:
            raise ValueError("Empty time dict")

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


def clean_chime6_text(text):
    """Cleans CHiME6 text for fair comparison."""
    if not text:
        return ""
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = text.replace("-", " ")
    text = re.sub(r'[^\w\s\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_lhotse_manifests(audio_dir: Path, trans_dir: Path, output_dir: Path):
    print("🔍 Scanning audio files...")
    
    # 1. Find all Kinect array files: Sxx_Uxx.CH1.wav
    audio_files = sorted(list(audio_dir.glob("*_U*.CH1.wav")))
    if not audio_files:
        print(f"❌ No far-field Kinect CH1 files found in {audio_dir}")
        return None
        
    recordings_list = []
    supervisions_list = []
    
    for ch1_path in tqdm(audio_files, desc="Preparing manifests"):
        # File name format: S01_U01.CH1.wav -> session: S01, array: U01
        parts = ch1_path.name.split('.')
        sess_arr = parts[0]  # e.g., S01_U01
        session_id, array_id = sess_arr.split('_')
        
        # Check channels 1 to 4
        sources = []
        info = None
        for ch in range(1, 5):
            ch_file = audio_dir / f"{sess_arr}.CH{ch}.wav"
            if not ch_file.exists():
                tqdm.write(f"⚠️ Channel file {ch_file} is missing, skipping this channel.")
                continue
            
            # Read info from CH1 to get samplerate and samples
            if info is None:
                info = sf.info(str(ch_file))
                
            sources.append(AudioSource(
                type="file",
                channels=[ch - 1],
                source=str(ch_file.absolute())
            ))
            
        if not sources:
            continue
            
        recording = Recording(
            id=sess_arr,
            sources=sources,
            sampling_rate=info.samplerate,
            num_samples=info.frames,
            duration=info.duration
        )
        recordings_list.append(recording)
        
        # Load transcription file: S01.json
        trans_file = trans_dir / f"{session_id}.json"
        if not trans_file.exists():
            tqdm.write(f"⚠️ Transcription file {trans_file} not found, skipping supervisions for {sess_arr}.")
            continue
            
        with open(trans_file, 'r', encoding='utf-8') as f:
            trans_data = json.load(f)
            
        seg_idx = 0
        for item in trans_data:
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
            
            if not clean_text:
                continue
                
            # Ensure the segment is within recording boundaries
            if info is not None:
                if start >= info.duration:
                    continue
                if end > info.duration:
                    end = info.duration
                    
            duration = end - start
            if duration < 0.1:
                continue
                
            # Create a supervision segment for this utterance
            segment = SupervisionSegment(
                id=f"{sess_arr}_{speaker}_{seg_idx:05d}",
                recording_id=sess_arr,
                start=start,
                duration=duration,
                channel=list(range(len(sources))), # use all channels
                speaker=speaker,
                text=clean_text
            )
            supervisions_list.append(segment)
            seg_idx += 1
            
    if not recordings_list:
        print("❌ No recordings could be prepared.")
        return None
        
    print(f"💾 Saving Lhotse manifests to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recording_set = RecordingSet.from_recordings(recordings_list)
    supervision_set = SupervisionSet.from_segments(supervisions_list)
    
    recording_set.to_file(output_dir / "recordings.jsonl.gz")
    supervision_set.to_file(output_dir / "supervisions.jsonl.gz")
    
    print("✂️ Creating CutSets...")
    # 1. Cuts per recording (recording-level cuts with supervisions)
    cuts_rec = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    cuts_rec_path = output_dir / "cuts_per_recording.jsonl.gz"
    cuts_rec.to_file(cuts_rec_path)
    
    # 2. Cuts per segment (segment-level cuts, trimmed to supervisions)
    cuts_seg = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    trimmed_cuts = cuts_seg.trim_to_supervisions()
    
    # Filter trimmed cuts to make sure they contain supervisions and have valid duration
    trimmed_cuts = trimmed_cuts.filter(lambda cut: len(cut.supervisions) > 0 and cut.duration >= 0.1)
    
    cuts_seg_path = output_dir / "cuts_per_segment.jsonl.gz"
    trimmed_cuts.to_file(cuts_seg_path)
    
    print(f"✅ Recording CutSet saved to: {cuts_rec_path}")
    print(f"✅ Segment CutSet saved to: {cuts_seg_path}")
    return cuts_rec_path, cuts_seg_path


def main():
    audio_dir = PROJECT_ROOT / "tests" / "chime6_data" / "audio"
    trans_dir = PROJECT_ROOT / "tests" / "chime6_data" / "transcriptions"
    manifests_dir = PROJECT_ROOT / "tests" / "chime6_data" / "gss_manifests"
    enhanced_dir = audio_dir / "gss"
    
    print("==================================================")
    print("🚀 GSS PREPARATION & RUNNER SCRIPT FOR CHiME-6")
    print("==================================================")
    
    res = prepare_lhotse_manifests(audio_dir, trans_dir, manifests_dir)
    if not res:
        sys.exit(1)
        
    cuts_rec_path, cuts_seg_path = res
    enhanced_manifest = manifests_dir / "cuts_enhanced.jsonl.gz"
    
    gss_command = [
        "gss", "enhance", "cuts",
        str(cuts_rec_path),
        str(cuts_seg_path),
        str(enhanced_dir),
        "--enhanced-manifest", str(enhanced_manifest),
        "--bss-iterations", "10",
        "--context-duration", "5.0",
        "--use-garbage-class"
    ]
    
    print("\n" + "=" * 50)
    print("👉 RUNNING GSS ENHANCEMENT COMMAND:")
    print(" ".join(gss_command))
    print("=" * 50 + "\n")
    
    # Check if GPU-accelerated dependencies (like cupy) might be missing
    try:
        import cupy
        print(f"✨ CUDA/CuPy GPU acceleration detected on device {cupy.cuda.Device(0).id}")
    except ImportError:
        print("⚠️ Warning: CuPy is not installed. GSS execution might fall back to CPU or fail.")
        print("To run GSS on GPU, install CuPy corresponding to your CUDA version (e.g., pip install cupy-cuda12x)")
        
    # Ask if the user wants to execute the command directly
    try:
        response = input("Do you want to run GSS right now? (y/n): ").strip().lower()
        if response == 'y':
            print("⏳ Running Guided Source Separation (this can take a few minutes)...")
            enhanced_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(gss_command)
            if result.returncode == 0:
                print(f"✅ GSS Enhancement finished successfully!")
                print(f"Enhanced segments saved in: {enhanced_dir}")
            else:
                print("❌ GSS Enhancement command failed.")
        else:
            print("ℹ️ Setup completed. You can run the command printed above manually in your terminal.")
    except (KeyboardInterrupt, EOFError):
        print("\nℹ️ Setup completed. You can run the command printed above manually in your terminal.")


if __name__ == "__main__":
    main()
