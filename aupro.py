import wave
import pyaudiowpatch as pyaudio
import keyboard
import numpy as np
import torch
import webrtcvad
import os

# --- Constants ---
FORMAT = pyaudio.paInt16
RATE = 48000
FRAME_DURATION_MS = 30
OUTPUT_FILENAME = "system_recorded.wav"

def find_best_voicemeeter_device(p):
    """
    Tries to find the specific VAIO Output (B1) with valid channels.
    """
    print("🔍 Scanning for valid Voicemeeter Output...")
    target_keywords = ["voicemeeter output", "voicemeeter vaio", "b1"]
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev["name"].lower()
        
        if any(key in name for key in target_keywords):
            if dev["maxInputChannels"] > 0:
                print(f"⭐ Candidate Found: {dev['name']} (Index {i}) with {dev['maxInputChannels']} channels")
                return i
            
    return None

def check_speech(data, vad_model, silero, rate, channels):
    """
    Dual-layer VAD: Fast WebRTC filter -> Accurate Silero confirmation.
    Includes padding logic to resolve the 30ms vs 32ms conflict.
    """
    # 1. Prevent Phase Cancellation: Take ONLY the Left channel (index 0)
    audio_np = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
    mono_audio = audio_np[:, 0].copy()

    # 2. Safety Check for WebRTC (Needs exactly 30ms = 1440 samples @ 48kHz)
    if len(mono_audio) != int(rate * 30 / 1000):
        return False, 0.0

    try:
        # --- LAYER 1: WebRTC ---
        # WebRTC is extremely fast, so it acts as our first gatekeeper.
        if vad_model.is_speech(mono_audio.tobytes(), rate):
            
            # --- LAYER 2: Silero ---
            # Downsample 48kHz to 16kHz (1440 samples -> 480 samples)
            if rate == 48000:
                silero_audio = mono_audio[::3]
                silero_rate = 16000
            else:
                silero_audio = mono_audio
                silero_rate = rate

            # THE FIX: Silero strictly requires at least 512 samples.
            # We have 480. We pad the end with 32 zeros (2ms of silence).
            if len(silero_audio) < 512:
                pad_length = 512 - len(silero_audio)
                silero_audio = np.pad(silero_audio, (0, pad_length), 'constant')

            # Convert to float32 for the Neural Network
            audio_float = silero_audio.astype(np.float32) / 32768.0
            
            with torch.no_grad():
                confidence = silero(torch.from_numpy(audio_float), silero_rate).item()
            
            # If confidence is > 25%, we have a confirmed human voice!
            return confidence > 0.25, confidence 
            
    except Exception as e:
        print(f"VAD Error: {e}") 
        
    return False, 0.0

def record():
    p = pyaudio.PyAudio()
    device_index = find_best_voicemeeter_device(p)

    if device_index is None:
        print("❌ Could not find a Voicemeeter device with active input channels.")
        p.terminate()
        return

    dev_info = p.get_device_info_by_index(device_index)
    
    CHANNELS = 2 
    frame_size = int(RATE * FRAME_DURATION_MS / 1000)
    
    print(f"✅ Final Choice: {dev_info['name']}")
    print(f"ℹ️ Recording as: {CHANNELS} Channels @ {RATE}Hz")

    # Initialize VADs
    vad = webrtcvad.Vad(1) # 3 is most aggressive
    print("🧠 Loading Silero AI Model...")
    silero_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)

    frames = []
    stream = None

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frame_size
        )

        print("\n" + "="*40)
        print("🔴 RECORDING... Press SPACE to stop.")
        print("="*40 + "\n")

        while not keyboard.is_pressed("space"):
            data = stream.read(frame_size, exception_on_overflow=False)
            
            # Run our Dual-Layer VAD
            is_speech, conf = check_speech(data, vad, silero_model, RATE, CHANNELS)

            if is_speech:
                # Keep the real audio frame
                frames.append(data)
            else:
                # Replace the audio with pure digital silence (zeros) of the exact same length
                frames.append(b'\x00' * len(data))

            status = "🎙️  [ VOICE DETECTED ]" if is_speech else "😶 [ MUTED SILENCE ]"
            # Now we can see the exact math!
            print(f"Status: {status} | AI Confidence: {conf:.2f}       ", end='\r')

    except Exception as e:
        print(f"\n⚠️ Error: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        
        if frames:
            print("\n🛑 Saving file...")
            with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            print(f"💾 Saved as {os.path.abspath(OUTPUT_FILENAME)}")
        
        p.terminate()

if __name__ == "__main__":
    record()