import wave
import pyaudio
import keyboard
import time
import numpy as np
import torch
import webrtcvad

# Audio Recording Constants
FORMAT = pyaudio.paInt16
CHANNELS = 2  # System audio is almost always Stereo (2 channels)
RATE = 48000
FRAME_DURATION_MS = 30 
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
OUTPUT_FILENAME = "system_recorded.wav"

def find_system_audio_device(p):
    """
    Automatically finds the Windows WASAPI Loopback device or Stereo Mix.
    """
    print("🔍 Searching for system audio device...")
    
    # 1. Try to find the Windows WASAPI Host API index
    wasapi_info = None
    for i in range(p.get_host_api_count()):
        host_info = p.get_host_api_info_by_index(i)
        if "WASAPI" in host_info["name"]:
            wasapi_info = host_info
            break

    if wasapi_info:
        # Search for Loopback devices within WASAPI
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            # Look for devices with "Loopback" and that belong to WASAPI
            if dev["hostApi"] == wasapi_info["index"] and "Loopback" in dev["name"]:
                print(f"✅ Found WASAPI Loopback: {dev['name']} (Index: {i})")
                return i

    # 2. Fallback: Search for "Stereo Mix" or "What U Hear" in any API
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev["name"].lower()
        if "stereo mix" in name or "what u hear" in name:
            print(f"✅ Found Fallback: {dev['name']} (Index: {i})")
            return i

    print("❌ Could not find a system audio device automatically.")
    return None

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Auto-detect device
input_device_index = find_system_audio_device(audio)

if input_device_index is None:
    print("Defaulting to standard input. Note: This might record your MIC, not SYSTEM audio.")
    input_device_index = audio.get_default_input_device_info()['index']

# Initialize VAD
vad = webrtcvad.Vad(3)
silero_model = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)

def detect_voice_activity(data):
    """VAD requires mono, but we are recording stereo."""
    audio_np = np.frombuffer(data, dtype=np.int16)
    
    # Convert stereo to mono for VAD processing
    if CHANNELS == 2:
        # Average the two channels to make mono
        audio_np_mono = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
    else:
        audio_np_mono = audio_np

    try:
        # WebRTC VAD
        is_speech_webrtc = vad.is_speech(audio_np_mono.tobytes(), RATE)
        if is_speech_webrtc:
            # Silero VAD
            audio_tensor = torch.from_numpy(audio_np_mono.astype(np.float32) / 32768.0)
            with torch.no_grad():
                confidence = silero_model(audio_tensor, RATE).item()
            return confidence > 0.4
    except:
        pass
    return False

def live_record():
    frames = []
    
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            input_device_index=input_device_index,
                            frames_per_buffer=FRAME_SIZE)

        print("\n🔴 Recording... Press SPACE to stop.")
        
        while not keyboard.is_pressed("space"):
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            
            # VAD check
            is_speech = detect_voice_activity(data)
            
            if is_speech:
                frames.append(data)
            else:
                # Add silence bytes to keep the timeline consistent
                frames.append(b'\x00' * len(data))

            print(f"Status: {'[VOICE DETECTED]' if is_speech else '[SYSTEM SOUNDS]'}   ", end='\r')

        print("\n🛑 Recording finished.")
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("Tip: If you get 'Invalid number of channels', try setting CHANNELS = 1 at the top.")
    
    finally:
        # Save file
        if frames:
            with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            print(f"💾 File saved as {OUTPUT_FILENAME}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    live_record()