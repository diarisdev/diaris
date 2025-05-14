import wave
import pyaudio
import keyboard
import time
import numpy as np
import silero_vad.utils_vad
import webrtcvad
import torch
import silero_vad

#  Audio Recording Constants
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono (set to 2 if stereo is needed)
RATE = 48000  # Sample rate
FRAME_DURATION_MS = 20 # Duration of each frame
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000) # Number of frames per buffer
OUTPUT_FILENAME = "output.wav"  # Output file

#  Select the correct input device
input_device_index = 0

#  Initialize PyAudio
audio = pyaudio.PyAudio()

#  Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # Aggressiveness mode: 0, 1, 2, or 3 (higher is more aggressive)

# Load Silero VAD
silero_model = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)



# Save to WAV file
def save():
    with wave.open(OUTPUT_FILENAME, 'wb') as waveFile:
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
    print(f"✅ Audio saved as {OUTPUT_FILENAME}")

#  Display audio devices
print("\nAvailable Audio Devices:")
for i in range(audio.get_device_count()):
    dev = audio.get_device_info_by_index(i)
    print(f"{i}: {dev['name']} | Channels: {dev['maxInputChannels']} | Rate: {int(dev['defaultSampleRate'])}")


if input_device_index is None:
    input_device_index = int(input("\nEnter the index of your input device: "))

#  Open the audio stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=FRAME_SIZE)

frames = []

# Detect voice activity using WebRTC VAD
def detect_voice_activity(data):
    is_speech_webrtc = vad.is_speech(data, RATE)
    # If WebRTC VAD detects speech, use Silero VAD for more accurate results
    if is_speech_webrtc:
        # Convert the audio data to a tensor
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)
        # Get the speech timestamps
        with torch.no_grad():
            is_speech_silero = silero_model(audio_tensor, RATE).item() > 0.5

        return is_speech_silero  # Return True only if both VADs detect speech
    else:
        return False


def live_record():
    try:
        print("🎤 Now Streaming... Press SPACE to stop.")
        while True:
            try:                
                data = stream.read(FRAME_SIZE, exception_on_overflow=False)  # Prevent buffer errors
                is_speech = detect_voice_activity(data)
                if is_speech:
                    frames.append(data)
                else:
                    frames.append(b'\x00' * len(data))

                # Calculate the volume level to display
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume_level = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))

                # Display the volume level
                print(f"Volume Level: {volume_level:.2f} | {'Speaking' if is_speech else 'Silence'} ", end='\r')

            except OSError as e:
                print(f"⚠️ Buffer Overflow Error: {e}")  # If buffer overflow happens, warn the user

            if keyboard.is_pressed("space"):  # Stop when SPACE is pressed
                print("🛑 Recording stopped.")
                time.sleep(0.3)  # Prevent double presses
                break
    except KeyboardInterrupt:
        print("🛑 Recording stopped.")

    # Stop & close the stream 
    stream.stop_stream()
    stream.close()
    audio.terminate()

    save()


live_record()





