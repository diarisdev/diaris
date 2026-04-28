import pyaudio
audio = pyaudio.PyAudio()
print("\n--- AVAILABLE INPUT DEVICES ---")
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Index {i}: {info['name']}")
audio.terminate()