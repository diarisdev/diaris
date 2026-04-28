import wave
import pyaudio
import keyboard
import numpy as np

# --- BURAYI DEĞİŞTİR ---
# Bir önceki adımda bulduğun Loopback indeksini buraya yaz
INPUT_DEVICE_INDEX = 12 
# -----------------------

FORMAT = pyaudio.paInt16
CHUNK = 1024
OUTPUT_FILENAME = "sistem_kaydi.wav"

audio = pyaudio.PyAudio()

try:
    # Seçilen cihazın teknik bilgilerini otomatik çek
    dev_info = audio.get_device_info_by_index(INPUT_DEVICE_INDEX)
    
    # Cihazın desteklediği kanal sayısını al (Hata almamak için en kritik yer)
    supported_channels = int(dev_info["maxInputChannels"])
    # Bazı Loopback cihazları 0 kanal raporlayabilir, bu durumda 2 (Stereo) zorla
    actual_channels = supported_channels if supported_channels > 0 else 2
    
    # Cihazın desteklediği örnekleme hızını al
    actual_rate = int(dev_info["defaultSampleRate"])

    print(f"\n--- BAĞLANTI AYARLARI ---")
    print(f"Cihaz: {dev_info['name']}")
    print(f"Kanal Sayısı: {actual_channels}")
    print(f"Hız (Rate): {actual_rate} Hz")
    print(f"-------------------------\n")

    stream = audio.open(
        format=FORMAT,
        channels=actual_channels,
        rate=actual_rate,
        input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )

    print("🔴 Kayıt yapılıyor... Durdurmak için SPACE (Boşluk) tuşuna bas.")
    frames = []

    while not keyboard.is_pressed("space"):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Ses geliyor mu görselleştir
            audio_data = np.frombuffer(data, dtype=np.int16)
            peak = np.abs(audio_data).max()
            print(f"Ses Seviyesi: {peak}      ", end='\r')
        except Exception:
            continue

    print("\n🛑 Kayıt bitti.")

    stream.stop_stream()
    stream.close()

    # Dosyayı kaydet
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(actual_channels)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(actual_rate)
        wf.writeframes(b''.join(frames))
    print(f"✅ Başarıyla kaydedildi: {OUTPUT_FILENAME}")

except Exception as e:
    print(f"\n⚠️ KRİTİK HATA: {e}")
    print("\nÇÖZÜM ÖNERİLERİ:")
    print("1. Bluetooth kulaklığının 'Ses Ayarları'ndan 'Hands-free' değil 'Stereo' olarak seçili olduğundan emin ol.")
    print("2. Eğer mikrofon kullanan bir uygulama (Discord, Zoom) açıksa kapat.")
    print("3. Denetim Masası > Ses > Kayıt sekmesinden kulaklığının Hands-free özelliğini devre dışı bırakmayı dene.")

finally:
    audio.terminate()