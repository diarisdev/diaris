import wave
import pyaudiowpatch as pyaudio
import keyboard
import numpy as np
import torch
import webrtcvad
import os
import queue
import threading
import tempfile
import soundfile as sf
import warnings  
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# --- GEREKSİZ UYARILARI SESSİZE AL ---
warnings.filterwarnings("ignore")
os.environ["SYMLINK_WARNING"] = "0"

# --- System Constants ---
FORMAT = pyaudio.paInt16
RATE = 48000
FRAME_DURATION_MS = 30
OUTPUT_FILENAME = "system_recorded.wav"

# --- HYBRID CONFIGURATION ---
# ⚠️ TOKEN VE KLASÖR YOLLARIN (Aynı kalıyor)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_MODELS_DIR = r"C:\Users\yusuf\Desktop\Github\Audio-process\Local_Models"
WHISPER_PATH = os.path.join(LOCAL_MODELS_DIR, "whisper-small")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

audio_queue = queue.Queue()

# ==========================================
# 1. AI WORKER THREAD (Background Process)
# ==========================================
def transcription_worker():
    print(f"\n🧠 [AI Worker] {DEVICE.upper()} üzerinde başlatılıyor...")
    try:
        diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            token=HF_TOKEN
        )
        if DEVICE == "cuda":
            diarizer.to(torch.device("cuda"))
            
        transcriber = WhisperModel(WHISPER_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("✅ [AI Worker] Modeller yüklendi, ses bekleniyor...\n")
    except Exception as e:
        print(f"❌ [AI Worker Error] Modeller yüklenemedi. Hata: {e}")
        return

    while True:
        chunk_bytes = audio_queue.get()
        if chunk_bytes is None:
            break 

        # --- FIX: BOŞ DİZİ HATASINI ENGELLE ---
        # Eğer gelen ses paketi boşsa (0 byte), hiç işleme sokmadan yoksay
        if len(chunk_bytes) == 0:
            audio_queue.task_done()
            continue

        audio_np_int16 = np.frombuffer(chunk_bytes, dtype=np.int16).reshape(-1, 2)
        
        # Eğer reshape sonrası array boşaldıysa yine yoksay (Mean of empty slice hatasını önler)
        if audio_np_int16.size == 0:
            audio_queue.task_done()
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_np_int16, RATE)
            tmp_path = tmp_file.name

        mono_float32 = audio_np_int16.mean(axis=1).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(mono_float32).unsqueeze(0) 
        
        pyannote_input = {
            "waveform": waveform, 
            "sample_rate": RATE
        }

        try:
            pipeline_output = diarizer(pyannote_input)
            
            if hasattr(pipeline_output, "speaker_diarization"):
                diarization = pipeline_output.speaker_diarization
            else:
                diarization = pipeline_output
            
            segments, _ = transcriber.transcribe(tmp_path, word_timestamps=True, language="en")
            
            # Sadece eğer gerçekten bir metin algılandıysa ekrana yazdır
            segments = list(segments) # Generator'ı listeye çevir
            if len(segments) > 0:
                print("\n" + "-"*50)
                for segment in segments:
                    segment_center = segment.start + (segment.end - segment.start) / 2
                    current_speaker = "Unknown"
                    
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if turn.start <= segment_center <= turn.end:
                            current_speaker = speaker
                            break
                    
                    print(f"[{current_speaker}] {segment.start:.1f}s - {segment.end:.1f}s: {segment.text}")
                print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"\n⚠️ [Transcription Error]: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            audio_queue.task_done()

# ==========================================
# 2. AUDIO CAPTURE & VAD
# ==========================================
def find_best_voicemeeter_device(p):
    target_keywords = ["voicemeeter output", "voicemeeter vaio", "b1"]
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev["name"].lower()
        if any(key in name for key in target_keywords):
            if dev["maxInputChannels"] > 0:
                return i
    return None

def check_speech(data, vad_model, silero, rate, channels):
    audio_np = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
    mono_audio = audio_np[:, 0].copy()

    if len(mono_audio) != int(rate * 30 / 1000):
        return False, 0.0

    try:
        if vad_model.is_speech(mono_audio.tobytes(), rate):
            if rate == 48000:
                silero_audio = mono_audio[::3]
                silero_rate = 16000
            else:
                silero_audio = mono_audio
                silero_rate = rate

            if len(silero_audio) < 512:
                pad_length = 512 - len(silero_audio)
                silero_audio = np.pad(silero_audio, (0, pad_length), 'constant')

            audio_float = silero_audio.astype(np.float32) / 32768.0
            
            with torch.no_grad():
                confidence = silero(torch.from_numpy(audio_float), silero_rate).item()
            
            return confidence > 0.25, confidence 
    except:
        pass
    return False, 0.0

def record():
    p = pyaudio.PyAudio()
    device_index = find_best_voicemeeter_device(p)

    if device_index is None:
        print("❌ Voicemeeter VAIO bulunamadı.")
        p.terminate()
        return

    CHANNELS = 2 
    frame_size = int(RATE * FRAME_DURATION_MS / 1000)

    vad = webrtcvad.Vad(1)
    silero_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)

    ai_thread = threading.Thread(target=transcription_worker, daemon=True)
    ai_thread.start()

    frames = []           
    chunk_buffer = []     
    silence_counter = 0   
    has_spoken = False    
    SILENCE_LIMIT = 50    

    stream = None

    try:
        stream = p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE,
            input=True, input_device_index=device_index, frames_per_buffer=frame_size
        )

        print("\n" + "="*40)
        print("🔴 CANLI DİNLENİYOR VE ÇEVRİLİYOR...")
        print("Durdurmak için BOŞLUK (SPACE) tuşuna bas.")
        print("="*40 + "\n")

        while not keyboard.is_pressed("space"):
            data = stream.read(frame_size, exception_on_overflow=False)
            is_speech, conf = check_speech(data, vad, silero_model, RATE, CHANNELS)

            if is_speech:
                frames.append(data)
                chunk_buffer.append(data)
                silence_counter = 0 
                has_spoken = True
                status = "🎙️  [ KONUŞULUYOR ]"
            else:
                silence_bytes = b'\x00' * len(data)
                frames.append(silence_bytes)
                
                if has_spoken:
                    chunk_buffer.append(silence_bytes)
                    silence_counter += 1
                    status = "⏱️  [ BEKLENİYOR ] "
                    
                    if silence_counter > SILENCE_LIMIT:
                        # Eğer buffer boş değilse gönder
                        if len(chunk_buffer) > 0:
                            audio_queue.put(b''.join(chunk_buffer))
                        chunk_buffer = []
                        silence_counter = 0
                        has_spoken = False
                        status = "📦 [ YAPAY ZEKAYA İLETİLDİ ]"
                else:
                    status = "😶 [ SESSİZLİK ]  "

            print(f"Durum: {status} | AI: {conf:.2f}       ", end='\r')

    except Exception as e:
        print(f"\n⚠️ Main Loop Error: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
            
        print("\n🛑 AI Kapatılıyor, lütfen bekleyin...")
        audio_queue.put(None) 
        ai_thread.join()      
        
        if frames:
            print("\n💾 Ana ses dosyası kaydediliyor...")
            with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            print(f"✅ Dosya kaydedildi: {os.path.abspath(OUTPUT_FILENAME)}")
        
        p.terminate()

if __name__ == "__main__":
    record()