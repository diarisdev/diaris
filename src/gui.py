import customtkinter as ctk
import threading
from .pipeline import run

class AudioProcessApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Process AI")
        self.geometry("900x650")
        
        # Tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ana çerçeve
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Üst Panel (Durum ve Butonlar)
        self.top_panel = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.top_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.status_label = ctk.CTkLabel(
            self.top_panel, 
            text="Hazır.", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.status_label.pack(side="left", padx=10)
        
        self.stop_btn = ctk.CTkButton(
            self.top_panel, 
            text="Durdur", 
            fg_color="#e74c3c", 
            hover_color="#c0392b",
            state="disabled",
            command=self.stop_recording
        )
        self.stop_btn.pack(side="right", padx=10)
        
        self.start_btn = ctk.CTkButton(
            self.top_panel, 
            text="Başlat", 
            fg_color="#2ecc71", 
            hover_color="#27ae60",
            command=self.start_recording
        )
        self.start_btn.pack(side="right", padx=10)
        
        # Transkripsiyon Alanı
        self.textbox = ctk.CTkTextbox(
            self.main_frame, 
            font=ctk.CTkFont(family="Consolas", size=14),
            wrap="word"
        )
        self.textbox.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.textbox.insert("0.0", "Yapay zeka sonuçları burada görünecek...\n\n")
        
        # Arka plan thread değişkenleri
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        
    def on_status_change(self, status_text):
        # UI güncellemeleri ana thread'de yapılmalı
        self.after(0, lambda: self.status_label.configure(text=status_text))
        
    def on_transcription(self, text):
        def append_text():
            self.textbox.insert("end", text + "\n\n")
            self.textbox.see("end")
        self.after(0, append_text)
        
    def start_recording(self):
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.textbox.insert("end", "==================================================\n")
        self.textbox.insert("end", "🔴 CANLI DİNLENİYOR VE ÇEVRİLİYOR...\n")
        self.textbox.insert("end", "==================================================\n\n")
        self.textbox.see("end")
        
        self.stop_event.clear()
        
        # Arka planda pipeline'ı başlat
        self.pipeline_thread = threading.Thread(
            target=run, 
            kwargs={
                "stop_event": self.stop_event,
                "on_status_change": self.on_status_change,
                "on_transcription": self.on_transcription
            },
            daemon=True
        )
        self.pipeline_thread.start()
        
    def stop_recording(self):
        self.stop_btn.configure(state="disabled")
        self.stop_event.set()
        # Thread'in güvenle kapanmasını beklemek için kontrol döngüsü başlat
        self.after(1000, self.check_thread_dead)
        
    def check_thread_dead(self):
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.after(500, self.check_thread_dead)
        else:
            self.start_btn.configure(state="normal")
