import customtkinter as ctk
import threading

# pyrefly: ignore [missing-import]
import pyaudiowpatch as pyaudio

from .audio.device import list_loopback_devices
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

        self.main_frame.grid_rowconfigure(3, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- Cihaz Seçimi ---
        self.device_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.device_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")

        device_label = ctk.CTkLabel(
            self.device_frame,
            text="Ses Cihazı:",
            font=ctk.CTkFont(size=13),
        )
        device_label.pack(side="left", padx=(10, 5))

        self.device_combo = ctk.CTkComboBox(
            self.device_frame,
            state="readonly",
            width=500,
            font=ctk.CTkFont(size=12),
        )
        self.device_combo.pack(side="left", fill="x", expand=True, padx=5)
        self.device_combo.set("Cihaz taranıyor...")

        self.refresh_btn = ctk.CTkButton(
            self.device_frame,
            text="Yenile",
            width=55,
            command=self.refresh_devices,
        )
        self.refresh_btn.pack(side="left", padx=(0, 10))

        # --- Dil Seçimi ---
        self.lang_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.lang_frame.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="ew")

        lang_label = ctk.CTkLabel(
            self.lang_frame,
            text="Dil / Çeviri:",
            font=ctk.CTkFont(size=13),
        )
        lang_label.pack(side="left", padx=(10, 5))

        self.languages_dict = {
            "İngilizce": "en",
            "Türkçe": "tr",
            "Almanca": "de",
            "Fransızca": "fr",
            "İspanyolca": "es",
            "İtalyanca": "it",
            "Rusça": "ru",
            "Arapça": "ar",
            "Çince": "zh",
            "Japonca": "ja",
            "Korece": "ko",
            "Hollandaca": "nl",
            "Portekizce": "pt"
        }
        self.lang_names = list(self.languages_dict.keys())

        self.source_lang_combo = ctk.CTkComboBox(
            self.lang_frame,
            state="readonly",
            width=150,
            values=self.lang_names,
            font=ctk.CTkFont(size=12),
        )
        self.source_lang_combo.pack(side="left", padx=5)

        from .config import WHISPER_LANGUAGE
        default_src_name = "İngilizce"
        for name, code in self.languages_dict.items():
            if code == WHISPER_LANGUAGE:
                default_src_name = name
                break
        self.source_lang_combo.set(default_src_name)

        arrow_label = ctk.CTkLabel(
            self.lang_frame,
            text="➔",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        arrow_label.pack(side="left", padx=5)

        self.target_lang_combo = ctk.CTkComboBox(
            self.lang_frame,
            state="readonly",
            width=150,
            values=self.lang_names,
            font=ctk.CTkFont(size=12),
        )
        self.target_lang_combo.pack(side="left", padx=5)
        self.target_lang_combo.set("Türkçe")

        # --- Üst Panel (Durum ve Butonlar) ---
        self.top_panel = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.top_panel.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.top_panel,
            text="Hazır.",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.status_label.pack(side="left", padx=10)

        self.stop_btn = ctk.CTkButton(
            self.top_panel,
            text="Durdur",
            fg_color="#e74c3c",
            hover_color="#c0392b",
            state="disabled",
            command=self.stop_recording,
        )
        self.stop_btn.pack(side="right", padx=10)

        self.start_btn = ctk.CTkButton(
            self.top_panel,
            text="Başlat",
            fg_color="#2ecc71",
            hover_color="#27ae60",
            command=self.start_recording,
        )
        self.start_btn.pack(side="right", padx=10)

        # --- Transkripsiyon Alanı ---
        self.textbox = ctk.CTkTextbox(
            self.main_frame,
            font=ctk.CTkFont(family="Consolas", size=14),
            wrap="word",
        )
        self.textbox.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.textbox.insert("0.0", "Yapay zeka sonuçları burada görünecek...\n\n")

        # Arka plan thread değişkenleri
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        self.finalized_segments = []

        # Cihaz listesi
        self.loopback_devices = []
        self.refresh_devices()

    # ------------------------------------------------------------------ #
    #  Cihaz yönetimi                                                     #
    # ------------------------------------------------------------------ #

    def refresh_devices(self):
        """Loopback cihazlarını tarayıp combobox'a yükler. Varsayılan cihazı seçer."""
        p = pyaudio.PyAudio()
        default_name = ""
        try:
            self.loopback_devices = list_loopback_devices(p)
            try:
                default_output = p.get_default_output_device_info()
                default_name = default_output["name"].lower()
            except Exception:
                pass
        finally:
            p.terminate()

        if self.loopback_devices:
            default_idx = 0
            names = []
            for i, dev in enumerate(self.loopback_devices):
                name = dev["name"]
                if default_name and default_name in name.lower():
                    name += "  (Varsayılan)"
                    default_idx = i
                names.append(name)

            self._device_display_names = names
            self.device_combo.configure(values=names)
            self.device_combo.set(names[default_idx])
        else:
            self._device_display_names = []
            self.device_combo.configure(values=["Loopback cihazı bulunamadı"])
            self.device_combo.set("Loopback cihazı bulunamadı")

    def _get_selected_device_index(self):
        """Seçili cihazın PyAudio device index'ini döndürür."""
        if not self.loopback_devices:
            return None

        selected = self.device_combo.get()
        for i, display_name in enumerate(self._device_display_names):
            if display_name == selected:
                return self.loopback_devices[i]["index"]
        return None

    # ------------------------------------------------------------------ #
    #  Pipeline callback'leri                                             #
    # ------------------------------------------------------------------ #

    def on_status_change(self, status_text):
        # UI güncellemeleri ana thread'de yapılmalı
        self.after(0, lambda: self.status_label.configure(text=status_text))

    def on_transcription(self, event):
        if not isinstance(event, dict):
            event = {"type": "final", "text": str(event)}

        def update_ui():
            self.textbox.configure(state="normal")
            self.textbox.delete("1.0", "end")
            
            # Re-insert headers
            self.textbox.insert("end", "==================================================\n")
            self.textbox.insert("end", "CANLI DİNLENİYOR VE ÇEVRİLİYOR...\n")
            self.textbox.insert("end", "==================================================\n\n")
            
            # If it's a final event, ensure space and insert
            if event["type"] == "final":
                idx = event.get("segment_index", len(self.finalized_segments))
                while len(self.finalized_segments) <= idx:
                    self.finalized_segments.append("")
                self.finalized_segments[idx] = event["text"]

            # Re-insert finalized segments
            for seg in self.finalized_segments:
                if seg:
                    self.textbox.insert("end", seg + "\n\n")
                
            if event["type"] != "final":
                # If it's a partial event, show it at the end
                self.textbox.insert("end", f"[Canlı] {event['text']}\n")
                
            self.textbox.configure(state="disabled")
            self.textbox.see("end")

        self.after(0, update_ui)

    def on_speaker_update(self, event):
        """
        Geriye dönük konuşmacı etiket güncellemesi.
        """
        def update_ui():
            segment_index = event["segment_index"]
            text = event["text"]
            
            if 0 <= segment_index < len(self.finalized_segments):
                self.finalized_segments[segment_index] = text
                
                self.textbox.configure(state="normal")
                self.textbox.delete("1.0", "end")
                
                # Re-insert headers
                self.textbox.insert("end", "==================================================\n")
                self.textbox.insert("end", "CANLI DİNLENİYOR VE ÇEVRİLİYOR...\n")
                self.textbox.insert("end", "==================================================\n\n")
                
                for seg in self.finalized_segments:
                    if seg:
                        self.textbox.insert("end", seg + "\n\n")
                    
                self.textbox.configure(state="disabled")
                self.textbox.see("end")

        self.after(0, update_ui)

    # ------------------------------------------------------------------ #
    #  Kayıt kontrolleri                                                  #
    # ------------------------------------------------------------------ #

    def start_recording(self):
        device_index = self._get_selected_device_index()
        if device_index is None:
            self.on_status_change("Ses cihazı seçilmedi.")
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.device_combo.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        self.source_lang_combo.configure(state="disabled")
        self.target_lang_combo.configure(state="disabled")

        self.finalized_segments = []
        self.textbox.configure(state="normal")
        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", "==================================================\n")
        self.textbox.insert("end", "CANLI DİNLENİYOR VE ÇEVRİLİYOR...\n")
        self.textbox.insert("end", "==================================================\n\n")
        self.textbox.configure(state="disabled")
        self.textbox.see("end")

        self.stop_event.clear()

        # Arka planda pipeline'ı başlat
        self.pipeline_thread = threading.Thread(
            target=run,
            kwargs={
                "stop_event": self.stop_event,
                "on_status_change": self.on_status_change,
                "on_transcription": self.on_transcription,
                "on_speaker_update": self.on_speaker_update,
                "device_index": device_index,
                "get_lang_pair": lambda: (
                    self.languages_dict.get(self.source_lang_combo.get(), "en"),
                    self.languages_dict.get(self.target_lang_combo.get(), "tr")
                )
            },
            daemon=True,
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
            self.device_combo.configure(state="readonly")
            self.refresh_btn.configure(state="normal")
            self.source_lang_combo.configure(state="readonly")
            self.target_lang_combo.configure(state="readonly")
