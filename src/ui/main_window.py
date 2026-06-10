import sys
import threading
from PySide6 import QtCore, QtGui, QtWidgets

# pyrefly: ignore [missing-import]
import pyaudiowpatch as pyaudio

from src.audio.device import list_loopback_devices
from src.pipeline import run
from src.ui.subtitle_overlay import SubtitleOverlay
from src.ui.tray import SystemTrayController

class PipelineSignals(QtCore.QObject):
    status_changed = QtCore.Signal(str)
    transcription_received = QtCore.Signal(dict)
    speaker_updated = QtCore.Signal(dict)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Audio Process AI")
        self.resize(800, 850)
        self.setMinimumSize(720, 800)
        
        # Sinyal köprüsü
        self.signals = PipelineSignals()
        self.signals.status_changed.connect(self.safe_on_status_change)
        self.signals.transcription_received.connect(self.safe_on_transcription)
        self.signals.speaker_updated.connect(self.safe_on_speaker_update)
        
        # Eyaletler
        self.overlay_visible = True
        self.is_dark_theme = True
        self.finalized_segments = []
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        self.loopback_devices = []
        
        # Dil Eyaleti ve Çeviriler
        self.TRANSLATIONS = {
            "tr": {
                "title": "Audio Process AI",
                "subtitle": "Canlı transkripsiyon ve altyazı kontrol paneli",
                "theme_light": "Aydınlık",
                "theme_dark": "Karanlık",
                "audio_source": "Ses kaynağı",
                "refresh_tooltip": "Cihazları yenile",
                "start": "Başlat",
                "stop": "Durdur",
                "exit": "Çıkış",
                "subtitle_window": "Altyazı penceresi",
                "font_size": "Yazı boyutu",
                "opacity": "Opaklık",
                "show_overlay": "Altyazı penceresini göster",
                "click_through": "Tıklamaları arkadaki pencereye geçir",
                "speaker_coloring": "Konuşmacı renklendirmesi",
                "audio_translation": "Ses / Çeviri",
                "ui_language": "Arayüz Dili",
                "live_log": "Canlı transkripsiyon günlüğü",
                "tray_show": "Kontrol Panelini Göster",
                "tray_toggle_overlay": "Altyazı Overlay Aç/Kapat",
                "tray_quit": "Uygulamadan Çık",
                "tray_bg_message": "Uygulama arka planda çalışmaya devam ediyor. Açmak için simgeye çift tıklayın.",
                "tray_recording_message": "Kayıt başladı. Kontrol panelini açmak için tepsideki simgeye çift tıklayın.",
                "lang_en": "İngilizce",
                "lang_tr": "Türkçe",
                "analyzing": "Çözümleniyor...",
                "live_prefix": "[Canlı] "
            },
            "en": {
                "title": "Audio Process AI",
                "subtitle": "Live transcription and subtitle control panel",
                "theme_light": "Light",
                "theme_dark": "Dark",
                "audio_source": "Audio source",
                "refresh_tooltip": "Refresh devices",
                "start": "Start",
                "stop": "Stop",
                "exit": "Exit",
                "subtitle_window": "Subtitle window",
                "font_size": "Font size",
                "opacity": "Opacity",
                "show_overlay": "Show subtitle window",
                "click_through": "Pass clicks to background window",
                "speaker_coloring": "Speaker coloring",
                "audio_translation": "Audio / Translation",
                "ui_language": "UI Language",
                "live_log": "Live transcription log",
                "tray_show": "Show Control Panel",
                "tray_toggle_overlay": "Toggle Subtitle Overlay",
                "tray_quit": "Exit Application",
                "tray_bg_message": "The application continues to run in the background. Double-click the icon to open.",
                "tray_recording_message": "Recording started. Double-click the tray icon to open the control panel.",
                "lang_en": "English",
                "lang_tr": "Turkish",
                "analyzing": "Resolving...",
                "live_prefix": "[Live] "
            }
        }
        
        system_locale = QtCore.QLocale.system().name()
        self.ui_lang = "tr" if system_locale.startswith("tr") else "en"
        
        # Alt bileşenler
        self.overlay = SubtitleOverlay()
        if self.overlay_visible:
            self.overlay.show()
            
        self.tray = SystemTrayController(self)
        
        # Arayüz ve Stil oluşturma
        self.setup_ui()
        self.retranslate_ui()
        self.apply_styles()
        
        # Cihazları tara
        self.refresh_devices()

    def setup_ui(self):
        # Ana Widget
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName("AppRoot")
        self.setCentralWidget(self.central_widget)
        
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(24, 22, 24, 24)
        self.layout.setSpacing(16)

        self.header = QtWidgets.QWidget(self.central_widget)
        self.header_layout = QtWidgets.QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)

        title_box = QtWidgets.QWidget(self.header)
        title_layout = QtWidgets.QVBoxLayout(title_box)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(3)

        self.app_title = QtWidgets.QLabel("Audio Process AI", title_box)
        self.app_title.setObjectName("AppTitle")
        self.app_subtitle = QtWidgets.QLabel("Canlı transkripsiyon ve altyazı kontrol paneli", title_box)
        self.app_subtitle.setObjectName("MutedLabel")

        title_layout.addWidget(self.app_title)
        title_layout.addWidget(self.app_subtitle)

        self.theme_btn = QtWidgets.QPushButton("Aydınlık", self.header)
        self.theme_btn.setObjectName("theme_btn")
        self.theme_btn.setMinimumHeight(38)
        self.theme_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DesktopIcon))
        self.theme_btn.clicked.connect(self.toggle_theme)

        self.header_layout.addWidget(title_box, 1)
        self.header_layout.addWidget(self.theme_btn)

        self.layout.addWidget(self.header)
        
        # --- CİHAZ SEÇİMİ PANELİ ---
        self.device_group = QtWidgets.QFrame(self.central_widget)
        self.device_group.setObjectName("Panel")
        self.device_layout = QtWidgets.QHBoxLayout(self.device_group)
        self.device_layout.setContentsMargins(16, 14, 16, 14)
        self.device_layout.setSpacing(12)
        
        self.dev_label = QtWidgets.QLabel("Ses kaynağı", self.device_group)
        self.dev_label.setObjectName("SectionHeader")
        
        self.device_combo = QtWidgets.QComboBox(self.device_group)
        self.device_combo.setMinimumHeight(42)
        
        self.refresh_btn = QtWidgets.QPushButton(self.device_group)
        self.refresh_btn.setObjectName("IconButton")
        self.refresh_btn.setFixedSize(42, 42)
        self.refresh_btn.setToolTip("Cihazları yenile")
        self.refresh_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_btn.clicked.connect(self.refresh_devices)
        
        self.device_layout.addWidget(self.dev_label)
        self.device_layout.addWidget(self.device_combo, 1)
        self.device_layout.addWidget(self.refresh_btn)
        
        self.layout.addWidget(self.device_group)
        
        # --- BUTONLAR VE DURUM PANELİ ---
        self.control_group = QtWidgets.QFrame(self.central_widget)
        self.control_group.setObjectName("Panel")
        self.control_layout = QtWidgets.QHBoxLayout(self.control_group)
        self.control_layout.setContentsMargins(16, 14, 16, 14)
        self.control_layout.setSpacing(10)
        
        self.status_label = QtWidgets.QLabel("Sistem Hazır.", self.control_group)
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setProperty("status", "ready")
        
        self.start_btn = QtWidgets.QPushButton("Başlat", self.control_group)
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_recording)
        
        self.stop_btn = QtWidgets.QPushButton("Durdur", self.control_group)
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setMinimumHeight(42)
        self.stop_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        
        self.exit_btn = QtWidgets.QPushButton("Çıkış", self.control_group)
        self.exit_btn.setObjectName("exit_btn")
        self.exit_btn.setMinimumHeight(42)
        self.exit_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton))
        self.exit_btn.clicked.connect(self.close_app)
        
        self.control_layout.addWidget(self.status_label, 1)
        self.control_layout.addWidget(self.start_btn)
        self.control_layout.addWidget(self.stop_btn)
        self.control_layout.addWidget(self.exit_btn)
        
        self.layout.addWidget(self.control_group)
        
        # --- ALTYAZI AYARLARI PANELİ ---
        self.settings_group = QtWidgets.QFrame(self.central_widget)
        self.settings_group.setObjectName("Panel")
        self.settings_layout = QtWidgets.QGridLayout(self.settings_group)
        self.settings_layout.setContentsMargins(18, 18, 18, 18)
        self.settings_layout.setHorizontalSpacing(14)
        self.settings_layout.setVerticalSpacing(12)
        
        self.title_settings = QtWidgets.QLabel("Altyazı penceresi", self.settings_group)
        self.title_settings.setObjectName("GroupTitle")
        self.settings_layout.addWidget(self.title_settings, 0, 0, 1, 3)
        
        # Yazı Boyutu Ayarı
        self.font_label = QtWidgets.QLabel("Yazı boyutu", self.settings_group)
        self.font_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.settings_group)
        self.font_slider.setRange(8, 32)
        self.font_slider.setValue(self.overlay.font_size)
        self.font_slider.setMinimumWidth(260)
        self.font_val_label = QtWidgets.QLabel(f"{self.overlay.font_size} px", self.settings_group)
        self.font_val_label.setObjectName("ValueLabel")
        self.font_slider.valueChanged.connect(self.on_font_changed)
        
        self.settings_layout.addWidget(self.font_label, 1, 0)
        self.settings_layout.addWidget(self.font_slider, 1, 1)
        self.settings_layout.addWidget(self.font_val_label, 1, 2)
        
        # Şeffaflık Ayarı
        self.opacity_label = QtWidgets.QLabel("Opaklık", self.settings_group)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.settings_group)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(int(self.overlay.opacity * 100))
        self.opacity_val_label = QtWidgets.QLabel(f"{int(self.overlay.opacity * 100)}%", self.settings_group)
        self.opacity_val_label.setObjectName("ValueLabel")
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        
        self.settings_layout.addWidget(self.opacity_label, 2, 0)
        self.settings_layout.addWidget(self.opacity_slider, 2, 1)
        self.settings_layout.addWidget(self.opacity_val_label, 2, 2)
        
        # Dil Seçimi Ayarı
        self.lang_label = QtWidgets.QLabel("Dil / Çeviri", self.settings_group)
        
        # Container for side-by-side comboboxes
        self.lang_container = QtWidgets.QWidget(self.settings_group)
        self.lang_container_layout = QtWidgets.QHBoxLayout(self.lang_container)
        self.lang_container_layout.setContentsMargins(0, 0, 0, 0)
        self.lang_container_layout.setSpacing(8)
        
        self.source_lang_combo = QtWidgets.QComboBox(self.lang_container)
        self.source_lang_combo.addItem("İngilizce", "en")
        self.source_lang_combo.addItem("Türkçe", "tr")
        self.source_lang_combo.setMinimumHeight(36)
        
        arrow_label = QtWidgets.QLabel("➔", self.lang_container)
        arrow_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #9aa6b2;")
        arrow_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        self.target_lang_combo = QtWidgets.QComboBox(self.lang_container)
        self.target_lang_combo.addItem("Türkçe", "tr")
        self.target_lang_combo.addItem("İngilizce", "en")
        self.target_lang_combo.setMinimumHeight(36)
        
        self.lang_container_layout.addWidget(self.source_lang_combo, 1)
        self.lang_container_layout.addWidget(arrow_label)
        self.lang_container_layout.addWidget(self.target_lang_combo, 1)
        
        self.settings_layout.addWidget(self.lang_label, 3, 0)
        self.settings_layout.addWidget(self.lang_container, 3, 1, 1, 2)

        # Arayüz Dili Ayarı (UI Language)
        self.ui_lang_label = QtWidgets.QLabel("Arayüz Dili", self.settings_group)
        self.ui_lang_combo = QtWidgets.QComboBox(self.settings_group)
        self.ui_lang_combo.addItem("Türkçe", "tr")
        self.ui_lang_combo.addItem("English", "en")
        self.ui_lang_combo.setMinimumHeight(36)
        
        # Set default selection
        idx = self.ui_lang_combo.findData(self.ui_lang)
        if idx >= 0:
            self.ui_lang_combo.setCurrentIndex(idx)
            
        self.ui_lang_combo.currentIndexChanged.connect(self.on_ui_lang_changed)
        
        self.settings_layout.addWidget(self.ui_lang_label, 4, 0)
        self.settings_layout.addWidget(self.ui_lang_combo, 4, 1, 1, 2)
        
        # Checkbox Kontrolleri
        self.show_overlay_cb = QtWidgets.QCheckBox("Altyazı penceresini göster", self.settings_group)
        self.show_overlay_cb.setChecked(self.overlay_visible)
        self.show_overlay_cb.toggled.connect(self.set_overlay_visible)
        
        self.click_through_cb = QtWidgets.QCheckBox("Tıklamaları arkadaki pencereye geçir", self.settings_group)
        self.click_through_cb.setChecked(self.overlay.click_through)
        self.click_through_cb.toggled.connect(self.overlay.set_click_through)
        
        self.speaker_color_cb = QtWidgets.QCheckBox("Konuşmacı renklendirmesi", self.settings_group)
        self.speaker_color_cb.setChecked(self.overlay.speaker_coloring)
        self.speaker_color_cb.setToolTip("Altyazı metnini konuşmacı rengine boyar")
        self.speaker_color_cb.toggled.connect(self.overlay.set_speaker_coloring)
        
        self.settings_layout.addWidget(self.show_overlay_cb, 5, 0, 1, 3)
        self.settings_layout.addWidget(self.click_through_cb, 6, 0, 1, 3)
        self.settings_layout.addWidget(self.speaker_color_cb, 7, 0, 1, 3)
        
        self.layout.addWidget(self.settings_group)
        
        # --- KAYIT / TRANSKRİPSİYON GÜNLÜĞÜ ---
        self.log_group = QtWidgets.QFrame(self.central_widget)
        self.log_group.setObjectName("Panel")
        self.log_layout = QtWidgets.QVBoxLayout(self.log_group)
        self.log_layout.setContentsMargins(18, 18, 18, 18)
        self.log_layout.setSpacing(12)
        
        self.log_title = QtWidgets.QLabel("Canlı transkripsiyon günlüğü", self.log_group)
        self.log_title.setObjectName("GroupTitle")
        
        self.log_text = QtWidgets.QTextEdit(self.log_group)
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 11))
        
        self.log_layout.addWidget(self.log_title)
        self.log_layout.addWidget(self.log_text)
        
        self.layout.addWidget(self.log_group, 1)

    def apply_styles(self):
        self.DARK_STYLESHEET = """
            QMainWindow {
                background-color: #0f1115;
            }
            QWidget#AppRoot {
                background-color: #0f1115;
            }
            QFrame#Panel {
                background-color: #171a21;
                border: 1px solid #252a34;
                border-radius: 8px;
            }
            QLabel {
                color: #e8edf3;
                font-size: 13px;
                font-family: "Segoe UI";
            }
            QLabel#AppTitle {
                color: #f8fafc;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#MutedLabel {
                color: #9aa6b2;
                font-size: 12px;
            }
            #SectionHeader {
                font-weight: 700;
                color: #9aa6b2;
                min-width: 86px;
            }
            #GroupTitle {
                color: #f8fafc;
                font-size: 15px;
                font-weight: 700;
                padding-bottom: 6px;
                border: none;
            }
            #StatusLabel {
                background-color: #202631;
                border: 1px solid #2d3543;
                border-radius: 16px;
                color: #9aa6b2;
                font-size: 13px;
                font-weight: 700;
                padding: 7px 12px;
            }
            #StatusLabel[status="live"] {
                background-color: rgba(45, 212, 191, 36);
                border-color: rgba(45, 212, 191, 89);
                color: #5eead4;
            }
            #StatusLabel[status="error"] {
                background-color: rgba(248, 113, 113, 33);
                border-color: rgba(248, 113, 113, 82);
                color: #fca5a5;
            }
            QComboBox {
                background-color: #10131a;
                border: 1px solid #2c3340;
                border-radius: 7px;
                color: #f8fafc;
                font-size: 13px;
                padding: 8px 12px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #171a21;
                border: 1px solid #2c3340;
                color: #f8fafc;
                selection-background-color: #2563eb;
            }
            QPushButton {
                background-color: #232936;
                border: 1px solid #303846;
                border-radius: 7px;
                color: #f8fafc;
                font-weight: 700;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background-color: #2d3544;
            }
            QPushButton:disabled {
                background-color: #151922;
                color: #606b78;
                border-color: #222836;
            }
            QPushButton#start_btn {
                background-color: #14b8a6;
                border-color: #14b8a6;
                color: #062925;
            }
            QPushButton#start_btn:hover {
                background-color: #2dd4bf;
            }
            QPushButton#stop_btn {
                background-color: #ef4444;
                border-color: #ef4444;
                color: #ffffff;
            }
            QPushButton#stop_btn:hover {
                background-color: #f87171;
            }
            QPushButton#exit_btn {
                background-color: #2a303b;
                border-color: #3a4352;
                color: #e8edf3;
            }
            QPushButton#exit_btn:hover {
                background-color: #3a4352;
            }
            QPushButton#theme_btn {
                background-color: #10131a;
                border-color: #2c3340;
                color: #dbeafe;
            }
            QPushButton#theme_btn:hover {
                background-color: #1d2430;
            }
            QPushButton#IconButton {
                padding: 0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #262d39;
                border: 1px solid #303846;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 2px solid #0f1115;
                width: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #7dd3fc;
            }
            QLabel#ValueLabel {
                color: #cbd5e1;
                min-width: 54px;
            }
            QCheckBox {
                color: #d7dee8;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #3a4352;
                border-radius: 5px;
                background-color: #10131a;
            }
            QCheckBox::indicator:checked {
                background-color: #14b8a6;
                border-color: #14b8a6;
            }
            QTextEdit {
                background-color: #0c0f14;
                border: 1px solid #252a34;
                border-radius: 8px;
                color: #e8edf3;
                padding: 12px;
                selection-background-color: #2563eb;
            }
            QScrollBar:vertical {
                background: #11151d;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #303846;
                border-radius: 5px;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
        """

        self.LIGHT_STYLESHEET = """
            QMainWindow {
                background-color: #f6f8fb;
            }
            QWidget#AppRoot {
                background-color: #f6f8fb;
            }
            QFrame#Panel {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 8px;
            }
            QLabel {
                color: #172033;
                font-size: 13px;
                font-family: "Segoe UI";
            }
            QLabel#AppTitle {
                color: #111827;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#MutedLabel {
                color: #64748b;
                font-size: 12px;
            }
            #SectionHeader {
                font-weight: 700;
                color: #64748b;
                min-width: 86px;
            }
            #GroupTitle {
                color: #111827;
                font-size: 15px;
                font-weight: 700;
                padding-bottom: 6px;
                border: none;
            }
            #StatusLabel {
                background-color: #eef2f7;
                border: 1px solid #d8e0ea;
                border-radius: 16px;
                color: #64748b;
                font-size: 13px;
                font-weight: 700;
                padding: 7px 12px;
            }
            #StatusLabel[status="live"] {
                background-color: #ccfbf1;
                border-color: #99f6e4;
                color: #0f766e;
            }
            #StatusLabel[status="error"] {
                background-color: #fee2e2;
                border-color: #fecaca;
                color: #b91c1c;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                color: #111827;
                font-size: 13px;
                padding: 8px 12px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                color: #111827;
                selection-background-color: #2563eb;
            }
            QPushButton {
                background-color: #eef2f7;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                color: #172033;
                font-weight: 700;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
            QPushButton:disabled {
                background-color: #f8fafc;
                color: #94a3b8;
                border-color: #e2e8f0;
            }
            QPushButton#start_btn {
                background-color: #14b8a6;
                border-color: #14b8a6;
                color: #062925;
            }
            QPushButton#start_btn:hover {
                background-color: #2dd4bf;
            }
            QPushButton#stop_btn {
                background-color: #ef4444;
                border-color: #ef4444;
                color: #ffffff;
            }
            QPushButton#stop_btn:hover {
                background-color: #dc2626;
            }
            QPushButton#exit_btn {
                background-color: #ffffff;
                border-color: #cbd5e1;
                color: #475569;
            }
            QPushButton#exit_btn:hover {
                background-color: #f1f5f9;
            }
            QPushButton#theme_btn {
                background-color: #ffffff;
                border-color: #cbd5e1;
                color: #1d4ed8;
            }
            QPushButton#theme_btn:hover {
                background-color: #eff6ff;
            }
            QPushButton#IconButton {
                padding: 0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0284c7;
                border: 2px solid #ffffff;
                width: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #0369a1;
            }
            QLabel#ValueLabel {
                color: #475569;
                min-width: 54px;
            }
            QCheckBox {
                color: #334155;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:checked {
                background-color: #14b8a6;
                border-color: #14b8a6;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 8px;
                color: #172033;
                padding: 12px;
                selection-background-color: #bfdbfe;
            }
            QScrollBar:vertical {
                background: #f1f5f9;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #cbd5e1;
                border-radius: 5px;
                min-height: 28px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
        """

        if self.is_dark_theme:
            self.setStyleSheet(self.DARK_STYLESHEET)
        else:
            self.setStyleSheet(self.LIGHT_STYLESHEET)

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.setStyleSheet(self.DARK_STYLESHEET)
            self.theme_btn.setText(self.TRANSLATIONS[self.ui_lang]["theme_light"])
        else:
            self.setStyleSheet(self.LIGHT_STYLESHEET)
            self.theme_btn.setText(self.TRANSLATIONS[self.ui_lang]["theme_dark"])

    # ------------------------------------------------------------------ #
    #  Cihaz yönetimi                                                     #
    # ------------------------------------------------------------------ #

    def refresh_devices(self):
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

        self.device_combo.clear()
        if self.loopback_devices:
            default_idx = 0
            for i, dev in enumerate(self.loopback_devices):
                name = dev["name"]
                if default_name and default_name in name.lower():
                    name += "  (Varsayılan)"
                    default_idx = i
                self.device_combo.addItem(name, dev["index"])
            self.device_combo.setCurrentIndex(default_idx)
        else:
            self.device_combo.addItem("Loopback cihazı bulunamadı", None)

    # ------------------------------------------------------------------ #
    #  Ayar Değişiklikleri                                                #
    # ------------------------------------------------------------------ #

    def on_font_changed(self, value):
        self.font_val_label.setText(f"{value} px")
        self.overlay.set_font_size(value)

    def on_opacity_changed(self, value):
        self.opacity_val_label.setText(f"{value}%")
        self.overlay.set_overlay_opacity(value / 100.0)

    def set_overlay_visible(self, checked):
        self.overlay_visible = checked
        self.show_overlay_cb.setChecked(checked)
        self.tray.update_overlay_state(checked)
        if checked:
            self.overlay.show()
        else:
            self.overlay.hide()

    def toggle_overlay(self):
        self.set_overlay_visible(not self.overlay_visible)

    # ------------------------------------------------------------------ #
    #  Thread-Safe Callback Köprüleri                                     #
    # ------------------------------------------------------------------ #

    def translate_status(self, text, lang):
        text_lower = text.lower()
        if "hazır" in text_lower or "ready" in text_lower:
            return "Sistem Hazır." if lang == "tr" else "System Ready."
        if "geçersiz" in text_lower or "invalid" in text_lower:
            return "Seçili cihaz geçersiz." if lang == "tr" else "Selected device is invalid."
        if "durduruldu (zaman aşımı)" in text_lower or "stopped (timeout)" in text_lower:
            return "Durduruldu (zaman aşımı)." if lang == "tr" else "Stopped (timeout)."
        if "durduruldu" in text_lower or "stopped" in text_lower:
            return "Durduruldu." if lang == "tr" else "Stopped."
        if "kapatılıyor" in text_lower or "shutting down" in text_lower or "ai kapatılıyor" in text_lower:
            import re
            m = re.search(r"\((.*?)\)", text)
            suffix = f" ({m.group(1)})" if m else ""
            if "sistem" in text_lower or "system" in text_lower:
                return f"Sistem kapatılıyor...{suffix}" if lang == "tr" else f"Shutting down system...{suffix}"
            if "zaman aşımı" in text_lower or "timeout" in text_lower:
                return "Zaman aşımı! Zorla kapatılıyor..." if lang == "tr" else "Timeout! Force shutting down..."
            return f"Yapay zeka kapatılıyor...{suffix}" if lang == "tr" else f"Shutting down AI...{suffix}"
        if "canlı" in text_lower or "listening" in text_lower:
            return "CANLI DİNLENİYOR VE ÇEVRİLİYOR..." if lang == "tr" else "LISTENING AND TRANSLATING LIVE..."
        return text

    def on_ui_lang_changed(self, index):
        self.ui_lang = self.ui_lang_combo.itemData(index)
        
        # Automatically update translation target based on UI language
        new_target_lang = self.ui_lang
        idx = self.target_lang_combo.findData(new_target_lang)
        if idx >= 0:
            self.target_lang_combo.setCurrentIndex(idx)
            
        self.retranslate_ui()

    def retranslate_ui(self):
        lang = self.ui_lang
        trans = self.TRANSLATIONS[lang]
        
        self.app_title.setText(trans["title"])
        self.app_subtitle.setText(trans["subtitle"])
        
        if self.is_dark_theme:
            self.theme_btn.setText(trans["theme_light"])
        else:
            self.theme_btn.setText(trans["theme_dark"])
            
        self.dev_label.setText(trans["audio_source"])
        self.refresh_btn.setToolTip(trans["refresh_tooltip"])
        
        self.start_btn.setText(trans["start"])
        self.stop_btn.setText(trans["stop"])
        self.exit_btn.setText(trans["exit"])
        
        self.title_settings.setText(trans["subtitle_window"])
        self.font_label.setText(trans["font_size"])
        self.opacity_label.setText(trans["opacity"])
        self.lang_label.setText(trans["audio_translation"])
        self.ui_lang_label.setText(trans["ui_language"])
        
        self.source_lang_combo.setItemText(0, trans["lang_en"])
        self.source_lang_combo.setItemText(1, trans["lang_tr"])
        
        self.target_lang_combo.setItemText(0, trans["lang_tr"])
        self.target_lang_combo.setItemText(1, trans["lang_en"])
        
        self.show_overlay_cb.setText(trans["show_overlay"])
        self.click_through_cb.setText(trans["click_through"])
        self.speaker_color_cb.setText(trans["speaker_coloring"])
        
        self.log_title.setText(trans["live_log"])
        
        # Translate status
        self.safe_on_status_change(self.status_label.text())
        
        # Retranslate system tray actions
        if hasattr(self, 'tray') and hasattr(self.tray, 'show_action'):
            self.tray.show_action.setText(trans["tray_show"])
            self.tray.toggle_overlay_action.setText(trans["tray_toggle_overlay"])
            self.tray.quit_action.setText(trans["tray_quit"])

        # Update overlay window language
        if hasattr(self, 'overlay'):
            self.overlay.set_language(lang)

    def safe_on_status_change(self, text):
        translated_text = self.translate_status(text, self.ui_lang)
        self.status_label.setText(translated_text)
        lower_text = translated_text.lower()
        if "canl" in lower_text or "listen" in lower_text:
            status = "live"
        elif any(keyword in lower_text for keyword in ("geçersiz", "gecersiz", "hata", "bulunamadı", "bulunamadi", "invalid", "error", "not found")):
            status = "error"
        else:
            status = "ready"

        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def safe_on_transcription(self, event):
        if not isinstance(event, dict):
            event = {"type": "final", "text": str(event)}
            
        # UI Güncellemesi
        if event["type"] == "final":
            idx = event.get("segment_index", len(self.finalized_segments))
            while len(self.finalized_segments) <= idx:
                self.finalized_segments.append("")
            self.finalized_segments[idx] = event["text"]
            partial = ""
        else:
            partial = event["text"]
            
        # Log alanını güncelle
        self.log_text.clear()
        for seg in self.finalized_segments:
            if seg:
                seg_translated = seg.replace("Çözümleniyor...", self.TRANSLATIONS[self.ui_lang]["analyzing"])
                self.log_text.append(seg_translated + "\n")
        if partial:
            self.log_text.append(f"{self.TRANSLATIONS[self.ui_lang]['live_prefix']}{partial}")
            
        # Kaydır
        self.log_text.ensureCursorVisible()
        
        # Altyazıyı güncelle
        finalized_translated = [
            seg.replace("Çözümleniyor...", self.TRANSLATIONS[self.ui_lang]["analyzing"]) if seg else ""
            for seg in self.finalized_segments
        ]
        self.overlay.update_subtitles(finalized_translated, partial)

    def safe_on_speaker_update(self, event):
        segment_index = event["segment_index"]
        text = event["text"]
        
        if 0 <= segment_index < len(self.finalized_segments):
            self.finalized_segments[segment_index] = text
            
            # Log alanını güncelle
            self.log_text.clear()
            for seg in self.finalized_segments:
                if seg:
                    seg_translated = seg.replace("Çözümleniyor...", self.TRANSLATIONS[self.ui_lang]["analyzing"])
                    self.log_text.append(seg_translated + "\n")
            
            self.log_text.ensureCursorVisible()
            
            # Altyazıyı güncelle
            finalized_translated = [
                seg.replace("Çözümleniyor...", self.TRANSLATIONS[self.ui_lang]["analyzing"]) if seg else ""
                for seg in self.finalized_segments
            ]
            self.overlay.update_subtitles(finalized_translated, "")

    # ------------------------------------------------------------------ #
    #  Kayıt Kontrolleri                                                  #
    # ------------------------------------------------------------------ #

    def start_recording(self):
        device_idx = self.device_combo.currentData()
        if device_idx is None:
            self.safe_on_status_change("Seçili cihaz geçersiz.")
            return
            
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        
        self.finalized_segments = []
        self.log_text.clear()
        self.overlay.update_subtitles([], "")
        
        self.stop_event.clear()
        
        # Callbacks
        def status_cb(text):
            self.signals.status_changed.emit(text)
        def trans_cb(event):
            self.signals.transcription_received.emit(event)
        def speaker_cb(event):
            self.signals.speaker_updated.emit(event)
            
        def get_lang_pair():
            source_lang = self.source_lang_combo.currentData() or "en"
            target_lang = self.target_lang_combo.currentData() or "tr"
            return source_lang, target_lang
            
        self.pipeline_thread = threading.Thread(
            target=run,
            kwargs={
                "stop_event": self.stop_event,
                "on_status_change": status_cb,
                "on_transcription": trans_cb,
                "on_speaker_update": speaker_cb,
                "device_index": device_idx,
                "get_lang_pair": get_lang_pair,
            },
            daemon=True
        )
        self.pipeline_thread.start()
        self.tray.update_icon(is_recording=True)
        
        # Altyazı overlay'ini aktif et ve göster
        self.set_overlay_visible(True)
        
        # Kontrol panelini gizle (sistem tepsisine gönder)
        self.hide()
        
        # Kullanıcıya tepside bilgilendirme mesajı göster
        self.tray.tray.showMessage(
            "Audio Process AI",
            self.TRANSLATIONS[self.ui_lang]["tray_recording_message"],
            QtWidgets.QSystemTrayIcon.MessageIcon.Information,
            3000
        )

    def stop_recording(self):
        self.stop_btn.setEnabled(False)
        self.safe_on_status_change("Yapay zeka kapatılıyor...")
        self.stop_event.set()
        self.tray.update_icon(is_recording=False)
        self._shutdown_retries = 0
        QtCore.QTimer.singleShot(500, self.check_thread_dead)

    def check_thread_dead(self):
        self._shutdown_retries += 1
        QtWidgets.QApplication.processEvents()
        
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            if self._shutdown_retries < 10:  # max 5s (10 * 500ms)
                self.safe_on_status_change(f"Yapay zeka kapatılıyor... ({5 - self._shutdown_retries // 2}s)")
                QtCore.QTimer.singleShot(500, self.check_thread_dead)
            else:
                # Give up waiting — force UI back to ready state
                self.start_btn.setEnabled(True)
                self.device_combo.setEnabled(True)
                self.refresh_btn.setEnabled(True)
                self.source_lang_combo.setEnabled(True)
                self.target_lang_combo.setEnabled(True)
                self.ui_lang_combo.setEnabled(True)
                self.safe_on_status_change("Durduruldu (zaman aşımı).")
        else:
            self.start_btn.setEnabled(True)
            self.device_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.source_lang_combo.setEnabled(True)
            self.target_lang_combo.setEnabled(True)
            self.ui_lang_combo.setEnabled(True)
            self.safe_on_status_change("Durduruldu.")

    def show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def close_app(self):
        # UI öğelerini devre dışı bırak
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.source_lang_combo.setEnabled(False)
        self.target_lang_combo.setEnabled(False)
        self.ui_lang_combo.setEnabled(False)
        self.exit_btn.setEnabled(False)
        
        # Eğer kayıt devam ediyorsa durdur ve beklemeye başla
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.safe_on_status_change("Yapay zeka kapatılıyor...")
            self.stop_event.set()
            self._close_retries = 0
            QtCore.QTimer.singleShot(500, self.wait_and_close_app)
        else:
            self.safe_on_status_change("Sistem kapatılıyor...")
            QtCore.QTimer.singleShot(500, self.finalize_close_app)

    def wait_and_close_app(self):
        self._close_retries += 1
        QtWidgets.QApplication.processEvents()
        
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            if self._close_retries < 10:  # max 5s (10 * 500ms)
                self.safe_on_status_change(f"Yapay zeka kapatılıyor... ({5 - self._close_retries // 2}s)")
                QtCore.QTimer.singleShot(500, self.wait_and_close_app)
            else:
                self.safe_on_status_change("Zaman aşımı! Zorla kapatılıyor...")
                QtCore.QTimer.singleShot(500, self.finalize_close_app)
        else:
            self.safe_on_status_change("Sistem kapatılıyor...")
            QtCore.QTimer.singleShot(500, self.finalize_close_app)

    def finalize_close_app(self):
        self.overlay.close()
        self.tray.tray.hide()
        QtWidgets.QApplication.quit()
        sys.exit(0)

    def closeEvent(self, event):
        # Kapatma butonuna basıldığında uygulamadan çıkmak yerine tepsiye küçült
        event.ignore()
        self.hide()
        self.tray.tray.showMessage(
            "Audio Process AI",
            self.TRANSLATIONS[self.ui_lang]["tray_bg_message"],
            QtWidgets.QSystemTrayIcon.MessageIcon.Information,
            3000
        )
