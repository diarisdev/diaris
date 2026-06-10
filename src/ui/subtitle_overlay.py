import re
from PySide6 import QtCore, QtGui, QtWidgets


class ResizeGrip(QtWidgets.QSizeGrip):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        if parent and hasattr(parent, 'TRANSLATIONS'):
            self.setToolTip(parent.TRANSLATIONS[parent.ui_lang]["resize_tooltip"])
        else:
            self.setToolTip("Sağ alt köşeden boyutlandır")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 145), 2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        for offset in (8, 14, 20):
            painter.drawLine(self.width() - offset, self.height() - 5, self.width() - 5, self.height() - offset)


class SubtitleOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Sürükleme (Drag) koordinatları
        self.drag_position = QtCore.QPoint()
        
        # Seçenekler
        self.font_size = 14
        self.opacity = 0.8
        self.click_through = False
        self.speaker_coloring = True
        
        # Çeviriler
        self.TRANSLATIONS = {
            "tr": {
                "window_title": "Altyazı Overlay",
                "live": "[Canlı]:",
                "listening": "Ses dinleniyor, altyazılar burada görünecek...",
                "resize_tooltip": "Sağ alt köşeden boyutlandır"
            },
            "en": {
                "window_title": "Subtitle Overlay",
                "live": "[Live]:",
                "listening": "Listening to audio, subtitles will appear here...",
                "resize_tooltip": "Resize from bottom-right"
            }
        }
        self.ui_lang = "tr"
        
        # Pençe başlığı ve temel nitelikler
        self.setWindowTitle(self.TRANSLATIONS[self.ui_lang]["window_title"])
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint | 
            QtCore.Qt.WindowType.WindowStaysOnTopHint | 
            QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Resizing states
        self.is_resizing = False
        self.resize_edge = None
        self.resize_margin = 22
        self.min_overlay_width = 360
        self.min_overlay_height = 110
        self.setMinimumSize(self.min_overlay_width, self.min_overlay_height)
        
        # İçerik değişkenleri (Arayüz çizilmeden önce kurulmalı)
        self.segments = [] # List of dict: {"speaker": str, "text": str, "color": str}
        self.partial_text = ""
        
        # Arayüz kurulumu
        self.setup_ui()
        
        self.setMouseTracking(True)
        self.container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.resize_grip.raise_()
        
        self.update_appearance()
        
        # Ekranın alt-orta kısmına varsayılan konumlandırma
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        width = 1050
        height = 200
        x = (screen.width() - width) // 2
        y = screen.height() - height - 100
        self.setGeometry(x, y, width, height)

    def setup_ui(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
        
        # Gövde çerçevesi (Glassmorphic Container)
        self.container = QtWidgets.QFrame(self)
        self.container.setObjectName("OverlayContainer")
        self.container_layout = QtWidgets.QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(18, 16, 18, 16)
        self.container_layout.setSpacing(6)
        
        # Altyazı etiketleri için alan
        self.sub_area = QtWidgets.QWidget(self.container)
        self.sub_layout = QtWidgets.QVBoxLayout(self.sub_area)
        self.sub_layout.setContentsMargins(0, 0, 0, 0)
        self.sub_layout.setSpacing(6)
        
        self.container_layout.addWidget(self.sub_area)
        self.layout.addWidget(self.container)

        self.resize_grip = ResizeGrip(self)

    def update_appearance(self):
        # Yarı saydam arka plan stili (Glassmorphism)
        alpha = round(max(0.2, min(1.0, self.opacity)) * 255)
        border_alpha = min(255, alpha + 30)
        self.container.setStyleSheet(f"""
            QFrame#OverlayContainer {{
                background-color: rgba(12, 15, 20, {alpha});
                border: 1px solid rgba(255, 255, 255, {border_alpha // 5});
                border-radius: 14px;
            }}
        """)
        self.render_subtitles()

    def set_font_size(self, size):
        self.font_size = size
        self.render_subtitles()

    def set_overlay_opacity(self, opacity):
        self.opacity = opacity
        self.update_appearance()

    def set_click_through(self, enabled):
        self.click_through = enabled
        flags = self.windowFlags()
        if enabled:
            # Tıklamaların alttaki pencerelere geçmesi için
            self.setWindowFlags(flags | QtCore.Qt.WindowType.WindowTransparentForInput)
        else:
            # Tıklamaları yakalayabilmesi için (sürükleme desteği)
            self.setWindowFlags(flags & ~QtCore.Qt.WindowType.WindowTransparentForInput)
        
        # Bayraklar güncellendikten sonra pencereyi tekrar göstermek gerekir
        self.show()

    def set_speaker_coloring(self, enabled):
        """Konuşmacı renginin tüm altyazı metnini boyayıp boyamayacağını ayarlar."""
        self.speaker_coloring = enabled
        self.render_subtitles()

    def set_language(self, lang):
        if lang in self.TRANSLATIONS:
            self.ui_lang = lang
            self.setWindowTitle(self.TRANSLATIONS[lang]["window_title"])
            if hasattr(self, 'resize_grip'):
                self.resize_grip.setToolTip(self.TRANSLATIONS[lang]["resize_tooltip"])
            self.render_subtitles()

    def update_subtitles(self, finalized_segments, partial_text):
        """Yeni altyazıları kabul eder ve arayüzde gösterir."""
        # Son 2 kesinleşmiş segmenti gösterelim
        self.segments = []
        
        # Renk paleti (Dinamik ve yumuşak renkler)
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22"]
        
        for idx, text in enumerate(finalized_segments[-2:]):
            speaker_tag = ""
            content = text
            
            # Match formats: "[SPEAKER_00] 1.2s - 3.4s: Hello" or "[[Calibrating... 35s]] 0.0s - 10.0s: Hello"
            match = re.match(r"^\[(.*)\] \d+\.\d+s\s+-\s+\d+\.\d+s:\s+(.*)$", text)
            if match:
                spk = match.group(1).strip()
                content = match.group(2).strip()
                
                # If nested brackets (e.g. [Calibrating... 35s]), strip the outer brackets
                if spk.startswith("[") and spk.endswith("]"):
                    spk = spk[1:-1].strip()
                speaker_tag = spk
            
            # Konuşmacıya göre renk atama
            spk_id = 0
            if "SPEAKER_" in speaker_tag:
                try:
                    spk_id = int(speaker_tag.split("_")[1])
                except Exception:
                    pass
            color = colors[spk_id % len(colors)]
            
            self.segments.append({
                "speaker": speaker_tag,
                "text": content,
                "color": color
            })
            
        self.partial_text = partial_text
        self.render_subtitles()

    def render_subtitles(self):
        # Temizle
        while self.sub_layout.count() > 0:
            item = self.sub_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Kesinleşmiş segmentleri ekle
        for seg in self.segments:
            row = QtWidgets.QWidget(self.sub_area)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)
            
            # Konuşmacı Etiketi
            if seg['speaker']:
                spk_label = QtWidgets.QLabel(f"{seg['speaker']}:", row)
                spk_label.setFont(QtGui.QFont("Segoe UI", self.font_size - 2, QtGui.QFont.Weight.Bold))
                spk_label.setStyleSheet(f"color: {seg['color']}; background: transparent; border: none;")
                spk_label.setMinimumWidth(92)
                spk_label.setMaximumWidth(132)
                row_layout.addWidget(spk_label, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            
            # Metin
            text_color = seg['color'] if self.speaker_coloring else "#ffffff"
            text_label = QtWidgets.QLabel(seg['text'], row)
            text_label.setFont(QtGui.QFont("Segoe UI", self.font_size))
            text_label.setStyleSheet(f"color: {text_color}; background: transparent; border: none;")
            text_label.setWordWrap(True)
            
            row_layout.addWidget(text_label, 1)
            self.sub_layout.addWidget(row)
            
        # Canlı/Anlık (Partial) metni ekle
        if self.partial_text:
            row = QtWidgets.QWidget(self.sub_area)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)
            
            spk_label = QtWidgets.QLabel(self.TRANSLATIONS[self.ui_lang]["live"], row)
            spk_label.setFont(QtGui.QFont("Segoe UI", self.font_size - 2, QtGui.QFont.Weight.Bold))
            spk_label.setStyleSheet("color: #aaaaaa; background: transparent; border: none;")
            spk_label.setMinimumWidth(92)
            spk_label.setMaximumWidth(132)
            
            text_label = QtWidgets.QLabel(self.partial_text, row)
            text_label.setFont(QtGui.QFont("Segoe UI", self.font_size, QtGui.QFont.Weight.Medium))
            text_label.setStyleSheet("color: #dddddd; background: transparent; border: none; font-style: italic;")
            text_label.setWordWrap(True)
            
            row_layout.addWidget(spk_label, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            row_layout.addWidget(text_label, 1)
            self.sub_layout.addWidget(row)
            
        # Eğer içerik tamamen boşsa bilgilendirme göster
        if not self.segments and not self.partial_text:
            info_label = QtWidgets.QLabel(self.TRANSLATIONS[self.ui_lang]["listening"], self.sub_area)
            info_label.setFont(QtGui.QFont("Segoe UI", self.font_size - 2, QtGui.QFont.Weight.Light))
            info_label.setStyleSheet("color: #888888; background: transparent; border: none;")
            info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.sub_layout.addWidget(info_label)

    # ------------------------------------------------------------------ #
    #  Sürükleme ve Boyutlandırma Mantığı (Drag & Resize)                #
    # ------------------------------------------------------------------ #

    def handle_press(self, global_pos, button):
        if button == QtCore.Qt.MouseButton.LeftButton:
            local_pos = self.mapFromGlobal(global_pos)
            action = self.get_drag_action(local_pos)
            if action.startswith("resize"):
                self.is_resizing = True
                self.resize_edge = action
                self.drag_start_geometry = self.geometry()
                self.drag_start_global_pos = global_pos
            else:
                self.is_resizing = False
                self.drag_position = global_pos - self.frameGeometry().topLeft()

    def handle_move(self, global_pos, buttons):
        local_pos = self.mapFromGlobal(global_pos)
        if buttons == QtCore.Qt.MouseButton.NoButton:
            action = self.get_drag_action(local_pos)
            if action == "resize_bottom_right":
                self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
            elif action == "resize_right":
                self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            elif action == "resize_bottom":
                self.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
            if self.is_resizing:
                delta = global_pos - self.drag_start_global_pos
                new_geom = QtCore.QRect(self.drag_start_geometry)
                if "right" in self.resize_edge:
                    new_w = max(self.min_overlay_width, self.drag_start_geometry.width() + delta.x())
                    new_geom.setWidth(new_w)
                if "bottom" in self.resize_edge:
                    new_h = max(self.min_overlay_height, self.drag_start_geometry.height() + delta.y())
                    new_geom.setHeight(new_h)
                self.setGeometry(new_geom)
            else:
                self.move(global_pos - self.drag_position)

    def handle_release(self):
        self.is_resizing = False
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def get_drag_action(self, pos):
        if not self.rect().contains(pos):
            return "drag"

        is_right = pos.x() >= self.width() - self.resize_margin
        is_bottom = pos.y() >= self.height() - self.resize_margin
        
        if is_right and is_bottom:
            return "resize_bottom_right"
        elif is_right:
            return "resize_right"
        elif is_bottom:
            return "resize_bottom"
        else:
            return "drag"

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if not self.click_through:
            self.handle_press(event.globalPosition().toPoint(), event.button())
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if not self.click_through:
            self.handle_move(event.globalPosition().toPoint(), event.buttons())
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if not self.click_through:
            self.handle_release()
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 12
        self.resize_grip.move(
            self.width() - self.resize_grip.width() - margin,
            self.height() - self.resize_grip.height() - margin,
        )
        self.resize_grip.raise_()
