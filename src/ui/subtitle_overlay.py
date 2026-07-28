import re
from PySide6 import QtCore, QtGui, QtWidgets

from ..model_folder.speaker_palette import color_for_speaker, display_name


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


class ScrollToBottomButton(QtWidgets.QPushButton):
    """Yuvarlak "en alta dön" düğmesi.

    Overlay'in cam diline sadık: koyu yarı saydam daire, ince beyaz kenar,
    beyaz aşağı ok. Yalnızca kullanıcı geçmişe kaydırdığında görünür.
    """

    SIZE = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setFlat(True)
        # Varsayılan buton çerçevesi cam görünümü bozar; tüm çizim paintEvent'te.
        self.setStyleSheet("background: transparent; border: none;")
        self._hover = False

    def enterEvent(self, event):  # noqa: N802 (Qt API)
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Cam daire — container ile aynı taban renk (12, 15, 20).
        painter.setBrush(QtGui.QColor(12, 15, 20, 235 if self._hover else 200))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 80 if self._hover else 50), 1))
        painter.drawEllipse(QtCore.QRectF(1, 1, self.width() - 2, self.height() - 2))

        # Aşağı ok
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 235), 2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        painter.drawLine(QtCore.QPointF(cx, cy - 6.0), QtCore.QPointF(cx, cy + 4.5))
        painter.drawPolyline(QtGui.QPolygonF([
            QtCore.QPointF(cx - 5.0, cy - 0.5),
            QtCore.QPointF(cx, cy + 5.0),
            QtCore.QPointF(cx + 5.0, cy - 0.5),
        ]))


class SubtitleOverlay(QtWidgets.QWidget):
    # Overlay'e çift tıklandığında ana kontrol panelini geri getirmek için yayınlanır
    restore_requested = QtCore.Signal()

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
        # Geçmiş yalnızca GERÇEKTEN değiştiğinde yeniden kurulur. Partial saniyede
        # birkaç kez yenileniyor; her seferinde tüm satırları yıkıp kurmak, geçmiş
        # büyüdükçe hızla pahalılaşırdı (ve kaydırma konumunu bozardı).
        self._last_finalized = None
        # Kullanıcı en alttayken yeni satır geldikçe otomatik kayar; yukarı
        # kaydırdıysa konumu korunur ve "en alta dön" düğmesi çıkar.
        self._stick_to_bottom = True
        # {kanonik_etiket: kullanıcı_ismi} — ana panelden gelir. Yalnızca
        # gösterimi etkiler; overlay'in tasarımı ve mantığı değişmez.
        self.speaker_names = {}

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
        
        # Kesinleşmiş altyazı geçmişi — kaydırılabilir.
        self.scroll_area = QtWidgets.QScrollArea(self.container)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.sub_area = QtWidgets.QWidget()
        self.sub_layout = QtWidgets.QVBoxLayout(self.sub_area)
        self.sub_layout.setContentsMargins(0, 0, 0, 0)
        self.sub_layout.setSpacing(6)
        self.scroll_area.setWidget(self.sub_area)

        # Fare olayları: container gibi kaydırma alanı ve içeriği de ŞEFFAF.
        # Böylece overlay'i metnin üzerinden sürükleme davranışı korunur
        # (bu pencerenin taşınabilir olması altyazı deneyiminin temel parçası).
        # Tekerlek bunun yerine overlay'in wheelEvent'inde ele alınır.
        for widget in (self.scroll_area, self.scroll_area.viewport(), self.sub_area):
            widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_scroll_changed)
        # Yeni satır eklenince menzil değişir; en alttaysak orada kalmalıyız.
        scrollbar.rangeChanged.connect(self._on_scroll_range_changed)

        # Canlı (partial) satır kaydırma alanının DIŞINDA, altta sabit durur:
        # geçmişe bakarken bile "şu an ne söyleniyor" görünür kalsın ve saniyede
        # birkaç kez yenilenen bu satır geçmişi yeniden kurdurmasın.
        self.partial_row = QtWidgets.QWidget(self.container)
        partial_layout = QtWidgets.QHBoxLayout(self.partial_row)
        partial_layout.setContentsMargins(0, 0, 0, 0)
        partial_layout.setSpacing(10)
        self.partial_speaker_label = QtWidgets.QLabel(
            self.TRANSLATIONS[self.ui_lang]["live"], self.partial_row)
        self.partial_speaker_label.setMinimumWidth(92)
        self.partial_speaker_label.setMaximumWidth(132)
        self.partial_text_label = QtWidgets.QLabel("", self.partial_row)
        self.partial_text_label.setWordWrap(True)
        partial_layout.addWidget(self.partial_speaker_label, 0,
                                 QtCore.Qt.AlignmentFlag.AlignTop)
        partial_layout.addWidget(self.partial_text_label, 1)
        self.partial_row.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.partial_row.hide()

        self.container_layout.addWidget(self.scroll_area, 1)
        self.container_layout.addWidget(self.partial_row, 0)
        self.layout.addWidget(self.container)

        # "En alta dön": overlay'in DOĞRUDAN çocuğu (resize_grip gibi) —
        # container şeffaf olduğu için tıklamayı ancak böyle alabilir.
        self.scroll_bottom_btn = ScrollToBottomButton(self)
        self.scroll_bottom_btn.clicked.connect(self.scroll_to_bottom)
        self.scroll_bottom_btn.hide()

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
        # Kaydırma alanı camın üstünde durur → tamamen saydam. Çubuk ince ve
        # sönük: konum göstergesi gibi okunsun, tasarımı bölmesin.
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical {
                background: transparent;
                width: 5px;
                margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 55);
                border-radius: 2px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)
        self.render_subtitles()

    def set_font_size(self, size):
        self.font_size = size
        self._update_partial_row()
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
        # Tıklama-geçirgen modda düğme tıklanamaz; göstermek yanıltıcı olur.
        self._update_scroll_button()

    def set_speaker_names(self, names):
        """Kullanıcının verdiği konuşmacı isimlerini uygular.

        Tasarıma dokunmaz — yalnızca konuşmacı sütununda gösterilen metin
        değişir. Renkler kanonik etiketten geldiği için aynı kalır.
        """
        self.speaker_names = dict(names or {})
        self._last_finalized = None   # yeniden ayrıştırmayı zorla
        self.render_subtitles()

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
            self._update_partial_row()
            self.render_subtitles()

    # Overlay'de tutulan en fazla kesinleşmiş satır. Geçmiş artık kaydırılabilir,
    # ama sınırsız değil: her satır ayrı bir widget ve çok uzun oturumlarda
    # layout maliyeti birikirdi. Kontrol panelindeki günlük tam geçmişi tutar.
    MAX_HISTORY_LINES = 300

    _LINE_RE = re.compile(r"^\[(.*)\] \d+\.\d+s\s+-\s+\d+\.\d+s:\s+(.*)$")

    def _parse_segments(self, finalized_segments):
        """Formatlanmış segment metinlerini satır satır ayrıştırır.

        Bir chunk BİRDEN ÇOK konuşmacı satırı içerebilir (diarization cümleyi
        konuşmacı sınırından böler → "[SPEAKER_00] ...\n[SPEAKER_01] ...").
        Her satır kendi konuşmacı rengiyle ayrı satır olur; renkler ortak
        speaker_palette'ten gelir → tüm bileşenlerle tutarlı.

        Son MAX_HISTORY_LINES satır tutulur (sondan başa toplanıp ters çevrilir).
        """
        parsed_tail = []
        for text in reversed(finalized_segments):
            if len(parsed_tail) >= self.MAX_HISTORY_LINES:
                break
            if not text:
                continue
            for line in reversed(text.split("\n")):
                line = line.strip()
                if not line:
                    continue
                speaker_tag = ""
                content = line
                match = self._LINE_RE.match(line)
                if match:
                    spk = match.group(1).strip()
                    content = match.group(2).strip()
                    # İç içe parantez (örn. [Calibrating... 35s]) → dış parantezi soy
                    if spk.startswith("[") and spk.endswith("]"):
                        spk = spk[1:-1].strip()
                    speaker_tag = spk
                parsed_tail.append((speaker_tag, content))
                if len(parsed_tail) >= self.MAX_HISTORY_LINES:
                    break

        # Ardışık satır aynı konuşmacıya aitse etiketi tekrarlamayız; satır
        # konuşmacı sütunu kadar girintilenir, böylece metin hizalı kalır.
        segments = []
        prev_tag = None
        for speaker_tag, content in reversed(parsed_tail):
            is_continuation = bool(speaker_tag) and speaker_tag == prev_tag
            prev_tag = speaker_tag
            # Kullanıcı isim verdiyse onu göster; renk kanonik etiketten gelir.
            shown = self.speaker_names.get(speaker_tag) or (
                display_name(speaker_tag) if speaker_tag else "")
            segments.append({
                "speaker": "" if is_continuation else shown,
                "text": content,
                "color": color_for_speaker(speaker_tag),
                "continuation": is_continuation,
            })
        return segments

    def update_subtitles(self, finalized_segments, partial_text):
        """Yeni altyazıları kabul eder ve arayüzde gösterir.

        Geçmiş YALNIZCA kesinleşmiş segmentler değiştiğinde yeniden kurulur.
        Partial saniyede birkaç kez yenilenir; onu her seferinde tüm geçmişi
        yıkıp kurarak göstermek hem pahalı olurdu hem de kullanıcının kaydırma
        konumunu her saniye sıfırlardı.
        """
        self.partial_text = partial_text
        self._update_partial_row()

        if finalized_segments == self._last_finalized:
            return

        self._last_finalized = list(finalized_segments)
        self.segments = self._parse_segments(finalized_segments)
        if not finalized_segments:
            # Yeni oturum (panel listeyi sıfırlar): önceki oturumda yukarı
            # kaydırılmışsa kullanıcı takibi kaybetmiş olarak başlamasın.
            self._stick_to_bottom = True
        self.render_subtitles()

    def render_subtitles(self):
        # Yeniden kurulum boyunca repaint'i askıya al: satır başına ayrı ayrı
        # boyama yerine sonda TEK repaint (görsel çıktı birebir aynı).
        self.sub_area.setUpdatesEnabled(False)
        try:
            self._rebuild_subtitle_rows()
        finally:
            self.sub_area.setUpdatesEnabled(True)

        # Kaydırma menzili layout hesaplandıktan SONRA netleşir; en alta çekmeyi
        # bir sonraki olay döngüsüne bırakıyoruz (rangeChanged'e ek güvence).
        if self._stick_to_bottom:
            QtCore.QTimer.singleShot(0, self._scroll_to_bottom_if_sticky)
        else:
            QtCore.QTimer.singleShot(0, self._update_scroll_button)

    def _scroll_to_bottom_if_sticky(self):
        if self._stick_to_bottom:
            bar = self.scroll_area.verticalScrollBar()
            bar.setValue(bar.maximum())
        self._update_scroll_button()

    def _rebuild_subtitle_rows(self):
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
            elif seg.get('continuation'):
                # Aynı konuşmacının devam satırı: etiket tekrarlanmaz ama metin
                # üstteki satırla hizalı kalsın diye konuşmacı sütunu kadar boşluk.
                spacer = QtWidgets.QLabel("", row)
                spacer.setMinimumWidth(92)
                spacer.setMaximumWidth(132)
                row_layout.addWidget(spacer, 0, QtCore.Qt.AlignmentFlag.AlignTop)

            # Metin
            text_color = seg['color'] if self.speaker_coloring else "#ffffff"
            text_label = QtWidgets.QLabel(seg['text'], row)
            text_label.setFont(QtGui.QFont("Segoe UI", self.font_size))
            text_label.setStyleSheet(f"color: {text_color}; background: transparent; border: none;")
            text_label.setWordWrap(True)
            
            row_layout.addWidget(text_label, 1)
            self.sub_layout.addWidget(row)

        # Eğer içerik tamamen boşsa bilgilendirme göster
        if not self.segments and not self.partial_text:
            info_label = QtWidgets.QLabel(self.TRANSLATIONS[self.ui_lang]["listening"], self.sub_area)
            info_label.setFont(QtGui.QFont("Segoe UI", self.font_size - 2, QtGui.QFont.Weight.Light))
            info_label.setStyleSheet("color: #888888; background: transparent; border: none;")
            info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.sub_layout.addWidget(info_label)

        # Satırlar yukarıdan paketlensin (kaydırma alanı içeriği germesin).
        self.sub_layout.addStretch(1)

    def _update_partial_row(self):
        """Canlı satırı yerinde günceller — geçmiş yeniden kurulmaz."""
        self.partial_speaker_label.setText(self.TRANSLATIONS[self.ui_lang]["live"])
        self.partial_speaker_label.setFont(
            QtGui.QFont("Segoe UI", self.font_size - 2, QtGui.QFont.Weight.Bold))
        self.partial_speaker_label.setStyleSheet(
            "color: #aaaaaa; background: transparent; border: none;")

        self.partial_text_label.setText(self.partial_text)
        self.partial_text_label.setFont(
            QtGui.QFont("Segoe UI", self.font_size, QtGui.QFont.Weight.Medium))
        self.partial_text_label.setStyleSheet(
            "color: #dddddd; background: transparent; border: none; font-style: italic;")

        self.partial_row.setVisible(bool(self.partial_text))

    # ------------------------------------------------------------------ #
    #  Kaydırma                                                           #
    # ------------------------------------------------------------------ #

    # En altta sayılmak için tolerans (px). Tam eşitlik beklemek, kesirli
    # yükseklikler yüzünden otomatik kaydırmayı rastgele kapatırdı.
    BOTTOM_EPSILON = 4

    def _is_at_bottom(self) -> bool:
        bar = self.scroll_area.verticalScrollBar()
        return bar.value() >= bar.maximum() - self.BOTTOM_EPSILON

    def scroll_to_bottom(self):
        """En güncel altyazıya döner ve otomatik takibi yeniden açar."""
        bar = self.scroll_area.verticalScrollBar()
        self._stick_to_bottom = True
        bar.setValue(bar.maximum())
        self._update_scroll_button()

    def _on_scroll_changed(self, _value):
        # Kullanıcı yukarı kaydırdıysa takip kapanır; en alta dönerse geri açılır.
        self._stick_to_bottom = self._is_at_bottom()
        self._update_scroll_button()

    def _on_scroll_range_changed(self, _minimum, _maximum):
        # Yeni satır geldi: en alttaysak orada kal (otomatik kaydırma).
        if self._stick_to_bottom:
            bar = self.scroll_area.verticalScrollBar()
            bar.setValue(bar.maximum())
        self._update_scroll_button()

    def _update_scroll_button(self):
        """Düğme yalnızca kaydırılacak geçmiş VARKEN ve yukarıdayken görünür."""
        bar = self.scroll_area.verticalScrollBar()
        should_show = (not self.click_through
                       and bar.maximum() > 0
                       and not self._is_at_bottom())
        if should_show:
            self._position_scroll_button()
            self.scroll_bottom_btn.show()
            self.scroll_bottom_btn.raise_()
        else:
            self.scroll_bottom_btn.hide()

    def _position_scroll_button(self):
        """Sağ altta, boyutlandırma tutamacının solunda."""
        margin = 12
        x = self.width() - self.scroll_bottom_btn.width() - self.resize_grip.width() - margin - 8
        y = self.height() - self.scroll_bottom_btn.height() - margin - 2
        self.scroll_bottom_btn.move(max(margin, x), max(margin, y))

    def wheelEvent(self, event: QtGui.QWheelEvent):  # noqa: N802
        """Tekerlek geçmişi kaydırır.

        Kaydırma alanı fare olaylarına şeffaf (overlay her yerinden
        sürüklenebilsin diye), bu yüzden tekerleği burada elle iletiyoruz.
        """
        if self.click_through:
            event.ignore()
            return
        bar = self.scroll_area.verticalScrollBar()
        if bar.maximum() <= 0:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta:
            # Bir "tık" (120) ≈ üç satır adımı — Qt'nin varsayılan davranışı.
            bar.setValue(bar.value() - int(delta / 120.0) * bar.singleStep() * 3)
        event.accept()

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

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        # Çift tıklama: ikon haline küçülmüş ana kontrol panelini geri getir
        if not self.click_through and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.restore_requested.emit()
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 12
        self.resize_grip.move(
            self.width() - self.resize_grip.width() - margin,
            self.height() - self.resize_grip.height() - margin,
        )
        self.resize_grip.raise_()
        # Yükseklik değişince kaydırılabilir alan da değişir → düğmeyi yeniden
        # konumlandır ve görünürlüğünü tazele.
        self._position_scroll_button()
        self._update_scroll_button()
