"""Embedding görünümü — konuşmacı kararlarını canlı olarak görselleştirir.

Üç panel, bilerek bu sırayla:

1. KARAR PANELİ (kesin)  — tracker'ın gerçekten kullandığı skorlar, etkin eşik,
   margin ve hangi kuralın ateşlediği. "İçeride mi dışarıda mı" sorusunun tam
   cevabı burasıdır.
2. BENZERLİK MATRİSİ (kesin) — konuşmacı centroid'leri arası kosinüs. Merge
   eşiğine yaklaşan, yani karışmaya aday çiftleri görünür kılar.
3. 2B DAĞILIM (yaklaşık) — 256B uzayın iki boyuta yansıması. Küme ŞEKLİ için
   sezgi verir ama karar bu grafikte alınmaz; yansıtma hatası taşıyan noktalar
   soluk çizilir (bkz. embedding_projection).

Çizim tamamen QPainter ile yapılır — matplotlib/pyqtgraph bağımlılığı eklemek
paketlemeyi (PyInstaller hidden-import + bundle boyutu) gereksiz yere zorlardı.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.embedding_projection import (
    fit_projection,
    needs_refit,
    similarity_matrix,
)
from ..model_folder.speaker_palette import color_for_speaker, display_name

# Karar etiketlerinin okunabilir karşılıkları.
DECISION_TEXT = {
    "tr": {
        "matched": "Eşleşti",
        "matched_ambiguous": "Eşleşti (belirsiz — profil güncellenmedi)",
        "sticky_short": "Kısa ses — en yakına yapıştı",
        "candidate_new": "Yeni aday kaydedildi",
        "candidate_pending": "Aday onay bekliyor",
        "candidate_promoted": "Yeni konuşmacı onaylandı",
        "unknown": "Bilinmiyor",
    },
    "en": {
        "matched": "Matched",
        "matched_ambiguous": "Matched (ambiguous — profile not updated)",
        "sticky_short": "Short utterance — stuck to nearest",
        "candidate_new": "New candidate registered",
        "candidate_pending": "Candidate awaiting confirmation",
        "candidate_promoted": "New speaker confirmed",
        "unknown": "Unknown",
    },
}

TEXT = {
    "tr": {
        "title": "Embedding Görünümü",
        "decision": "Karar",
        "matrix": "Konuşmacı benzerlik matrisi",
        "scatter": "2B dağılım (yaklaşık)",
        "refit": "Ekseni yeniden hesapla",
        "waiting": "Diarization verisi bekleniyor…",
        "warming": "Kalibrasyon sürüyor — henüz konuşmacı profili yok.",
        "no_basis": "Ayırt edici bir düzlem için en az 3 konuşmacı gerekli.",
        "stale": "Konuşmacı sayısı değişti — eksen eskidi.",
        "threshold": "etkin eşik",
        "margin": "margin",
        "chunk": "Chunk",
        "warmup_info": "Kalibrasyon: {n} embedding · {s:.1f} sn ses · {k} konuşmacı",
        "reservoir": "rezervuara eklendi",
        "approx": "Konum yaklaşıktır; kesin karar için sol panele bakın.",
        "open": "İz aç…",
        "live": "Canlıya dön",
        "play": "Oynat",
        "pause": "Duraklat",
        "speed": "Hız",
        "open_title": "Karar izi dosyası seç",
        "frame": "Kare {i}/{n}",
        "live_mode": "Canlı",
        "load_error": "İz okunamadı: {err}",
    },
    "en": {
        "title": "Embedding View",
        "decision": "Decision",
        "matrix": "Speaker similarity matrix",
        "scatter": "2D scatter (approximate)",
        "refit": "Recompute axes",
        "waiting": "Waiting for diarization data…",
        "warming": "Calibrating — no speaker profiles yet.",
        "no_basis": "At least 3 speakers are needed for a discriminative plane.",
        "stale": "Speaker count changed — axes are stale.",
        "threshold": "effective threshold",
        "margin": "margin",
        "chunk": "Chunk",
        "warmup_info": "Calibration: {n} embeddings · {s:.1f} s audio · {k} speakers",
        "reservoir": "added to reservoir",
        "approx": "Positions are approximate; see the left panel for the actual decision.",
        "open": "Open trace…",
        "live": "Back to live",
        "play": "Play",
        "pause": "Pause",
        "speed": "Speed",
        "open_title": "Choose a decision trace file",
        "frame": "Frame {i}/{n}",
        "live_mode": "Live",
        "load_error": "Could not read trace: {err}",
    },
}

# Oynatma hızları: kare/saniye. AMI replay'de bir toplantı yüzlerce karar karesi
# üretir; canlı cadans (chunk başına ~5-10 sn) burada geçerli değil, hızı
# kullanıcı belirler.
PLAYBACK_SPEEDS = (1.0, 2.0, 4.0, 8.0)


def _speaker_qcolor(label: str) -> QtGui.QColor:
    """speaker_palette rengini QColor'a çevirir — arayüzün geri kalanıyla tutarlı."""
    return QtGui.QColor(color_for_speaker(label))


def _short_label(label: str) -> str:
    """SPEAKER_03 -> S03. Dar sütunlarda (matris eksenleri) tam ad sığmıyor."""
    name = display_name(label)
    if name.startswith("SPEAKER_"):
        return "S" + name[len("SPEAKER_"):]
    return name[:6]


class _Panel(QtWidgets.QWidget):
    """Ortak çizim altyapısı: başlık, arka plan, tema renkleri."""

    def __init__(self, title_key: str, parent=None):
        super().__init__(parent)
        self.title_key = title_key
        self.lang = "tr"
        self.state = None
        self.setMinimumHeight(220)

    def set_state(self, state) -> None:
        self.state = state
        self.update()

    def set_language(self, lang: str) -> None:
        self.lang = lang if lang in TEXT else "en"
        self.update()

    # -- yardımcılar ------------------------------------------------------ #
    @property
    def strings(self) -> dict:
        return TEXT[self.lang]

    def _fg(self) -> QtGui.QColor:
        return self.palette().color(QtGui.QPalette.ColorRole.WindowText)

    def _muted(self) -> QtGui.QColor:
        color = self._fg()
        color.setAlpha(130)
        return color

    def _draw_title(self, painter: QtGui.QPainter) -> int:
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(self._fg())
        painter.drawText(12, 22, self.strings[self.title_key])
        font.setBold(False)
        painter.setFont(font)
        return 40

    def _draw_placeholder(self, painter: QtGui.QPainter, message: str) -> None:
        painter.setPen(self._muted())
        painter.drawText(
            self.rect().adjusted(16, 40, -16, -16),
            int(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap),
            message,
        )


class DecisionPanel(_Panel):
    """Panel 1 — tracker'ın gerçek sayıları. Yaklaşıklık yok."""

    BAR_HEIGHT = 20
    BAR_GAP = 8

    def __init__(self, parent=None):
        super().__init__("decision", parent)

    def paintEvent(self, event):  # noqa: N802 (Qt API)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        top = self._draw_title(painter)

        trace = (self.state or {}).get("trace")
        probes = (trace or {}).get("probes") or []
        if not probes:
            message = (self.strings["warming"] if (self.state or {}).get("warming_up")
                       else self.strings["waiting"])
            self._draw_placeholder(painter, message)
            return

        # Birden çok konuşmacılı chunk'ta en uzun sesi göster (en bilgilendirici).
        probe = max(probes, key=lambda p: p.get("duration") or 0.0)
        strings = self.strings

        painter.setPen(self._muted())
        duration = probe.get("duration")
        header = f"{strings['chunk']} #{trace.get('chunk_index', '?')}  ·  {probe['local_label']}"
        if duration is not None:
            header += f"  ·  {duration:.1f} sn  ·  kalite {probe['quality']:.2f}"
        painter.drawText(12, top, header)
        top += 22

        scores = sorted(probe["scores"].items(), key=lambda kv: kv[1], reverse=True)
        threshold = probe["effective_threshold"]
        left, right = 12, max(140, self.width() - 12)
        label_w, value_w = 96, 58
        track_left = left + label_w
        track_right = right - value_w
        track_w = max(20, track_right - track_left)

        # Skorlar [-1, 1] aralığında; görünür aralığı [0, 1]'e sıkıştırıyoruz —
        # negatif benzerlik pratikte "hiç benzemiyor" demek.
        def to_x(score: float) -> int:
            return int(track_left + track_w * max(0.0, min(1.0, score)))

        for label, score in scores:
            if top + self.BAR_HEIGHT > self.height() - 34:
                break
            color = _speaker_qcolor(label)
            is_best = label == probe.get("best")

            painter.setPen(self._fg() if is_best else self._muted())
            painter.drawText(left, top + self.BAR_HEIGHT - 5, display_name(label))

            track = QtCore.QRect(track_left, top, track_w, self.BAR_HEIGHT)
            painter.fillRect(track, QtGui.QColor(128, 128, 128, 30))

            filled = QtCore.QRect(track_left, top, to_x(score) - track_left, self.BAR_HEIGHT)
            color.setAlpha(255 if is_best else 130)
            painter.fillRect(filled, color)

            painter.setPen(self._fg() if is_best else self._muted())
            painter.drawText(track_right + 6, top + self.BAR_HEIGHT - 5, f"{score:.3f}")
            top += self.BAR_HEIGHT + self.BAR_GAP

        # Etkin eşik çizgisi — kararın gerçek sınırı.
        threshold_x = to_x(threshold)
        pen = QtGui.QPen(QtGui.QColor(220, 80, 80), 2, QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(threshold_x, 62, threshold_x, top - self.BAR_GAP)
        painter.setPen(QtGui.QColor(220, 80, 80))
        painter.drawText(threshold_x + 4, 60, f"{self.strings['threshold']} {threshold:.3f}")

        # Karar özeti
        top += 6
        margin = probe.get("margin")
        margin_text = "—" if margin is None else f"{margin:.3f}"
        painter.setPen(self._muted())
        painter.drawText(left, top + 12, f"{self.strings['margin']}: {margin_text}")

        decision = DECISION_TEXT[self.lang].get(probe["decision"], probe["decision"])
        if probe.get("reservoir_updated"):
            decision += f"  ·  {self.strings['reservoir']}"
        painter.setPen(self._fg())
        painter.drawText(left, top + 30, f"→ {display_name(probe.get('assigned') or '?')}")
        painter.setPen(self._muted())
        painter.drawText(left, top + 48, decision)


class MatrixPanel(_Panel):
    """Panel 2 — centroid'ler arası kosinüs. Karışma riskini gösterir."""

    def __init__(self, parent=None):
        super().__init__("matrix", parent)

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        top = self._draw_title(painter)

        speakers = ((self.state or {}).get("trace") or {}).get("speakers") or {}
        labels = sorted(speakers)
        if len(labels) < 2:
            self._draw_placeholder(painter, self.strings["waiting"])
            return

        matrix = similarity_matrix([speakers[label]["centroid"] for label in labels])
        merge_threshold = ((self.state or {}).get("trace") or {}).get("merge_threshold", 0.85)

        n = len(labels)
        margin_left, margin_top = 52, top + 18
        available = min(self.width() - margin_left - 16, self.height() - margin_top - 16)
        if available <= 0:
            return
        cell = max(14, int(available / n))

        for row in range(n):
            painter.setPen(self._muted())
            painter.drawText(8, margin_top + row * cell + cell // 2 + 4,
                             _short_label(labels[row]))
            for col in range(n):
                value = float(matrix[row, col])
                rect = QtCore.QRect(margin_left + col * cell, margin_top + row * cell,
                                    cell - 2, cell - 2)
                if row == col:
                    painter.fillRect(rect, QtGui.QColor(128, 128, 128, 40))
                    continue
                # Merge eşiğine yaklaşan çiftler kırmızıya kayar: bunlar
                # birleşmeye (ya da karışmaya) en yakın konuşmacılardır.
                ratio = max(0.0, min(1.0, value / max(merge_threshold, 1e-6)))
                painter.fillRect(rect, QtGui.QColor(
                    int(60 + 180 * ratio), int(120 * (1 - ratio) + 60), int(150 * (1 - ratio)),
                    int(60 + 170 * ratio),
                ))
                if cell >= 34:
                    painter.setPen(self._fg())
                    painter.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), f"{value:.2f}")


class ScatterPanel(_Panel):
    """Panel 3 — 2B yansıtma. YAKLAŞIK; konum güveni saydamlıkla kodlanır."""

    def __init__(self, parent=None):
        super().__init__("scatter", parent)
        self.projection = None
        self.setMinimumHeight(280)

    def refit(self) -> None:
        """Ekseni mevcut profillere göre yeniden kurar (kullanıcı tetikler)."""
        speakers = ((self.state or {}).get("trace") or {}).get("speakers") or {}
        centroids = [speakers[label]["centroid"] for label in sorted(speakers)]
        points = [p for label in speakers for p in speakers[label]["reservoir"]]
        self.projection = fit_projection(centroids=centroids, points=points)
        self.update()

    def set_state(self, state) -> None:
        self.state = state
        # İlk kurulum otomatik; sonraki yenilemeler kullanıcıya bırakılır —
        # kendiliğinden yeniden kurmak grafiği sürekli zıplatır.
        if self.projection is None:
            self.refit()
        self.update()

    def is_stale(self) -> bool:
        speakers = ((self.state or {}).get("trace") or {}).get("speakers") or {}
        return needs_refit(self.projection, len(speakers))

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        top = self._draw_title(painter)

        trace = (self.state or {}).get("trace") or {}
        speakers = trace.get("speakers") or {}
        if not speakers:
            self._draw_placeholder(painter, self.strings["waiting"])
            return
        if self.projection is None:
            self._draw_placeholder(painter, self.strings["no_basis"])
            return

        labels = sorted(speakers)
        vectors, meta = [], []
        for label in labels:
            vectors.append(speakers[label]["centroid"])
            meta.append((label, "centroid"))
            for point in speakers[label]["reservoir"]:
                vectors.append(point)
                meta.append((label, "reservoir"))
        for probe in trace.get("probes") or []:
            vectors.append(probe["embedding"])
            meta.append((probe.get("assigned") or "Unknown", "probe"))

        try:
            coords, in_plane = self.projection.project(vectors)
        except ValueError:
            # Embedding boyutu değişti (model değişimi) — ekseni yeniden kur.
            self.refit()
            return

        plot = QtCore.QRect(20, top + 6, max(40, self.width() - 36),
                            max(40, self.height() - top - 42))
        # Noktalar çerçevenin İÇİNE yerleşir: sağda konuşmacı adı, kenarlarda
        # işaretçi yarıçapı için pay bırakılmazsa etiketler kırpılıyor.
        inner = plot.adjusted(16, 16, -124, -16)
        span_x = np.ptp(coords[:, 0]) or 1.0
        span_y = np.ptp(coords[:, 1]) or 1.0
        min_x, min_y = coords[:, 0].min(), coords[:, 1].min()

        def to_point(xy) -> QtCore.QPointF:
            x = inner.left() + (xy[0] - min_x) / span_x * inner.width()
            y = inner.bottom() - (xy[1] - min_y) / span_y * inner.height()
            return QtCore.QPointF(x, y)

        painter.setPen(QtGui.QPen(QtGui.QColor(128, 128, 128, 60), 1))
        painter.drawRect(plot)

        # Önce rezervuar, sonra centroid, en son yeni chunk — üstte kalsın.
        order = {"reservoir": 0, "centroid": 1, "probe": 2}
        for index in sorted(range(len(meta)), key=lambda i: order[meta[i][1]]):
            label, kind = meta[index]
            point = to_point(coords[index])
            color = _speaker_qcolor(label)
            # Düzlem-dışı noktalar SOLUK: konumlarına güvenilmemeli.
            confidence = float(in_plane[index])

            if kind == "reservoir":
                color.setAlpha(int(40 + 110 * confidence))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(point, 4, 4)
            elif kind == "centroid":
                color.setAlpha(int(120 + 135 * confidence))
                painter.setPen(QtGui.QPen(color.darker(140), 2))
                painter.setBrush(color)
                painter.drawEllipse(point, 9, 9)
                painter.setPen(self._fg())
                painter.drawText(point + QtCore.QPointF(13, 4),
                                 display_name(label))
            else:
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(QtGui.QColor(240, 200, 60), 3))
                size = 9.0
                painter.drawLine(point + QtCore.QPointF(-size, 0), point + QtCore.QPointF(0, -size))
                painter.drawLine(point + QtCore.QPointF(0, -size), point + QtCore.QPointF(size, 0))
                painter.drawLine(point + QtCore.QPointF(size, 0), point + QtCore.QPointF(0, size))
                painter.drawLine(point + QtCore.QPointF(0, size), point + QtCore.QPointF(-size, 0))

        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(self._muted())
        note = self.strings["approx"]
        if self.is_stale():
            note = self.strings["stale"] + "  " + note
        painter.drawText(12, self.height() - 12, note)


class EmbeddingWindow(QtWidgets.QWidget):
    """Üç panelli teşhis penceresi — canlı veya kaydedilmiş iz oynatma modunda.

    Canlı modda güncelleme hızı diarization cadansıyla sınırlıdır (chunk başına
    ~5-10 sn) — bu, konuşmacı kararının ne sıklıkta ÜRETİLDİĞİNİN sonucudur,
    çizim hızının değil. Kaydedilmiş izlerde (AMI replay) böyle bir sınır yok:
    kareler arasında istediğin hızda gezinir ya da otomatik oynatırsın.
    """

    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lang = "tr"
        self.setWindowTitle(TEXT[self.lang]["title"])
        self.resize(1060, 680)

        self.decision_panel = DecisionPanel()
        self.matrix_panel = MatrixPanel()
        self.scatter_panel = ScatterPanel()
        self.panels = (self.decision_panel, self.matrix_panel, self.scatter_panel)

        # Oynatma durumu
        self.frames: list = []
        self.trace_warmup = None
        self.frame_index = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)

        strings = TEXT[self.lang]
        self.open_btn = QtWidgets.QPushButton(strings["open"])
        self.open_btn.clicked.connect(self.open_trace_dialog)
        self.live_btn = QtWidgets.QPushButton(strings["live"])
        self.live_btn.clicked.connect(self.return_to_live)
        self.prev_btn = QtWidgets.QToolButton()
        self.prev_btn.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn = QtWidgets.QToolButton()
        self.next_btn.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.next_btn.clicked.connect(lambda: self.step(1))
        self.play_btn = QtWidgets.QPushButton(strings["play"])
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed_combo = QtWidgets.QComboBox()
        for speed in PLAYBACK_SPEEDS:
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._restart_timer_if_playing)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.position_label = QtWidgets.QLabel(strings["live_mode"])

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.refit_btn = QtWidgets.QPushButton(strings["refit"])
        self.refit_btn.clicked.connect(self.scatter_panel.refit)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.decision_panel, 3)
        left.addWidget(self.matrix_panel, 2)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.scatter_panel, 1)

        columns = QtWidgets.QHBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        transport = QtWidgets.QHBoxLayout()
        transport.addWidget(self.open_btn)
        transport.addWidget(self.live_btn)
        transport.addWidget(self.prev_btn)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.next_btn)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.position_label)
        transport.addWidget(QtWidgets.QLabel(strings["speed"]))
        transport.addWidget(self.speed_combo)

        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(self.info_label, 1)
        footer.addWidget(self.refit_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(columns, 1)
        layout.addLayout(transport)
        layout.addLayout(footer)

        self._update_transport()

    # -- mod ------------------------------------------------------------- #
    @property
    def is_playback(self) -> bool:
        return bool(self.frames)

    def _update_transport(self) -> None:
        playback = self.is_playback
        for widget in (self.prev_btn, self.next_btn, self.play_btn, self.slider,
                       self.speed_combo, self.live_btn):
            widget.setEnabled(playback)
        strings = TEXT[self.lang]
        if playback:
            self.position_label.setText(strings["frame"].format(
                i=self.frame_index + 1, n=len(self.frames)))
        else:
            self.position_label.setText(strings["live_mode"])

    # -- iz yükleme ------------------------------------------------------ #
    def open_trace_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, TEXT[self.lang]["open_title"], "", "Trace (*.npz)")
        if path:
            self.load_trace_file(path)

    def load_trace_file(self, path) -> bool:
        """Kaydedilmiş izi yükler ve oynatma moduna geçer."""
        # Geç import: teşhis penceresi açılmadan iz modülü yüklenmesin.
        from ..core.embedding_trace import load_trace

        try:
            data = load_trace(path)
        except Exception as exc:  # bozuk/eski dosya kullanıcıya bildirilir
            self.info_label.setText(TEXT[self.lang]["load_error"].format(err=exc))
            return False

        self.stop_play()
        self.frames = data.get("frames") or []
        self.trace_warmup = data.get("warmup")
        self.frame_index = 0
        # Yeni bir kayıt = yeni embedding uzayı; eksen yeniden kurulmalı.
        self.scatter_panel.projection = None
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.frames) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        source = data.get("source") or Path(path).stem
        self.setWindowTitle(f"{TEXT[self.lang]['title']} — {source}")
        self._show_frame()
        self._update_transport()
        return True

    def return_to_live(self) -> None:
        self.stop_play()
        self.frames = []
        self.trace_warmup = None
        self.frame_index = 0
        self.scatter_panel.projection = None
        self.setWindowTitle(TEXT[self.lang]["title"])
        self._update_transport()

    # -- gezinme --------------------------------------------------------- #
    def step(self, delta: int) -> None:
        if not self.frames:
            return
        self.frame_index = max(0, min(len(self.frames) - 1, self.frame_index + delta))
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_index)
        self.slider.blockSignals(False)
        self._show_frame()
        self._update_transport()

    def _on_slider(self, value: int) -> None:
        self.frame_index = int(value)
        self._show_frame()
        self._update_transport()

    def _advance(self) -> None:
        if self.frame_index >= len(self.frames) - 1:
            self.stop_play()
            return
        self.step(1)

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.stop_play()
        else:
            self.start_play()

    def start_play(self) -> None:
        if not self.frames:
            return
        speed = self.speed_combo.currentData() or 1.0
        self._timer.start(max(30, int(1000 / speed)))
        self.play_btn.setText(TEXT[self.lang]["pause"])

    def stop_play(self) -> None:
        self._timer.stop()
        self.play_btn.setText(TEXT[self.lang]["play"])

    def _restart_timer_if_playing(self) -> None:
        if self._timer.isActive():
            self.start_play()

    def _show_frame(self) -> None:
        if not self.frames:
            return
        frame = self.frames[self.frame_index]
        self._apply_state({
            "trace": frame.get("trace"),
            "warmup": self.trace_warmup,
            "warming_up": frame.get("warming_up", False),
            "time": frame.get("time"),
        })

    # -- ortak durum uygulama -------------------------------------------- #
    def _apply_state(self, state: dict) -> None:
        for panel in self.panels:
            panel.set_state(state)

        strings = TEXT[self.lang]
        warmup = (state or {}).get("warmup")
        if warmup:
            text = strings["warmup_info"].format(
                n=warmup.get("embedding_count", 0),
                s=warmup.get("audio_ms", 0) / 1000.0,
                k=len(warmup.get("clusters") or {}),
            )
            time_sec = (state or {}).get("time")
            if time_sec is not None:
                text += f"  ·  t = {time_sec:.1f} sn"
            self.info_label.setText(text)
        elif (state or {}).get("warming_up"):
            self.info_label.setText(strings["warming"])

    def update_state(self, state: dict) -> None:
        """Canlı pipeline'dan gelen güncelleme — oynatma modundayken yok sayılır."""
        if self.is_playback:
            return
        self._apply_state(state)

    def set_language(self, lang: str) -> None:
        self.lang = lang if lang in TEXT else "en"
        strings = TEXT[self.lang]
        self.setWindowTitle(strings["title"])
        self.refit_btn.setText(strings["refit"])
        self.open_btn.setText(strings["open"])
        self.live_btn.setText(strings["live"])
        self.play_btn.setText(strings["pause"] if self._timer.isActive() else strings["play"])
        for panel in self.panels:
            panel.set_language(self.lang)
        self._update_transport()

    def closeEvent(self, event):  # noqa: N802
        self.stop_play()
        self.closed.emit()
        super().closeEvent(event)
