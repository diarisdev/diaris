"""Konuşmacı şeridi — oturumda bulunan sesler, adlandırılabilir.

Uygulamanın tüm amacı "kim ne dedi" ama kullanıcı `SPEAKER_00`, `SPEAKER_01`
görüp kafasında eşleştirmek zorundaydı. Bu şerit oturumdaki konuşmacıları
renkleriyle listeler ve her birine isim verilmesini sağlar.

KAPSAM — bilinçli olarak dar: isim eşlemesi YALNIZCA gösterim katmanındadır.
Pipeline, tracker, izler ve AMI ölçümleri kanonik `SPEAKER_XX` etiketleriyle
çalışmaya devam eder. Böylece bir hata olsa bile en fazla yanlış isim görünür,
veri bozulmaz.

İsimler OTURUMA ÖZELDİR ve kayıt yeniden başlayınca sıfırlanır. Kalıcı olsalardı
yanlış olurdu: ikinci oturumdaki `SPEAKER_00` aynı kişi DEĞİL — etikete göre
isim hatırlamak yanlış isim göstermek demek. Sesi hatırlamak ayrı ve çok daha
büyük bir özellik.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..model_folder.speaker_palette import color_for_speaker, display_name, speaker_index


def is_nameable(label: str) -> bool:
    """Yalnızca gerçek konuşmacılar adlandırılabilir.

    "Çözümleniyor...", "[Calibrating... 12s]", "Unknown" gibi sözde etiketler
    geçici durumlardır; onlara isim vermek anlamsız olurdu.
    """
    return speaker_index(label) is not None


class FlowLayout(QtWidgets.QLayout):
    """Sığmayan öğeleri alt satıra saran yerleşim.

    Qt'de hazırı yok. Konuşmacı sayısı önceden bilinmiyor (2 de olabilir 9 da);
    sabit sütunlu bir ızgara dar pencerede taşar, tek satır ise kırpar.
    """

    def __init__(self, parent=None, spacing=8):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._layout(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _layout(self, rect: QtCore.QRect, apply: bool) -> int:
        x, y, line_height = rect.x(), rect.y(), 0
        spacing = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if apply:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class SpeakerChip(QtWidgets.QFrame):
    """Tek konuşmacı: renk noktası + (düzenlenebilir) isim + konuşma süresi."""

    renamed = QtCore.Signal(str, str)   # (kanonik etiket, yeni isim)

    def __init__(self, label: str, seconds: float, name: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("SpeakerChip")
        self.label = label
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 12, 5)
        layout.setSpacing(8)

        self.dot = QtWidgets.QLabel(self)
        self.dot.setFixedSize(10, 10)
        self.dot.setStyleSheet(
            f"background-color: {color_for_speaker(label)}; border-radius: 5px;")

        self.name_label = QtWidgets.QLabel(self)
        self.name_label.setObjectName("SpeakerChipName")

        self.name_edit = QtWidgets.QLineEdit(self)
        self.name_edit.setObjectName("SpeakerChipEdit")
        self.name_edit.setMinimumWidth(110)
        self.name_edit.hide()
        self.name_edit.editingFinished.connect(self._commit)
        self.name_edit.installEventFilter(self)

        self.duration_label = QtWidgets.QLabel(self)
        self.duration_label.setObjectName("SpeakerChipDuration")

        layout.addWidget(self.dot)
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.duration_label)

        self.set_name(name)
        self.set_seconds(seconds)

    # -- görünüm --------------------------------------------------------- #
    def set_name(self, name: str) -> None:
        self._name = (name or "").strip()
        self.name_label.setText(self._name or display_name(self.label))
        self.setToolTip(self.label if self._name else "")

    def set_seconds(self, seconds: float) -> None:
        total = int(max(0.0, seconds))
        self.duration_label.setText(f"{total // 60}:{total % 60:02d}")

    # -- düzenleme ------------------------------------------------------- #
    def begin_edit(self) -> None:
        self.name_edit.setText(self._name or display_name(self.label))
        self.name_label.hide()
        self.name_edit.show()
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _end_edit(self) -> None:
        self.name_edit.hide()
        self.name_label.show()

    def _commit(self) -> None:
        if not self.name_edit.isVisible():
            return
        text = self.name_edit.text().strip()
        # Varsayılan etikete eşitse ya da boşsa "isim yok" demektir; kullanıcı
        # böylece verdiği ismi geri alabilir.
        if text == display_name(self.label):
            text = ""
        self._end_edit()
        if text != self._name:
            self.set_name(text)
            self.renamed.emit(self.label, text)

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.name_edit and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self._end_edit()   # iptal: değişiklik yayınlanmaz
                return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.begin_edit()
            event.accept()
            return
        super().mousePressEvent(event)


class SpeakerBar(QtWidgets.QWidget):
    """Oturumdaki konuşmacıların şeridi. Konuşmacı yoksa tamamen gizlenir."""

    renamed = QtCore.Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SpeakerBar")
        self._layout = FlowLayout(self)
        self._chips: dict[str, SpeakerChip] = {}

    def update_speakers(self, durations: dict, names: dict) -> None:
        """Şeridi tazeler.

        Args:
            durations: {kanonik_etiket: toplam_saniye} — en çok konuşan önce.
            names: {kanonik_etiket: kullanıcı_ismi}
        """
        wanted = [label for label in durations if is_nameable(label)]
        wanted.sort(key=lambda label: -durations[label])

        # Artık görünmeyen konuşmacıların çipini kaldır (yeni oturum vb.).
        for label in list(self._chips):
            if label not in wanted:
                chip = self._chips.pop(label)
                self._layout.removeWidget(chip)
                chip.deleteLater()

        for label in wanted:
            chip = self._chips.get(label)
            if chip is None:
                chip = SpeakerChip(label, durations[label], names.get(label, ""), self)
                chip.renamed.connect(self.renamed)
                self._chips[label] = chip
                self._layout.addWidget(chip)
            else:
                chip.set_seconds(durations[label])
                # Düzenleme sürerken ismi ezme — kullanıcı yazıyor olabilir.
                if not chip.name_edit.isVisible():
                    chip.set_name(names.get(label, ""))

        self.setVisible(bool(wanted))
        self.updateGeometry()
