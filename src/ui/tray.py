from PySide6 import QtCore, QtGui, QtWidgets

class SystemTrayController(QtCore.QObject):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        
        # Tray Icon oluşturma
        self.tray = QtWidgets.QSystemTrayIcon(self)
        
        # Dinamik olarak şık bir daire ikonu çizelim (Dosya gerektirmemesi için)
        self.update_icon(is_recording=False)
        
        # Menü kurulumu
        self.setup_menu()
        
        # Tıklama aksiyonu (Tepsiye çift tıklandığında kontrol panelini aç)
        self.tray.activated.connect(self.on_activated)
        
        # Göster
        self.tray.show()

    def update_icon(self, is_recording=False):
        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Kayıt esnasında yanıp sönen kırmızı daire, normalde yeşil/mavi daire
        if is_recording:
            color = "#e74c3c" # Kırmızı
        else:
            color = "#3498db" # Şık mavi
            
        painter.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        
        # İç kısmına şık bir beyaz halka veya nokta çizelim
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
        painter.drawEllipse(12, 12, 8, 8)
        
        painter.end()
        self.tray.setIcon(QtGui.QIcon(pixmap))

    def setup_menu(self):
        self.menu = QtWidgets.QMenu()
        
        # Menü öğeleri
        self.show_action = QtGui.QAction("Kontrol Panelini Göster", self)
        self.show_action.triggered.connect(self.main_window.show_and_raise)
        self.menu.addAction(self.show_action)
        
        self.toggle_overlay_action = QtGui.QAction("Altyazı Overlay Aç/Kapat", self)
        self.toggle_overlay_action.setCheckable(True)
        self.toggle_overlay_action.setChecked(self.main_window.overlay_visible)
        self.toggle_overlay_action.triggered.connect(self.main_window.toggle_overlay)
        self.menu.addAction(self.toggle_overlay_action)
        
        self.menu.addSeparator()
        
        self.quit_action = QtGui.QAction("Uygulamadan Çık", self)
        self.quit_action.triggered.connect(self.main_window.close_app)
        self.menu.addAction(self.quit_action)
        
        self.tray.setContextMenu(self.menu)

    def on_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self.main_window.show_and_raise()
            
    def update_overlay_state(self, is_visible):
        self.toggle_overlay_action.setChecked(is_visible)
