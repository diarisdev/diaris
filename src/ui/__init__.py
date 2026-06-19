import sys
from PySide6 import QtWidgets
from .main_window import MainWindow
from .resources import app_icon


def _set_windows_app_id():
    """Windows görev çubuğunun python.exe yerine bizim ikonu göstermesi için
    açık bir AppUserModelID ata. Windows dışında / başarısızsa sessizce atlar."""
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AudioProcess.AI.CeviriSistemi")
    except Exception:
        pass


def run_qt_app():
    _set_windows_app_id()
    app = QtWidgets.QApplication(sys.argv)

    # Uygulama-seviyesi ikon (görev çubuğu + tüm pencereler için varsayılan)
    app.setWindowIcon(app_icon())

    # Pencere kapandığında uygulamanın tamamen kapanmasını engelle (Sistem tepsisinde çalışmaya devam etmesi için)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
