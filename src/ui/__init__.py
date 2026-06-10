import sys
from PySide6 import QtWidgets
from .main_window import MainWindow

def run_qt_app():
    app = QtWidgets.QApplication(sys.argv)
    
    # Pencere kapandığında uygulamanın tamamen kapanmasını engelle (Sistem tepsisinde çalışmaya devam etmesi için)
    app.setQuitOnLastWindowClosed(False)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
