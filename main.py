"""
Audio-process: Canlı Ses Transkripsiyon ve Konuşmacı Ayrıştırma Sistemi

Bu dosya uygulamanın giriş noktasıdır.
Kullanım: python main.py, py main.py --cli(eğer GUI istemezseniz, terminalden çalışır)
"""

import os
import sys
import warnings

from src.config import configure_cuda_dll_paths, ensure_output_dir

# Gereksiz uyarıları sessizle
warnings.filterwarnings("ignore")
os.environ["SYMLINK_WARNING"] = "0"

def main():
    configure_cuda_dll_paths()
    ensure_output_dir()

    if "--cli" in sys.argv:
        from src.pipeline import run
        run(allow_interactive_device=True)
    elif "--tk" in sys.argv or "--tkinter" in sys.argv:
        from src.gui import AudioProcessApp
        app = AudioProcessApp()
        app.mainloop()
    else:
        from src.ui import run_qt_app
        run_qt_app()

if __name__ == "__main__":
    main()
