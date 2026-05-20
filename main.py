"""
Audio-process: Canlı Ses Transkripsiyon ve Konuşmacı Ayrıştırma Sistemi

Bu dosya uygulamanın giriş noktasıdır.
Kullanım: python main.py, py main.py --cli(eğer GUI istemezseniz, terminalden çalışır)
"""

import warnings
import os
import sys

# Gereksiz uyarıları sessizle
warnings.filterwarnings("ignore")
os.environ["SYMLINK_WARNING"] = "0"

def main():
    if "--cli" in sys.argv:
        from src.pipeline import run
        run()
    else:
        from src.gui import AudioProcessApp
        app = AudioProcessApp()
        app.mainloop()

if __name__ == "__main__":
    main()
