"""
Audio-process: Canlı Ses Transkripsiyon ve Konuşmacı Ayrıştırma Sistemi

Bu dosya uygulamanın giriş noktasıdır.
Kullanım: python main.py
"""

import warnings
import os

# Gereksiz uyarıları sessizle
warnings.filterwarnings("ignore")
os.environ["SYMLINK_WARNING"] = "0"

from src.pipeline import run

if __name__ == "__main__":
    run()
