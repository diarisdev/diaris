"""Kaydedilmiş konuşmacı karar izini görüntüler (bağımsız pencere).

AMI replay gerçek zamandan hızlı aktığı için koşuyu canlı izlemek işe yaramaz;
izler kaydedilir, sonra burada istenen hızda gezilir.

Kullanım:
    # 1) Replay'i iz kaydı açıkken çalıştır
    python -m tests.benchmarks.ami_replay --only IS1009a --trace-embeddings

    # 2) Üretilen izi aç
    python scripts/view_embedding_trace.py datasets/ami/ami_hyp/traces/IS1009a.npz

Dosya verilmezse pencere boş açılır; "İz aç…" ile seçebilirsin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Konuşmacı karar izi görüntüleyici.")
    parser.add_argument("trace", nargs="?", type=Path,
                        help="Görüntülenecek .npz iz dosyası (ami_replay --trace-embeddings çıktısı).")
    parser.add_argument("--lang", default="tr", choices=["tr", "en"],
                        help="Arayüz dili.")
    args = parser.parse_args()

    if args.trace is not None and not args.trace.exists():
        raise SystemExit(f"İz dosyası bulunamadı: {args.trace}")

    from PySide6 import QtWidgets

    from src.ui.embedding_window import EmbeddingWindow

    app = QtWidgets.QApplication(sys.argv)
    window = EmbeddingWindow()
    window.set_language(args.lang)
    if args.trace is not None and not window.load_trace_file(args.trace):
        raise SystemExit(f"İz okunamadı: {args.trace}")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
