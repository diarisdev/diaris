"""UI kaynakları — uygulama ikonu çözümü.

İkon dosyasını `src/ui/assets/` altına koy: tercihen `icon.png` (Qt'de en sağlamı),
ya da `icon.svg` / `icon.ico`. Dosya yoksa boş QIcon döner (uygulama yine çalışır).
"""

from pathlib import Path

from PySide6 import QtGui

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Aranacak isimler (öncelik sırası). PNG en sağlam; PySide6 SVG'yi de okuyabilir.
_ICON_NAMES = ("icon.png", "icon.svg", "icon.ico")


def app_icon_path() -> Path | None:
    for name in _ICON_NAMES:
        p = ASSETS_DIR / name
        if p.exists():
            return p
    return None


def app_icon() -> QtGui.QIcon:
    """Uygulama ikonu (yoksa boş QIcon)."""
    p = app_icon_path()
    return QtGui.QIcon(str(p)) if p else QtGui.QIcon()
