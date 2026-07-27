"""Diaris ikon üreteci — docs/assets/icon.svg ile AYNI tasarımı PNG/ICO üretir.

Neden ayrı bir script: SVG web/README için ideal ama Qt ikonu PNG, Windows exe
ikonu ICO ister. Tasarımı elle iki kez çizmek yerine burada tek kaynaktan
(aşağıdaki sabitler) tüm formatlar üretilir — SVG ile birebir aynı geometri.

Çalıştırma (proje kökünden):
    python scripts/make_icons.py

Üretilenler:
    src/ui/assets/icon.png    512x512  — Qt uygulama ikonu
    src/ui/assets/icon.ico    çok boyutlu — Windows .exe ikonu
    docs/assets/icon-512.png  512x512  — GitHub/e-posta profil fotoğrafı
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

# --- Tasarım (docs/assets/icon.svg ile birebir aynı) ---
SIZE = 512
RADIUS = 112
BG_TOP = (20, 29, 46)      # #141d2e
BG_BOTTOM = (15, 52, 96)   # #0f3460

# (x, y, genişlik, yükseklik, renk) — renk GRUPLARI farklı konuşmacıları temsil eder
BARS = [
    (94, 201, 44, 110, (52, 152, 219)),    # #3498db mavi
    (164, 151, 44, 210, (52, 152, 219)),
    (234, 106, 44, 300, (46, 204, 113)),   # #2ecc71 yeşil
    (304, 161, 44, 190, (155, 89, 182)),   # #9b59b6 mor
    (374, 196, 44, 120, (155, 89, 182)),
]

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
SS = 4  # supersampling — kenar yumuşatma için 4x çizip küçültüyoruz


def _vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    """Dikey gradyan arka plan."""
    grad = Image.new("RGB", (1, size))
    px = grad.load()
    for y in range(size):
        t = y / max(1, size - 1)
        px[0, y] = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return grad.resize((size, size), Image.Resampling.BILINEAR)


def render(size: int = SIZE) -> Image.Image:
    """İkonu `size` çözünürlükte, supersampling ile üretir."""
    s = size * SS
    scale = s / SIZE

    canvas = _vertical_gradient(s, BG_TOP, BG_BOTTOM).convert("RGBA")

    # Yuvarlatılmış kare maskesi (app-icon görünümü)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, s - 1, s - 1], radius=int(RADIUS * scale), fill=255
    )

    # Dalga çubukları
    bars = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bars)
    for x, y, w, h, color in BARS:
        x0, y0 = x * scale, y * scale
        x1, y1 = (x + w) * scale, (y + h) * scale
        bd.rounded_rectangle([x0, y0, x1, y1], radius=(w / 2) * scale, fill=color + (255,))
    canvas = Image.alpha_composite(canvas, bars)

    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.paste(canvas, (0, 0), mask)
    return out.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    ui_assets = ROOT / "src" / "ui" / "assets"
    docs_assets = ROOT / "docs" / "assets"
    ui_assets.mkdir(parents=True, exist_ok=True)
    docs_assets.mkdir(parents=True, exist_ok=True)

    icon = render(SIZE)

    png_path = ui_assets / "icon.png"
    icon.save(png_path)
    print(f"[OK] {png_path}  (Qt uygulama ikonu)")

    ico_path = ui_assets / "icon.ico"
    icon.save(ico_path, format="ICO", sizes=[(n, n) for n in ICO_SIZES])
    print(f"[OK] {ico_path}  (Windows .exe ikonu, {len(ICO_SIZES)} boyut)")

    profile_path = docs_assets / "icon-512.png"
    icon.save(profile_path)
    print(f"[OK] {profile_path}  (GitHub / e-posta profil fotografi)")

    print("\nProfil fotoğrafı olarak docs/assets/icon-512.png dosyasını yükleyin.")


if __name__ == "__main__":
    main()
