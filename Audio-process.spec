# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — Audio-process (onedir).

CPU/GPU AYRIMI: bu spec her iki varyant için AYNIDIR. Fark yalnızca build
ortamındaki torch tekerleğidir (CPU torch vs CUDA torch). İki temiz venv'den
aynı spec'i çalıştırmak iki varyantı üretir — spec çatallanmaz.

Çalıştırma (proje kökünden, hedef torch'un kurulu olduğu venv'de):
    pyinstaller Audio-process.spec --noconfirm

Çıktı: dist/AudioProcess/  (onedir — exe + _internal/). ASLA onefile değil:
onefile her açılışta GB'larca bundle'ı temp'e açar; onedir anında başlar.

NEDEN onedir + harici modeller:
  * models/ (GB'larca) ve .env bundle'a GÖMÜLMEZ; exe'nin yanında dururlar
    (config._resolve_project_root donmuş modda exe dizinini kök alır).
  * Böylece installer küçük kalır ve modeller ilk açılışta indirilebilir.
"""
import os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_dynamic_libs,
    collect_submodules,
)

# AP_BUILD_CONSOLE=1 ile konsollu (hata ayıklama) build. İlk paketleme
# denemelerinde AÇIK tutun: GUI açılışta eksik bir import'tan çökerse konsol
# traceback'i gösterir. Çalışan bir build elde edince kapatın (yalın GUI).
_console = os.environ.get("AP_BUILD_CONSOLE", "0") == "1"

datas = []
binaries = []
hiddenimports = []

# --- Dinamik-import yapan paketler: alt-modül + veri + native kütüphaneleri
#     TAMAMEN topla. PyInstaller'ın import grafiği bunları tek başına bulamaz;
#     pyannote/lightning ekosisteminin en sık paketleme kırılma noktasıdır. ---
_COLLECT_ALL = [
    "pyannote",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.database",
    "pyannote.metrics",
    "pyannote.pipeline",
    "lightning",
    "lightning_fabric",
    "pytorch_lightning",
    "torchmetrics",
    "asteroid_filterbanks",
    "torch_audiomentations",
    "torch_pitch_shift",
    "julius",
    "primePy",
    "speechbrain",
    "onnxruntime",
    "ctranslate2",
    "faster_whisper",
    "sentencepiece",
    "huggingface_hub",
    "transformers",
    "einops",
]
for pkg in _COLLECT_ALL:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Paket kurulu değilse (örn. transformers opsiyonel) sessizce atla.
        pass

# --- Native kütüphane bağımlılıkları ---
for pkg in ("soundfile", "pyaudiowpatch"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# --- CUDA runtime DLL'leri (yalnız GPU build; CPU venv'de nvidia wheel'leri yok) ---
# ctranslate2 (faster-whisper) ve torch, cublas64_12.dll / cudnn*.dll gibi
# kütüphaneleri ÇALIŞMA ANINDA adla yükler; PyInstaller bağımlılık taraması
# bunları göremez, o yüzden ELLE topluyoruz. Bundle KÖKÜNE (".") konur ki
# PyInstaller o dizini DLL arama yoluna eklesin (config.py de ekler).
import glob as _glob
import site as _site

try:
    _site_dirs = list(_site.getsitepackages())
except AttributeError:
    _site_dirs = []
_cuda_dll_count = 0
for _sp in _site_dirs:
    # nvidia-*-cu12 wheel'leri: site-packages/nvidia/<lib>/bin/*.dll
    for _dll in _glob.glob(os.path.join(_sp, "nvidia", "*", "bin", "*.dll")):
        binaries.append((_dll, "."))
        _cuda_dll_count += 1
    # torch kendi CUDA kütüphanelerini torch/lib altında taşıyabilir
    for _dll in _glob.glob(os.path.join(_sp, "torch", "lib", "*.dll")):
        binaries.append((_dll, "."))
        _cuda_dll_count += 1
print(f"[spec] CUDA/torch DLL toplandı: {_cuda_dll_count} "
      f"({'GPU build' if _cuda_dll_count else 'CPU build — CUDA yok'})")

# --- Elle gereken gizli import'lar (grafik dışı) ---
hiddenimports += [
    "webrtcvad",
    "keyboard",
    "scipy.special.cython_special",
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
]
hiddenimports += collect_submodules("scipy")

# --- Uygulama veri dosyaları (kaynak ağaç yapısı korunur) ---
# UI ikonu resources.py tarafından src/ui/assets/ altından okunur.
datas += [("src/ui/assets", "src/ui/assets")]

# --- Boyut/çakışma için hariç tutulanlar ---
# NOT: İLK ÇALIŞAN BUILD için agresif olma. pandas/matplotlib pyannote tarafından
# import ediliyor olabilir; önce ÇALIŞSIN, sonra küçült. Aşağıdakiler kesin
# gereksizdir (yalnız test/benchmark/dev).
excludes = [
    "tests",
    "pytest",
    "_pytest",
    "IPython",
    "jupyter",
    "notebook",
    "ipykernel",
    "tkinter",
    "tcl",
    "tk",
]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir: binariler COLLECT'e bırakılır
    name="AudioProcess",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                       # UPX YOK: antivirüs yanlış-pozitifi + native DLL riski
    console=_console,                # AP_BUILD_CONSOLE=1 -> hata ayıklama konsolu
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="src/ui/assets/icon.png",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AudioProcess",
)
