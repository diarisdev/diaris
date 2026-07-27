"""Diaris .exe paketleme yardımcısı (PyInstaller onedir).

Bu script PyInstaller'ı çağırır ve build SONRASI kurulumu tamamlar: modelleri
ve .env'i exe'nin yanına yerleştirir (donmuş modda config bunları orada arar).

Ön koşullar (HEDEF varyantın venv'inde):
    pip install pyinstaller
    # CPU varyantı için CPU torch, GPU varyantı için CUDA torch kurulu olmalı.
    python scripts/download_models.py     # models/ dolu olmalı

Çalıştırma (proje kökünden):
    python scripts/build_app.py                 # yalın GUI build
    python scripts/build_app.py --debug         # konsollu (hata ayıklama) build
    python scripts/build_app.py --with-models    # modelleri de dist'e kopyala
    python scripts/build_app.py --clean          # önce build/ ve dist/ temizle

İLK BUILD İPUCU: `--debug` ile başlayın. GUI açılışta eksik bir import'tan
çökerse konsol traceback'i gösterir; hidden import iterasyonunun tek yolu budur.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_APP = ROOT / "dist" / "Diaris"


def _wipe_build_dirs() -> None:
    """build/ ve dist/ dizinlerini tamamen siler.

    PyInstaller'ın kendi `--clean`'i yalnızca cache + ara dosyaları temizler;
    dist/ İÇİNDEKİ eski çıktıya DOKUNMAZ ve `--noconfirm` yalnızca bu build'in
    ürettiği klasörü üzerine yazar. Sonuç: adı değişen (Audio-process -> Diaris)
    ya da artık üretilmeyen eski build'ler dist/ altında kalıcı olarak durur ve
    GB'larca yer kaplar. Gerçekten temiz bir çıktı için dizinleri burada siliyoruz.
    """
    for name in ("build", "dist"):
        path = ROOT / name
        if path.is_dir():
            print(f"   siliniyor: {path}")
            shutil.rmtree(path, ignore_errors=True)


def _run_pyinstaller(debug: bool, clean: bool) -> int:
    env = dict(os.environ)
    env["AP_BUILD_CONSOLE"] = "1" if debug else "0"
    cmd = [sys.executable, "-m", "PyInstaller", "Diaris.spec", "--noconfirm"]
    if clean:
        cmd.append("--clean")
    print(f">>> {'DEBUG (konsollu)' if debug else 'GUI'} build başlıyor...")
    print("    " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), env=env).returncode


def _place_runtime_assets(with_models: bool) -> None:
    """Modelleri ve .env'i exe'nin yanına yerleştirir."""
    if not DIST_APP.is_dir():
        print(f"!! Beklenen çıktı yok: {DIST_APP}")
        return

    # .env: varsa kopyala, yoksa .env.example'ı şablon olarak koy.
    dst_env = DIST_APP / ".env"
    if not dst_env.exists():
        src_env = ROOT / ".env"
        example = ROOT / ".env.example"
        if src_env.exists():
            shutil.copy2(src_env, dst_env)
            print(f"   .env kopyalandı -> {dst_env}")
            print("   UYARI: .env HF_TOKEN içerebilir; dağıtmadan önce temizleyin.")
        elif example.exists():
            shutil.copy2(example, dst_env)
            print(f"   .env.example -> {dst_env} (şablon; kullanıcı dolduracak)")

    # models/: opsiyonel — büyük olduğundan varsayılan olarak KOPYALANMAZ.
    dst_models = DIST_APP / "models"
    if with_models:
        src_models = ROOT / "models"
        if src_models.is_dir() and not dst_models.exists():
            print(f"   models/ kopyalanıyor (büyük olabilir)... -> {dst_models}")
            shutil.copytree(src_models, dst_models)
            print("   models/ kopyalandı.")
    elif not dst_models.exists():
        (DIST_APP / "models").mkdir(exist_ok=True)
        print("   Boş models/ oluşturuldu. Modelleri buraya koyun ya da ilk "
              "açılışta download_models ile indirin.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diaris .exe paketle.")
    ap.add_argument("--debug", action="store_true",
                    help="Konsollu build (açılış hatalarını görmek için).")
    ap.add_argument("--with-models", action="store_true",
                    help="models/ dizinini de dist'e kopyala (offline installer).")
    ap.add_argument("--clean", action="store_true",
                    help="Build öncesi build/ ve dist/ dizinlerini TAMAMEN sil "
                         "(eski/adı değişmiş build artıkları kalmasın).")
    args = ap.parse_args()

    if not (ROOT / "Diaris.spec").exists():
        raise SystemExit("Diaris.spec bulunamadı — proje kökünden çalıştırın.")

    if args.clean:
        print(">>> Temizlik: build/ ve dist/ siliniyor...")
        _wipe_build_dirs()

    rc = _run_pyinstaller(args.debug, args.clean)
    if rc != 0:
        raise SystemExit(f"PyInstaller başarısız (exit {rc}).")

    _place_runtime_assets(args.with_models)

    print("\n" + "=" * 68)
    print(f"Build tamam: {DIST_APP}")
    print("Çalıştırmak için:")
    print(f'   "{DIST_APP / "Diaris.exe"}"')
    print("\nÖnce şunları doğrulayın:")
    print("  1. models/ dolu (whisper, pyannote, nllb, silero_vad.onnx)")
    print("  2. .env mevcut (ve dağıtım için HF_TOKEN temiz)")
    print("  3. İlk deneme --debug ile: açılış hatası varsa konsolda görünür")
    print("=" * 68)


if __name__ == "__main__":
    main()
