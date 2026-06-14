"""Diarization config plumbing (Windows yol + runtime YAML).

Pyannote, config'deki 'segmentation'/'embedding' alanlarını HuggingFace repo ID
olarak doğrular; Türkçe karakter (ü) ve boşluk içeren Windows yollarında patlar.
Çözüm: yerel model yollarını Windows 8.3 kısa-yol formatına çevirip ASCII-safe
bir temp config'e yazmak. Bu çekirdek diarization mantığı değil, ayrı bir endişe.
"""

import os
import tempfile

import yaml

from ..config import LOCAL_MODELS_DIR


def get_short_path(long_path):
    """
    Windows 8.3 kısa yol formatına çevir.
    Özel karakterler (ü, boşluk vb.) içeren yolları ASCII-safe yapar.
    """
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        result = ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 512)
        if result > 0:
            return buf.value
    except Exception:
        pass
    return long_path


def prepare_runtime_config(config_path):
    """
    Diarization config dosyasını okur, model yollarını runtime'da doğru mutlak
    yollarla (Windows 8.3 kısa-yol) günceller ve ASCII-safe bir temp dizine yazar.

    Returns:
        str: Yazılan geçici config dosyasının yolu.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    models_dir = LOCAL_MODELS_DIR
    params = config.get("pipeline", {}).get("params", {})
    # Model yollarını güncelle (Windows kısa yol — özel karakterlerden kaçın)
    seg_path = get_short_path(os.path.join(models_dir, "pyannote-segmentation"))
    emb_path = get_short_path(os.path.join(models_dir, "pyannote-embeddings"))
    if os.path.isdir(seg_path):
        params["segmentation"] = seg_path
        print(f"   Segmentation model: {seg_path}")
    if os.path.isdir(emb_path):
        params["embedding"] = emb_path
        print(f"   Embedding model: {emb_path}")
    # Geçici dosyaya yaz (ASCII-safe temp dizini)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(config, tmp_file, default_flow_style=False, allow_unicode=True)
    tmp_file.close()
    print(f"   Runtime config: {tmp_file.name}")
    return tmp_file.name
