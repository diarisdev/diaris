"""Dataset manager paketi — dataset başına indir/yükle, tek registry.

Kullanım:
    from tests.dataset_managers import get_dataset
    flores = get_dataset("flores200"); flores.download()
    pairs = flores.get_pairs("en", "tr")

Veri kökü: <proje kökü>/datasets/ (gitignore'lı). Hepsini indirmek için:
    python -m tests.dataset_managers
"""

from .base import DATASETS_ROOT, _download_with_progress, download_targz
from .librispeech import DatasetManager, _parse_transcription_file
from .flores200 import FloresDatasetManager, FLORES_LANG_MAP
from .ami import AmiDiarizationManager, AmiReferenceManager
from .chime6 import Chime6DatasetManager, parse_time_to_seconds, clean_chime6_text

_REGISTRY = {
    "librispeech": DatasetManager,
    "flores200": FloresDatasetManager,
    "ami": AmiDiarizationManager,          # küçük DER alt kümesi (2 toplantı)
    "ami_refs": AmiReferenceManager,       # replay/score için 16-toplantı test referansları (lhotse)
    "chime6": Chime6DatasetManager,
}


def get_dataset(name, **kwargs):
    """İsimle bir dataset manager örneği döndürür."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Bilinmeyen dataset '{name}'. Mevcut: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_datasets():
    return sorted(_REGISTRY.keys())


__all__ = [
    "DATASETS_ROOT", "_download_with_progress", "download_targz",
    "DatasetManager", "_parse_transcription_file",
    "FloresDatasetManager", "FLORES_LANG_MAP",
    "AmiDiarizationManager", "AmiReferenceManager", "Chime6DatasetManager",
    "parse_time_to_seconds", "clean_chime6_text",
    "get_dataset", "available_datasets",
]
