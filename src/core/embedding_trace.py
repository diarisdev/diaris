"""Konuşmacı karar izlerini diske yaz/oku — offline inceleme için.

Canlı arayüz izleri anlık gösterir ve atar. Offline değerlendirmede (AMI replay)
ise koşu gerçek zamandan hızlı akar: pencereyi canlı izlemek işe yaramaz, ama
koşuyu KAYDEDİP sonra istediğin hızda adım adım gezmek çok işe yarar — hatalı
bir atamanın üstünde durup skorlara, eşiğe ve margin'e bakabilirsin.

Format (.npz)
-------------
Embedding'ler JSON'a gömülürse dosya devasa olur (256 float × yüzlerce chunk).
Bunun yerine TÜM vektörler tek bir (M, D) float32 dizisinde havuzlanır; JSON
tarafında vektör yerine `{"__vec__": index}` durur.

    vectors : (M, D) float32   — havuzlanmış embedding'ler
    meta    : JSON metni       — kayıt yapısı, vektörler indeksle

Warm-up izi kayıt başına bir kez saklanır (her karede tekrarlanmaz); içindeki
n×n benzerlik matrisi düz liste olarak gider — bir kez yazılır, küçüktür.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TRACE_VERSION = 1
VECTOR_KEY = "__vec__"


class _VectorPool:
    """Vektörleri havuzlar ve JSON'a indeks olarak yazar."""

    def __init__(self, dim: int | None = None):
        self.vectors: list[np.ndarray] = []
        self.dim = dim

    def add(self, vector: np.ndarray) -> int:
        self.vectors.append(np.asarray(vector, dtype=np.float32).ravel())
        return len(self.vectors) - 1

    def stack(self) -> np.ndarray:
        if not self.vectors:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        return np.stack(self.vectors)


def _is_embedding(value, dim: int | None) -> bool:
    """1B ve beklenen boyutta mı? (2B matrisler havuza girmez, liste olur.)"""
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        return False
    return dim is None or value.shape[0] == dim


def _encode(value, pool: _VectorPool):
    if _is_embedding(value, pool.dim):
        return {VECTOR_KEY: pool.add(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _encode(v, pool) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, pool) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _decode(value, vectors: np.ndarray):
    if isinstance(value, dict):
        if VECTOR_KEY in value and len(value) == 1:
            return vectors[int(value[VECTOR_KEY])]
        return {k: _decode(v, vectors) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, vectors) for v in value]
    return value


def _infer_dim(trace) -> int | None:
    """İlk probe/centroid'den embedding boyutunu çıkarır."""
    for probe in (trace or {}).get("probes") or []:
        emb = probe.get("embedding")
        if isinstance(emb, np.ndarray) and emb.ndim == 1:
            return int(emb.shape[0])
    for record in ((trace or {}).get("speakers") or {}).values():
        centroid = record.get("centroid")
        if isinstance(centroid, np.ndarray) and centroid.ndim == 1:
            return int(centroid.shape[0])
    return None


class EmbeddingTraceWriter:
    """Kare kare iz biriktirip tek .npz dosyasına yazar."""

    def __init__(self, source: str = ""):
        self.source = source
        self.frames: list[dict] = []
        self.warmup = None
        self._dim: int | None = None

    def add(self, trace, warming_up: bool = False, time_sec: float | None = None) -> None:
        """Bir diarization karesini kaydeder. `trace` None ise kare atlanır."""
        if not trace:
            return
        if self._dim is None:
            self._dim = _infer_dim(trace)
        self.frames.append({
            "trace": trace,
            "warming_up": bool(warming_up),
            "time": None if time_sec is None else float(time_sec),
        })

    def set_warmup(self, warmup) -> None:
        """Kalibrasyon izini saklar (kayıt başına bir kez)."""
        if warmup:
            self.warmup = warmup

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pool = _VectorPool(self._dim)
        meta = {
            "version": TRACE_VERSION,
            "source": self.source,
            "dim": self._dim,
            "warmup": _encode(self.warmup, pool) if self.warmup else None,
            "frames": [_encode(frame, pool) for frame in self.frames],
        }
        np.savez_compressed(
            path,
            vectors=pool.stack(),
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )
        return path


def load_trace(path) -> dict:
    """Kaydedilmiş izi okur.

    Returns:
        {"source": str, "dim": int|None, "warmup": dict|None, "frames": [...]}
        Her kare arayüzün beklediği biçimdedir:
        {"trace": ..., "warming_up": bool, "time": float|None}
    """
    with np.load(Path(path), allow_pickle=False) as data:
        vectors = data["vectors"]
        meta = json.loads(str(data["meta"].item()))

    version = meta.get("version")
    if version != TRACE_VERSION:
        raise ValueError(f"Desteklenmeyen iz sürümü: {version} (beklenen {TRACE_VERSION})")

    return {
        "source": meta.get("source", ""),
        "dim": meta.get("dim"),
        "warmup": _decode(meta.get("warmup"), vectors) if meta.get("warmup") else None,
        "frames": [_decode(frame, vectors) for frame in meta.get("frames", [])],
    }
