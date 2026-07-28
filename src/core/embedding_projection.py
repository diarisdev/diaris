"""Embedding uzayını 2B'ye yansıtma — görselleştirme için.

Konuşmacı embedding'leri yüksek boyutlu (pyannote/wespeaker: 256B) ve karar
KOSİNÜS BENZERLİĞİ üzerinden veriliyor. 2B bir dağılım grafiği bu kararın
yalnızca YAKLAŞIK bir resmidir: grafikte üst üste düşen iki nokta tracker için
uzak olabilir. Bu yüzden buradaki her yansıtma, konumun ne kadar güvenilir
olduğunu söyleyen bir "düzlem-içi oran" da döner — arayüz bunu saydamlık/boyut
olarak kodlayıp grafiğin kendi hatasını itiraf etmesini sağlar.

Kesin karar için grafiğe değil, SpeakerTracker'ın skor/eşik sayılarına bakılmalı.

Tasarım
-------
* DONMUŞ BAZ: baz bir kez kurulur, sonraki noktalar hep aynı baza yansıtılır.
  Her noktada PCA'yı yeniden hesaplamak grafiği sürekli zıplatır ve hiçbir şey
  takip edilemez hale gelir. Baz yalnızca konuşmacı kümesi değiştiğinde (ya da
  kullanıcı isteyince) yeniden kurulur.
* CENTROID-PCA: baz, tüm noktalar yerine KONUŞMACI CENTROID'lerinin ilk iki ana
  bileşenidir. Elde edilen düzlem, konuşmacıları birbirinden en çok ayıran
  düzlemdir — yani "bu chunk hangi konuşmacıya yakın" sorusunun düzlemi.
  Centroid sayısı yetmezse (K < 3) nokta bulutuna düşülür.

Saf numpy — torch/Qt/model yok, birim testleri modelsiz koşar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Centroid-PCA anlamlı bir düzlem verebilmesi için gereken min konuşmacı sayısı.
# K centroid en fazla K-1 boyut yayar; 2 centroid tek eksen verir.
MIN_CENTROIDS_FOR_BASIS = 3
# Nokta bulutundan baz kurmak için gereken min örnek.
MIN_POINTS_FOR_BASIS = 3


def _as_matrix(vectors) -> np.ndarray:
    """(N, D) float64 matris — liste/tensor/ndarray karışık gelebilir."""
    rows = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if not rows:
        return np.zeros((0, 0))
    return np.stack(rows)


def _orthonormal_complete(basis: np.ndarray, dim: int, want: int) -> np.ndarray:
    """Bazı `want` satıra tamamlar (rank düşükse eksik eksenleri doldurur)."""
    rows = [row for row in basis]
    seed = 0
    while len(rows) < want:
        candidate = np.zeros(dim, dtype=np.float64)
        candidate[seed % dim] = 1.0
        seed += 1
        for row in rows:
            candidate = candidate - np.dot(candidate, row) * row
        norm = np.linalg.norm(candidate)
        if norm < 1e-9:
            if seed > dim * 2:  # tükendi — sıfır eksenle devam et
                rows.append(np.zeros(dim, dtype=np.float64))
            continue
        rows.append(candidate / norm)
    return np.stack(rows[:want])


def _pca_basis(matrix: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(mean, basis) — basis satırları ortonormal, en büyük varyans önce."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # full_matrices=False: Vt (min(N,D), D) — bize sadece ilk satırlar lazım.
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    keep = [vt[i] for i in range(len(singular)) if singular[i] > 1e-9][:components]
    basis = np.stack(keep) if keep else np.zeros((0, matrix.shape[1]))
    if basis.shape[0] < components:
        basis = _orthonormal_complete(basis, matrix.shape[1], components)
    return mean, basis


@dataclass(frozen=True)
class Projection:
    """Donmuş 2B yansıtma bazı."""

    mean: np.ndarray            # (D,)
    basis: np.ndarray           # (2, D) — ortonormal satırlar
    source: str                 # "centroids" | "points"
    speaker_count: int          # baz kurulurken bilinen konuşmacı sayısı

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def project(self, vectors) -> tuple[np.ndarray, np.ndarray]:
        """Vektörleri 2B'ye yansıtır.

        Returns:
            (coords, in_plane):
                coords:   (N, 2) düzlem koordinatları
                in_plane: (N,) [0, 1] — merkezlenmiş vektörün ne kadarı düzlemde
                          kaldı. 1.0 = konum vektörü tam temsil ediyor,
                          0'a yakın = bu noktanın 2B konumuna güvenme.
        """
        matrix = _as_matrix(vectors)
        if matrix.size == 0:
            return np.zeros((0, 2)), np.zeros(0)
        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"Embedding boyutu bazla uyuşmuyor: {matrix.shape[1]} != {self.dim}"
            )

        centered = matrix - self.mean
        coords = centered @ self.basis.T
        full = np.linalg.norm(centered, axis=1)
        planar = np.linalg.norm(coords, axis=1)
        # Sıfır uzunluklu (merkezle çakışan) vektör: konum belirsiz ama hatalı
        # değil — 1.0 verip görsel olarak cezalandırmıyoruz.
        in_plane = np.where(full > 1e-9, planar / np.maximum(full, 1e-9), 1.0)
        return coords, np.clip(in_plane, 0.0, 1.0)


def fit_projection(centroids=None, points=None) -> Projection | None:
    """Yansıtma bazını kurar.

    Öncelik centroid'lerdedir: konuşmacıları ayıran düzlem, nokta bulutunun en
    çok varyans taşıyan düzleminden daha bilgilendiricidir (varyans çoğu zaman
    konuşmacı farkı değil, kayıt/kanal gürültüsüdür).

    Args:
        centroids: konuşmacı centroid vektörleri (iterable) — tercih edilen kaynak.
        points: yedek nokta bulutu (rezervuar embedding'leri vb.).

    Returns:
        Projection, ya da baz kuracak kadar veri yoksa None.
    """
    centroid_matrix = _as_matrix(centroids or [])
    speaker_count = int(centroid_matrix.shape[0])

    if speaker_count >= MIN_CENTROIDS_FOR_BASIS:
        mean, basis = _pca_basis(centroid_matrix)
        return Projection(mean=mean, basis=basis, source="centroids",
                          speaker_count=speaker_count)

    point_matrix = _as_matrix(points or [])
    if point_matrix.shape[0] >= MIN_POINTS_FOR_BASIS:
        mean, basis = _pca_basis(point_matrix)
        return Projection(mean=mean, basis=basis, source="points",
                          speaker_count=speaker_count)

    # Tek konuşmacı / birkaç nokta: ayırt edici bir düzlem yok. Uydurmak yerine
    # None dönüyoruz — arayüz "henüz yeterli veri yok" gösterir.
    return None


def needs_refit(projection: Projection | None, speaker_count: int) -> bool:
    """Konuşmacı kümesi değiştiyse baz eskimiştir.

    Bazı KENDİLİĞİNDEN yenilemek grafiği zıplatır; bu yüzden karar arayüze
    bırakılır (kullanıcıya "yeniden hesapla" önerilir).
    """
    if projection is None:
        return speaker_count >= MIN_CENTROIDS_FOR_BASIS
    return speaker_count != projection.speaker_count


def similarity_matrix(vectors) -> np.ndarray:
    """Kosinüs benzerlik matrisi (N, N) — konuşmacılar arası yakınlık haritası.

    Bu matris KESİN: tracker'ın kullandığı benzerliğin ta kendisi (yansıtma
    yaklaşıklığı yok). Birbirine merge eşiğine yaklaşan konuşmacı çiftlerini
    görünür kılar.
    """
    matrix = _as_matrix(vectors)
    if matrix.size == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-9)
    return np.clip(unit @ unit.T, -1.0, 1.0)
