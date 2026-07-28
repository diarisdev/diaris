"""embedding_projection birim testleri.

Saf numpy (torch/Qt/model yok) — CI'da tam hızda koşar.
"""

import numpy as np

from src.core.embedding_projection import (
    MIN_CENTROIDS_FOR_BASIS,
    Projection,
    fit_projection,
    needs_refit,
    similarity_matrix,
)


def unit(*components) -> np.ndarray:
    vector = np.array(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


# Üç eksende ayrışan üç konuşmacı — 4 boyutlu uzayda, 4. boyut kullanılmıyor.
CENTROIDS = [unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)]


# --------------------------------------------------------------------------- #
# Baz kurulumu
# --------------------------------------------------------------------------- #
def test_no_projection_when_data_is_insufficient():
    assert fit_projection() is None
    assert fit_projection(centroids=[unit(1, 0, 0, 0)]) is None
    assert fit_projection(centroids=[unit(1, 0, 0, 0), unit(0, 1, 0, 0)]) is None


def test_centroids_are_preferred_over_points():
    projection = fit_projection(centroids=CENTROIDS, points=[unit(0, 0, 0, 1)] * 10)
    assert projection is not None
    assert projection.source == "centroids"
    assert projection.speaker_count == 3


def test_falls_back_to_points_when_centroids_are_too_few():
    projection = fit_projection(
        centroids=[unit(1, 0, 0, 0)],
        points=[unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)],
    )
    assert projection is not None
    assert projection.source == "points"
    assert projection.speaker_count == 1


def test_basis_is_orthonormal():
    projection = fit_projection(centroids=CENTROIDS)
    gram = projection.basis @ projection.basis.T
    assert np.allclose(gram, np.eye(2), atol=1e-9)


def test_basis_is_completed_when_data_is_rank_deficient():
    """Aynı doğrultuda yayılan centroid'ler tek eksen verir; baz yine 2 satır olmalı."""
    collinear = [unit(1, 0, 0, 0), unit(2, 0, 0, 0) * 1.0, unit(3, 0, 0, 0)]
    projection = fit_projection(centroids=collinear)
    assert projection.basis.shape == (2, 4)
    gram = projection.basis @ projection.basis.T
    assert np.allclose(gram, np.eye(2), atol=1e-9)


# --------------------------------------------------------------------------- #
# Yansıtma
# --------------------------------------------------------------------------- #
def test_projection_is_frozen_across_calls():
    """Aynı vektör, araya başka noktalar girse de aynı koordinata gitmeli."""
    projection = fit_projection(centroids=CENTROIDS)
    probe = unit(0.9, 0.1, 0, 0)

    first, _ = projection.project([probe])
    projection.project([unit(0, 0, 0, 1)] * 20)  # araya gürültü
    second, _ = projection.project([probe])

    assert np.allclose(first, second)


def test_separated_speakers_land_apart():
    projection = fit_projection(centroids=CENTROIDS)
    coords, _ = projection.project(CENTROIDS)
    distances = [
        np.linalg.norm(coords[i] - coords[j])
        for i in range(3) for j in range(i + 1, 3)
    ]
    assert min(distances) > 0.5


def test_in_plane_ratio_flags_out_of_plane_points():
    """Bazın kapsamadığı yönde uzanan nokta düşük güvenle işaretlenmeli."""
    projection = fit_projection(centroids=CENTROIDS)

    _, in_plane_centroid = projection.project([CENTROIDS[0]])
    # 4. boyut centroid düzleminde temsil edilmiyor.
    _, in_plane_outside = projection.project([unit(0, 0, 0, 1)])

    assert in_plane_centroid[0] > 0.9
    assert in_plane_outside[0] < in_plane_centroid[0]
    assert 0.0 <= in_plane_outside[0] <= 1.0


def test_projection_rejects_wrong_dimension():
    projection = fit_projection(centroids=CENTROIDS)
    try:
        projection.project([np.zeros(7)])
    except ValueError as exc:
        assert "boyut" in str(exc).lower()
    else:
        raise AssertionError("Boyut uyuşmazlığı hata vermeliydi")


def test_empty_input_projects_to_empty_output():
    projection = fit_projection(centroids=CENTROIDS)
    coords, in_plane = projection.project([])
    assert coords.shape == (0, 2)
    assert in_plane.shape == (0,)


# --------------------------------------------------------------------------- #
# Yenileme kararı
# --------------------------------------------------------------------------- #
def test_refit_is_needed_only_when_the_speaker_set_changes():
    projection = fit_projection(centroids=CENTROIDS)
    assert needs_refit(projection, 3) is False
    assert needs_refit(projection, 4) is True


def test_refit_is_needed_once_enough_speakers_exist():
    assert needs_refit(None, MIN_CENTROIDS_FOR_BASIS - 1) is False
    assert needs_refit(None, MIN_CENTROIDS_FOR_BASIS) is True


# --------------------------------------------------------------------------- #
# Benzerlik matrisi (kesin — yansıtma yaklaşıklığı yok)
# --------------------------------------------------------------------------- #
def test_similarity_matrix_is_exact_cosine():
    matrix = similarity_matrix([unit(1, 0, 0, 0), unit(1, 1, 0, 0), unit(0, 1, 0, 0)])
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert np.isclose(matrix[0, 1], 1 / np.sqrt(2))
    assert np.isclose(matrix[0, 2], 0.0, atol=1e-9)


def test_similarity_matrix_handles_unnormalised_input():
    """Girdi L2-normalize değilse de kosinüs doğru çıkmalı."""
    matrix = similarity_matrix([np.array([3.0, 0.0]), np.array([7.0, 0.0])])
    assert np.isclose(matrix[0, 1], 1.0)


def test_similarity_matrix_of_nothing_is_empty():
    assert similarity_matrix([]).shape == (0, 0)


def test_projection_dataclass_reports_dimension():
    projection = Projection(mean=np.zeros(5), basis=np.zeros((2, 5)),
                            source="points", speaker_count=0)
    assert projection.dim == 5
