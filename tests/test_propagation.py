import numpy as np
import pytest

from cardiep import EPGeometry, RootSchedule, anisotropic_eikonal, synthetic_tetra_geometry


def test_orthotropic_eikonal_matches_axis_travel_times() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    result = anisotropic_eikonal(
        geometry,
        RootSchedule(
            nodes=np.asarray([0]),
            activation_ms=np.asarray([0.0]),
            method="test",
        ),
        {
            "fibre_speed": 0.1,
            "sheet_speed": 0.05,
            "normal_speed": 0.025,
        },
    )
    assert result.activation_ms == pytest.approx([0.0, 10.0, 20.0, 40.0])


def test_scar_modifier_slows_a_scar_node() -> None:
    base = synthetic_tetra_geometry(with_fibres=False)
    healthy = anisotropic_eikonal(
        base,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"isotropic_speed": 0.1},
    )
    scarred = type(base)(
        **{**base.__dict__, "scar_labels": np.asarray([0, 2, 0, 0], dtype=int)}
    )
    slowed = anisotropic_eikonal(
        scarred,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"isotropic_speed": 0.1, "scar_speed_scale": 0.1},
    )
    assert slowed.activation_ms[1] > healthy.activation_ms[1]


def test_fibreless_geometry_reports_isotropic_fallback() -> None:
    result = anisotropic_eikonal(
        synthetic_tetra_geometry(with_fibres=False),
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"isotropic_speed": 0.1},
    )
    assert any("isotropic" in warning for warning in result.warnings)



def _structured_cube(n: int) -> EPGeometry:
    index = {}
    points = []
    counter = 0
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                index[(i, j, k)] = counter
                points.append([i / n, j / n, k / n])
                counter += 1

    tetrahedra = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v000 = index[(i, j, k)]
                v100 = index[(i + 1, j, k)]
                v010 = index[(i, j + 1, k)]
                v001 = index[(i, j, k + 1)]
                v110 = index[(i + 1, j + 1, k)]
                v101 = index[(i + 1, j, k + 1)]
                v011 = index[(i, j + 1, k + 1)]
                v111 = index[(i + 1, j + 1, k + 1)]
                tetrahedra.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v100, v101, v111],
                        [v000, v010, v110, v111],
                        [v000, v010, v011, v111],
                        [v000, v001, v101, v111],
                        [v000, v001, v011, v111],
                    ]
                )

    return EPGeometry(
        node_xyz_cm=np.asarray(points, dtype=float),
        tetrahedra=np.asarray(tetrahedra, dtype=int),
        root_nodes=(0,),
    )


def _activation_at(geometry: EPGeometry, xyz: np.ndarray) -> float:
    index = int(
        np.argmin(np.linalg.norm(geometry.node_xyz_cm - xyz[None, :], axis=1))
    )
    assert geometry.node_xyz_cm[index] == pytest.approx(xyz)
    result = anisotropic_eikonal(
        geometry,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "analytic-refinement"),
        {"isotropic_speed": 1.0},
    )
    return float(result.activation_ms[index])


def test_tetrahedral_update_reduces_off_axis_error_under_mesh_refinement() -> None:
    target = np.asarray([1.0, 0.5, 0.25])
    exact_ms = float(np.linalg.norm(target))

    coarse_error = abs(_activation_at(_structured_cube(4), target) - exact_ms)
    fine_error = abs(_activation_at(_structured_cube(8), target) - exact_ms)

    assert coarse_error > 0.0
    assert fine_error < 0.8 * coarse_error
    assert fine_error < 0.04
