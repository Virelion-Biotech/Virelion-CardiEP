import numpy as np
import pytest

from cardiep import RootSchedule, anisotropic_eikonal, synthetic_tetra_geometry


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
