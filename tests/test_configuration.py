import numpy as np
import pytest

from cardiep import EPGeometry, synthetic_tetra_geometry, validate_native_configuration


def test_unknown_native_parameter_is_rejected() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="Unknown numpy-eikonal-v1 parameter"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0]},
            {"fibbre_speed": 0.1},
        )


def test_us_fiber_speed_alias_is_explicitly_supported() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    validate_native_configuration(
        geometry,
        {"root_nodes": [0]},
        {"fiber_speed": 0.1},
    )


def test_conflicting_aliases_are_rejected() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="only one alias"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0]},
            {"fibre_speed": 0.1, "fiber_speed": 0.11},
        )


def test_unused_conduction_parameters_fail_closed() -> None:
    fibreless = synthetic_tetra_geometry(with_fibres=False)
    with pytest.raises(ValueError, match="no effect without a fibre field"):
        validate_native_configuration(
            fibreless,
            {"root_nodes": [0]},
            {"fibre_speed": 0.1},
        )

    fibred = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="isotropic_speed has no effect"):
        validate_native_configuration(
            fibred,
            {"root_nodes": [0]},
            {"isotropic_speed": 0.1},
        )


def test_scar_and_purkinje_parameters_require_their_inputs() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="without scar_labels"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0]},
            {"scar_speed_scale": 0.2},
        )
    with pytest.raises(ValueError, match="without purkinje_root_distance_cm"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0]},
            {"purkinje_speed": 0.3},
        )


def test_constant_apd_cannot_silently_override_gradient() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0]},
            {"apd_ms": 280.0, "apd_gradient_tm": 1.0},
        )


def test_unknown_setting_and_wrong_setting_types_are_rejected() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="Unknown numpy-eikonal-v1 setting"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0], "ecg_sample_rate": 250.0},
            {},
        )
    with pytest.raises(TypeError, match="with_ecg"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0], "with_ecg": "false"},
            {},
        )
    with pytest.raises(ValueError, match="auto_root_count"):
        validate_native_configuration(
            geometry,
            {"auto_root_count": -1},
            {},
        )


def test_transverse_speed_only_valid_for_fibre_only_geometry() -> None:
    full = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="only used"):
        validate_native_configuration(
            full,
            {"root_nodes": [0]},
            {"transverse_speed": 0.04},
        )

    fibre_only = EPGeometry(
        **{
            **full.__dict__,
            "sheet": None,
            "normal": None,
        }
    )
    validate_native_configuration(
        fibre_only,
        {"root_nodes": [0]},
        {"fibre_speed": 0.1, "transverse_speed": 0.04},
    )


def test_nonfinite_and_invalid_native_controls_are_rejected() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="positive and finite"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0], "ecg_sample_rate_hz": np.inf},
            {},
        )
    with pytest.raises(ValueError, match="positive integer"):
        validate_native_configuration(
            geometry,
            {"root_nodes": [0], "max_iterations": 0},
            {},
        )
