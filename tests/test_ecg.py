import numpy as np
import pytest

from cardiep import RootSchedule, anisotropic_eikonal, apd_map, pseudo_ecg, synthetic_tetra_geometry


def test_standard_electrodes_produce_12_leads() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    propagation = anisotropic_eikonal(
        geometry,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    repolarization = apd_map(
        geometry,
        propagation.activation_ms,
        {"apd_ms": 280.0},
    )
    ecg = pseudo_ecg(
        geometry,
        propagation.activation_ms,
        repolarization.repolarization_ms,
        sample_rate_hz=250.0,
    )
    assert ecg.lead_names == (
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    )
    assert ecg.values.shape[0] == 12
    assert np.isfinite(ecg.values).all()
    assert np.max(np.abs(ecg.values)) <= 1.0 + 1e-12



def test_pseudo_ecg_has_pre_activation_baseline_and_qrs_reference() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    propagation = anisotropic_eikonal(
        geometry,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    repolarization = apd_map(
        geometry,
        propagation.activation_ms,
        {"apd_ms": 280.0},
    )
    ecg = pseudo_ecg(
        geometry,
        propagation.activation_ms,
        repolarization.repolarization_ms,
        sample_rate_hz=500.0,
        pre_activation_ms=120.0,
    )
    assert ecg.time_ms[0] == pytest.approx(-120.0)
    assert ecg.reference_time_ms is not None
    assert np.min(propagation.activation_ms) - 20.0 <= ecg.reference_time_ms
    assert ecg.reference_time_ms <= np.max(propagation.activation_ms) + 20.0
    assert ecg.reference_method in {
        "max_multilead_rms_within_activation_window",
        "median_activation_fallback",
    }
    assert np.max(np.abs(ecg.values[:, 0])) < 1e-6


def test_pseudo_ecg_rejects_negative_pre_activation_window() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="pre_activation_ms"):
        pseudo_ecg(
            geometry,
            np.asarray([0.0, 10.0, 20.0, 40.0]),
            np.asarray([280.0, 290.0, 300.0, 320.0]),
            pre_activation_ms=-1.0,
        )
