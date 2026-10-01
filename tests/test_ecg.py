import numpy as np

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
