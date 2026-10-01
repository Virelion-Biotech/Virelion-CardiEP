from __future__ import annotations

import numpy as np

from .conduction import RootSchedule
from .ecg import pseudo_ecg
from .geometry import EPGeometry
from .propagation import anisotropic_eikonal
from .repolarization import apd_map


def synthetic_tetra_geometry(*, with_fibres: bool = True) -> EPGeometry:
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    fibre = np.tile([1.0, 0.0, 0.0], (4, 1)) if with_fibres else None
    sheet = np.tile([0.0, 1.0, 0.0], (4, 1)) if with_fibres else None
    normal = np.tile([0.0, 0.0, 1.0], (4, 1)) if with_fibres else None
    electrodes = {
        "RA": np.array([-2.0, 0.0, 0.0]),
        "LA": np.array([2.0, 0.0, 0.0]),
        "LL": np.array([0.0, -2.0, 0.0]),
        "V1": np.array([0.2, 2.0, 0.0]),
        "V2": np.array([0.5, 2.0, 0.0]),
        "V3": np.array([0.8, 2.0, 0.0]),
        "V4": np.array([1.1, 2.0, 0.0]),
        "V5": np.array([1.4, 2.0, 0.0]),
        "V6": np.array([1.7, 2.0, 0.0]),
    }
    return EPGeometry(
        node_xyz_cm=nodes,
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=int),
        fibre=fibre,
        sheet=sheet,
        normal=normal,
        ventricular_coordinates={"tm": np.asarray([0.0, 0.2, 0.6, 1.0])},
        electrodes_cm=electrodes,
        root_nodes=(0,),
        endocardial_nodes=np.asarray([0, 1, 2]),
        metadata={"fixture": True},
    )


def run_reference_validation() -> dict:
    roots = RootSchedule(nodes=np.asarray([0]), activation_ms=np.asarray([0.0]), method="fixture")
    isotropic_geo = synthetic_tetra_geometry(with_fibres=False)
    isotropic = anisotropic_eikonal(
        isotropic_geo,
        roots,
        {"isotropic_speed": 0.1},
    )
    expected = np.asarray([0.0, 10.0, 10.0, 10.0])
    isotropic_error = float(np.max(np.abs(isotropic.activation_ms - expected)))

    anisotropic_geo = synthetic_tetra_geometry(with_fibres=True)
    anisotropic = anisotropic_eikonal(
        anisotropic_geo,
        roots,
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    directional_expected = np.asarray([0.0, 10.0, 20.0, 40.0])
    anisotropic_error = float(
        np.max(np.abs(anisotropic.activation_ms - directional_expected))
    )

    repolarization = apd_map(
        anisotropic_geo,
        anisotropic.activation_ms,
        {"apd_min_ms": 250.0, "apd_max_ms": 300.0, "apd_gradient_tm": 1.0},
    )
    ecg_a = pseudo_ecg(
        anisotropic_geo,
        anisotropic.activation_ms,
        repolarization.repolarization_ms,
        sample_rate_hz=250.0,
    )
    ecg_b = pseudo_ecg(
        anisotropic_geo,
        anisotropic.activation_ms,
        repolarization.repolarization_ms,
        sample_rate_hz=250.0,
    )
    deterministic_error = float(np.max(np.abs(ecg_a.values - ecg_b.values)))

    checks = {
        "isotropic_edge_travel_time": isotropic_error < 1e-10,
        "orthotropic_edge_travel_time": anisotropic_error < 1e-10,
        "repolarization_ordering": bool(
            np.all(repolarization.repolarization_ms > anisotropic.activation_ms)
        ),
        "standard_12_lead_shape": ecg_a.values.shape[0] == 12,
        "deterministic_pseudo_ecg": deterministic_error == 0.0,
    }
    return {
        "passed": all(checks.values()),
        "validation_status": "software_reference_checks",
        "checks": checks,
        "metrics": {
            "isotropic_max_abs_error_ms": isotropic_error,
            "orthotropic_max_abs_error_ms": anisotropic_error,
            "pseudo_ecg_repeat_max_abs_error": deterministic_error,
        },
        "scientific_boundary": (
            "These checks validate implementation invariants only; they do not establish "
            "physiological, clinical, or numerical equivalence to monodomain/bidomain solvers."
        ),
    }
