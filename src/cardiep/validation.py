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
        "pre_activation_baseline": bool(ecg_a.time_ms[0] < np.min(anisotropic.activation_ms)),
        "qrs_reference_present": bool(
            ecg_a.reference_time_ms is not None
            and np.isfinite(ecg_a.reference_time_ms)
        ),
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



def _structured_cube_geometry(n: int) -> EPGeometry:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Structured refinement level must be a positive integer")
    index: dict[tuple[int, int, int], int] = {}
    points: list[list[float]] = []
    counter = 0
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                index[(i, j, k)] = counter
                points.append([i / n, j / n, k / n])
                counter += 1

    tetrahedra: list[list[int]] = []
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
        metadata={"fixture": "structured-unit-cube", "refinement_n": n},
    )


def run_eikonal_refinement_validation(
    *,
    levels: tuple[int, ...] = (4, 8),
) -> dict:
    if len(levels) < 2:
        raise ValueError("At least two refinement levels are required")
    if any(level % 4 != 0 for level in levels):
        raise ValueError(
            "Default off-axis target requires refinement levels divisible by 4"
        )
    if any(later <= earlier for earlier, later in zip(levels, levels[1:], strict=True)):
        raise ValueError("Refinement levels must be strictly increasing")

    target = np.asarray([1.0, 0.5, 0.25], dtype=float)
    exact_ms = float(np.linalg.norm(target))
    rows = []
    errors = []
    roots = RootSchedule(
        nodes=np.asarray([0]),
        activation_ms=np.asarray([0.0]),
        method="analytic-refinement",
    )
    for level in levels:
        geometry = _structured_cube_geometry(level)
        index = int(
            np.argmin(
                np.linalg.norm(
                    geometry.node_xyz_cm - target[None, :],
                    axis=1,
                )
            )
        )
        if not np.allclose(geometry.node_xyz_cm[index], target):
            raise RuntimeError("Refinement mesh did not contain the analytic target")
        result = anisotropic_eikonal(
            geometry,
            roots,
            {"isotropic_speed": 1.0},
        )
        observed_ms = float(result.activation_ms[index])
        error_ms = abs(observed_ms - exact_ms)
        errors.append(error_ms)
        rows.append(
            {
                "n": int(level),
                "h_cm": 1.0 / level,
                "n_nodes": int(geometry.n_nodes),
                "n_tetrahedra": int(len(geometry.tetrahedra)),
                "activation_ms": observed_ms,
                "exact_ms": exact_ms,
                "absolute_error_ms": error_ms,
            }
        )

    monotone = all(
        later < earlier
        for earlier, later in zip(errors, errors[1:], strict=True)
    )
    reduction_ratio = errors[-1] / max(errors[0], 1e-15)
    passed = bool(
        monotone
        and reduction_ratio < 0.8
        and errors[-1] < 0.04
    )
    return {
        "passed": passed,
        "validation_status": "numerical_refinement_check",
        "benchmark": "isotropic-unit-cube-off-axis",
        "target_cm": target.tolist(),
        "isotropic_speed_cm_per_ms": 1.0,
        "levels": rows,
        "metrics": {
            "monotone_error_reduction": monotone,
            "fine_over_coarse_error_ratio": reduction_ratio,
            "fine_absolute_error_ms": errors[-1],
        },
        "scientific_boundary": (
            "This manufactured isotropic refinement study verifies that the native "
            "tetrahedral Eikonal discretization reduces off-axis numerical error. It "
            "does not establish equivalence to monodomain/bidomain electrophysiology."
        ),
    }
