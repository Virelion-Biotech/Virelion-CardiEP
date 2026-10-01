from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import EPGeometry

STANDARD_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class ECGResult:
    lead_names: tuple[str, ...]
    time_ms: np.ndarray
    values: np.ndarray
    sample_rate_hz: float
    model: str
    units: str = "a.u."

    def to_dict(self) -> dict:
        return {
            "lead_names": list(self.lead_names),
            "time_ms": self.time_ms.tolist(),
            "values": self.values.tolist(),
            "sample_rate_hz": self.sample_rate_hz,
            "model": self.model,
            "units": self.units,
        }


def _source_matrix(
    activation_ms: np.ndarray,
    repolarization_ms: np.ndarray,
    time_ms: np.ndarray,
    *,
    qrs_sigma_ms: float,
    t_sigma_ms: float,
    repolarization_scale: float,
) -> np.ndarray:
    t = time_ms[None, :]
    a = activation_ms[:, None]
    r = repolarization_ms[:, None]
    depol = np.exp(-0.5 * ((t - a) / qrs_sigma_ms) ** 2)
    repol = np.exp(-0.5 * ((t - r) / t_sigma_ms) ** 2)
    return depol - repolarization_scale * repol


def pseudo_ecg(
    geometry: EPGeometry,
    activation_ms: np.ndarray,
    repolarization_ms: np.ndarray,
    *,
    sample_rate_hz: float = 500.0,
    duration_ms: float | None = None,
    qrs_sigma_ms: float = 5.0,
    t_sigma_ms: float = 20.0,
    repolarization_scale: float = 0.55,
    chunk_size: int = 2048,
) -> ECGResult:
    if not geometry.electrodes_cm:
        raise ValueError(
            "Pseudo-ECG generation requires electrode coordinates in the EP geometry"
        )
    if sample_rate_hz <= 0 or qrs_sigma_ms <= 0 or t_sigma_ms <= 0:
        raise ValueError("ECG sampling rate and temporal widths must be positive")
    activation = np.asarray(activation_ms, dtype=float)
    repolarization = np.asarray(repolarization_ms, dtype=float)
    if activation.shape != (geometry.n_nodes,) or repolarization.shape != (geometry.n_nodes,):
        raise ValueError("Activation/repolarization maps must be node-wise")
    end_ms = float(duration_ms) if duration_ms is not None else float(np.max(repolarization) + 80.0)
    if end_ms <= 0:
        raise ValueError("ECG duration must be positive")
    dt_ms = 1000.0 / float(sample_rate_hz)
    time = np.arange(0.0, end_ms + 0.5 * dt_ms, dt_ms, dtype=float)

    names = list(geometry.electrodes_cm)
    electrode_xyz = np.stack([geometry.electrodes_cm[name] for name in names], axis=0)
    potentials = np.zeros((len(names), len(time)), dtype=float)
    center = geometry.node_xyz_cm.mean(axis=0)
    radial = geometry.node_xyz_cm - center
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(radial_norm, 1e-9)

    for start in range(0, geometry.n_nodes, max(1, int(chunk_size))):
        stop = min(geometry.n_nodes, start + max(1, int(chunk_size)))
        nodes = geometry.node_xyz_cm[start:stop]
        source = _source_matrix(
            activation[start:stop],
            repolarization[start:stop],
            time,
            qrs_sigma_ms=qrs_sigma_ms,
            t_sigma_ms=t_sigma_ms,
            repolarization_scale=repolarization_scale,
        )
        for ei, electrode in enumerate(electrode_xyz):
            displacement = electrode[None, :] - nodes
            distance = np.linalg.norm(displacement, axis=1)
            direction = displacement / np.maximum(distance[:, None], 1e-6)
            dipole_projection = np.sum(radial[start:stop] * direction, axis=1)
            weight = dipole_projection / np.maximum(distance**2, 1e-4)
            potentials[ei] += weight @ source

    by_name = {name: potentials[i] for i, name in enumerate(names)}
    standard_required = {"RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"}
    if standard_required <= set(by_name):
        ra, la, ll = by_name["RA"], by_name["LA"], by_name["LL"]
        wct = (ra + la + ll) / 3.0
        lead_values = [
            la - ra,
            ll - ra,
            ll - la,
            ra - 0.5 * (la + ll),
            la - 0.5 * (ra + ll),
            ll - 0.5 * (ra + la),
            *[by_name[f"V{i}"] - wct for i in range(1, 7)],
        ]
        lead_names = STANDARD_12
        values = np.stack(lead_values, axis=0)
    else:
        lead_names = tuple(names)
        values = potentials

    values = values - values[:, :1]
    max_abs = np.max(np.abs(values), axis=1, keepdims=True)
    values = values / np.maximum(max_abs, 1e-12)
    return ECGResult(
        lead_names=tuple(lead_names),
        time_ms=time,
        values=values,
        sample_rate_hz=float(sample_rate_hz),
        model="inverse-distance nodal dipole proxy",
    )
