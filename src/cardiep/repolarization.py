from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import EPGeometry


@dataclass(frozen=True)
class RepolarizationResult:
    apd_ms: np.ndarray
    repolarization_ms: np.ndarray
    method: str


def apd_map(
    geometry: EPGeometry,
    activation_ms: np.ndarray,
    parameters: dict[str, float],
) -> RepolarizationResult:
    n = geometry.n_nodes
    if "apd_ms" in parameters:
        value = float(parameters["apd_ms"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("apd_ms must be positive")
        apd = np.full(n, value, dtype=float)
        method = "constant"
    else:
        apd_min = float(parameters.get("apd_min_ms", parameters.get("apd_min", 240.0)))
        apd_max = float(parameters.get("apd_max_ms", parameters.get("apd_max", 320.0)))
        if not (np.isfinite(apd_min) and np.isfinite(apd_max) and 0 < apd_min <= apd_max):
            raise ValueError("Expected 0 < apd_min_ms <= apd_max_ms")
        gradient = np.zeros(n, dtype=float)
        used = []
        for name, values in geometry.ventricular_coordinates.items():
            key = f"apd_gradient_{name}"
            if key in parameters:
                gradient += np.asarray(values, dtype=float) * float(parameters[key])
                used.append(name)
        if used:
            low, high = float(np.min(gradient)), float(np.max(gradient))
            normalized = np.zeros(n, dtype=float) if high <= low else (gradient - low) / (high - low)
            apd = apd_min + normalized * (apd_max - apd_min)
            method = "ventricular-coordinate-gradient:" + ",".join(sorted(used))
        else:
            apd = np.full(n, 0.5 * (apd_min + apd_max), dtype=float)
            method = "midpoint-without-coordinate-gradient"
    activation = np.asarray(activation_ms, dtype=float)
    if activation.shape != (n,):
        raise ValueError("activation_ms shape must match the EP mesh")
    return RepolarizationResult(
        apd_ms=apd,
        repolarization_ms=activation + apd,
        method=method,
    )
