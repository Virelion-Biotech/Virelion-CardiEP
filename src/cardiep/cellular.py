from __future__ import annotations

import numpy as np


def membrane_voltage(
    activation_ms: np.ndarray,
    repolarization_ms: np.ndarray,
    time_ms: np.ndarray,
    *,
    resting_mv: float = -85.0,
    plateau_mv: float = 20.0,
    upstroke_tau_ms: float = 1.5,
    repolarization_tau_ms: float = 12.0,
) -> np.ndarray:
    """Generate a smooth phenomenological transmembrane-voltage field.

    This is an integration/reference waveform, not a validated ionic-cell model.
    Heavyweight backends should replace it with reaction models (e.g. TNNP/ToR-ORd)
    when ionic-current fidelity matters.
    """
    arrays = [np.asarray(value, dtype=float) for value in (activation_ms, repolarization_ms, time_ms)]
    if any(value.ndim != 1 or value.size == 0 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Membrane maps and time axis must be non-empty finite vectors")
    if arrays[0].shape != arrays[1].shape or np.any(arrays[1] < arrays[0]):
        raise ValueError("Repolarization must match and follow activation")
    if np.any(np.diff(arrays[2]) <= 0):
        raise ValueError("Membrane time axis must be strictly increasing")
    settings = (resting_mv, plateau_mv, upstroke_tau_ms, repolarization_tau_ms)
    if any(not np.isfinite(value) for value in settings):
        raise ValueError("Membrane settings must be finite")
    activation = arrays[0][:, None]
    repolarization = np.asarray(repolarization_ms, dtype=float)[:, None]
    time = np.asarray(time_ms, dtype=float)[None, :]
    if activation.shape != repolarization.shape:
        raise ValueError("activation and repolarization maps must have equal shape")
    if upstroke_tau_ms <= 0 or repolarization_tau_ms <= 0:
        raise ValueError("time constants must be positive")
    up = 1.0 / (1.0 + np.exp(np.clip(-(time - activation) / upstroke_tau_ms, -60, 60)))
    down = 1.0 / (1.0 + np.exp(np.clip((time - repolarization) / repolarization_tau_ms, -60, 60)))
    phase = up * down
    return resting_mv + (plateau_mv - resting_mv) * phase
