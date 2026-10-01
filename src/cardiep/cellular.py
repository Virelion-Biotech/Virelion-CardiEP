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
    activation = np.asarray(activation_ms, dtype=float)[:, None]
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
