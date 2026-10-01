import numpy as np
import pytest

from cardiep.discrepancy import correlation_distance, normalized_rmse


def test_correlation_distance_is_shape_sensitive_but_scale_invariant() -> None:
    signal = np.sin(np.linspace(0, 2 * np.pi, 100))
    assert correlation_distance(signal, 3.0 * signal) == pytest.approx(0.0, abs=1e-12)
    assert correlation_distance(signal, -signal) == pytest.approx(2.0, abs=1e-12)


def test_normalized_rmse_zero_for_equal_waveforms() -> None:
    signal = np.linspace(-1.0, 1.0, 50)
    assert normalized_rmse(signal, signal) == pytest.approx(0.0)
