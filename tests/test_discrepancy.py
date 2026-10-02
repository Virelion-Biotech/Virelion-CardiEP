import numpy as np
import pytest

from cardiep.discrepancy import correlation_distance, ecg_discrepancy, normalized_rmse
from cardiep.ecg import ECGResult


def test_correlation_distance_is_shape_sensitive_but_scale_invariant() -> None:
    signal = np.sin(np.linspace(0, 2 * np.pi, 100))
    assert correlation_distance(signal, 3.0 * signal) == pytest.approx(0.0, abs=1e-12)
    assert correlation_distance(signal, -signal) == pytest.approx(2.0, abs=1e-12)


def test_normalized_rmse_zero_for_equal_waveforms() -> None:
    signal = np.linspace(-1.0, 1.0, 50)
    assert normalized_rmse(signal, signal) == pytest.approx(0.0)



def test_r_relative_ecg_alignment_is_invariant_to_simulation_clock_offset() -> None:
    raw = {
        "lead_names": ["I"],
        "relative_time_s": [-0.02, -0.01, 0.0, 0.01, 0.02],
        "beat_template": {"I": [0.0, 0.25, 1.0, 0.25, 0.0]},
    }
    simulated = ECGResult(
        lead_names=("I",),
        time_ms=np.asarray([80.0, 90.0, 100.0, 110.0, 120.0]),
        values=np.asarray([[0.0, 0.25, 1.0, 0.25, 0.0]]),
        sample_rate_hz=100.0,
        model="test",
        reference_time_ms=100.0,
        reference_method="test-reference",
    )
    assert ecg_discrepancy(raw, simulated, "correlation") == pytest.approx(0.0, abs=1e-12)


def test_r_relative_ecg_requires_simulated_reference_time() -> None:
    raw = {
        "lead_names": ["I"],
        "relative_time_s": [-0.01, 0.0, 0.01],
        "beat_template": {"I": [0.0, 1.0, 0.0]},
    }
    simulated = ECGResult(
        lead_names=("I",),
        time_ms=np.asarray([0.0, 10.0, 20.0]),
        values=np.asarray([[0.0, 1.0, 0.0]]),
        sample_rate_hz=100.0,
        model="test",
    )
    with pytest.raises(ValueError, match="reference_time_ms"):
        ecg_discrepancy(raw, simulated, "correlation")


def test_ecg_time_axis_must_match_template_and_be_monotonic() -> None:
    simulated = ECGResult(
        lead_names=("I",),
        time_ms=np.asarray([-10.0, 0.0, 10.0]),
        values=np.asarray([[0.0, 1.0, 0.0]]),
        sample_rate_hz=100.0,
        model="test",
        reference_time_ms=0.0,
    )
    bad_length = {
        "lead_names": ["I"],
        "relative_time_s": [-0.01, 0.0],
        "beat_template": {"I": [0.0, 1.0, 0.0]},
    }
    with pytest.raises(ValueError, match="length"):
        ecg_discrepancy(bad_length, simulated, "correlation")

    nonmonotonic = {
        "lead_names": ["I"],
        "relative_time_s": [-0.01, 0.01, 0.0],
        "beat_template": {"I": [0.0, 1.0, 0.0]},
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        ecg_discrepancy(nonmonotonic, simulated, "correlation")



def test_ecg_quality_scores_weight_morphology_discrepancy() -> None:
    signal = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0])
    raw = {
        "lead_names": ["I", "II"],
        "beat_template": {
            "I": signal.tolist(),
            "II": signal.tolist(),
        },
        "quality": {
            "lead_quality": {
                "I": {"quality_score": 1.0},
                "II": {"quality_score": 0.0},
            }
        },
    }
    simulated = ECGResult(
        lead_names=("I", "II"),
        time_ms=np.arange(5, dtype=float),
        values=np.stack([signal, -signal]),
        sample_rate_hz=1000.0,
        model="test",
    )
    weighted = ecg_discrepancy(raw, simulated, "correlation")
    equal = ecg_discrepancy(
        raw,
        simulated,
        "correlation",
        lead_weighting="equal",
    )
    assert weighted == pytest.approx(0.0, abs=1e-12)
    assert equal == pytest.approx(1.0, abs=1e-12)


def test_ecg_noise_weighting_falls_back_when_quality_scores_absent() -> None:
    signal = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0])
    raw = {
        "lead_names": ["I", "II"],
        "beat_template": {
            "I": signal.tolist(),
            "II": signal.tolist(),
        },
        "uncertainty": {
            "noise_sigma_by_lead": {
                "I": 0.1,
                "II": 1.0,
            }
        },
    }
    simulated = ECGResult(
        lead_names=("I", "II"),
        time_ms=np.arange(5, dtype=float),
        values=np.stack([signal, -signal]),
        sample_rate_hz=1000.0,
        model="test",
    )
    weighted = ecg_discrepancy(raw, simulated, "correlation")
    assert weighted < 0.05


def test_ecg_quality_metadata_fails_closed_when_malformed() -> None:
    signal = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0])
    raw = {
        "lead_names": ["I"],
        "beat_template": {"I": signal.tolist()},
        "quality": {
            "lead_quality": {
                "I": {"quality_score": 1.5},
            }
        },
    }
    simulated = ECGResult(
        lead_names=("I",),
        time_ms=np.arange(5, dtype=float),
        values=signal[None, :],
        sample_rate_hz=1000.0,
        model="test",
    )
    with pytest.raises(ValueError, match="quality_score"):
        ecg_discrepancy(raw, simulated, "correlation")
