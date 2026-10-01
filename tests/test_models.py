import pytest

from cardiep import ArtifactRef, EPCalibrationRequest, EPObservation


def test_calibration_requires_valid_bounds() -> None:
    obs = EPObservation(
        observation_id="o1",
        kind="ecg",
        artifact=ArtifactRef(
            artifact_id="a",
            kind="ecg",
            uri="file:///ecg",
        ),
    )
    with pytest.raises(ValueError):
        EPCalibrationRequest(
            subject_id="S1",
            anatomy_ref=ArtifactRef(
                artifact_id="m",
                kind="mesh",
                uri="file:///mesh",
            ),
            observations=[obs],
            backend="x",
            parameter_bounds={"speed": (1.0, 1.0)},
        )


def test_calibration_requires_observation() -> None:
    with pytest.raises(ValueError):
        EPCalibrationRequest(
            subject_id="S1",
            anatomy_ref=ArtifactRef(
                artifact_id="m",
                kind="mesh",
                uri="file:///mesh",
            ),
            observations=[],
            backend="x",
            parameter_bounds={"speed": (0.1, 1.0)},
        )
