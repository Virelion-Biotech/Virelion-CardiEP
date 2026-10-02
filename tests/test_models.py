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



def test_artifact_ref_preserves_optional_coordinate_frame() -> None:
    ref = ArtifactRef(
        artifact_id="mesh",
        kind="volume_mesh",
        uri="file:///mesh.npz",
        coordinate_frame="patient-LPS-mm",
    )
    payload = ref.model_dump(mode="json")
    assert payload["coordinate_frame"] == "patient-LPS-mm"
    assert ArtifactRef.model_validate(payload) == ref


def test_artifact_ref_accepts_cross_service_null_coordinate_frame() -> None:
    ref = ArtifactRef.model_validate(
        {
            "artifact_id": "mesh",
            "kind": "volume_mesh",
            "uri": "file:///mesh.npz",
            "sha256": None,
            "coordinate_frame": None,
            "metadata": {},
        }
    )
    assert ref.coordinate_frame is None
