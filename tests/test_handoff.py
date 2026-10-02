import pytest

from cardiep import calibration_request_from_electrotrace


def _handoff(entity_id: str = "S1") -> dict:
    return {
        "schema_version": "electrotrace-ep-calibration-v1",
        "entity_id": entity_id,
        "observations": [
            {
                "observation_id": "ecg-1",
                "kind": "ecg",
                "artifact": {
                    "artifact_id": "obs-artifact",
                    "kind": "electrotrace_ecg_calibration",
                    "uri": "file:///tmp/ecg.json",
                    "sha256": "a" * 64,
                },
                "coordinate_frame": "clinical_ecg",
                "units": "mV",
            }
        ],
    }


def test_calibration_builder_rejects_cross_subject_handoff() -> None:
    with pytest.raises(ValueError, match="does not match"):
        calibration_request_from_electrotrace(
            _handoff("S2"),
            subject_id="S1",
            anatomy_ref={
                "artifact_id": "mesh",
                "kind": "ep_geometry",
                "uri": "file:///tmp/mesh.json",
            },
            backend="numpy-eikonal-v1",
            parameter_bounds={"fibre_speed": (0.05, 0.15)},
        )


def test_calibration_builder_accepts_matching_subject_handoff() -> None:
    request = calibration_request_from_electrotrace(
        _handoff("S1"),
        subject_id="S1",
        anatomy_ref={
            "artifact_id": "mesh",
            "kind": "ep_geometry",
            "uri": "file:///tmp/mesh.json",
        },
        backend="numpy-eikonal-v1",
        parameter_bounds={"fibre_speed": (0.05, 0.15)},
    )
    assert request.subject_id == "S1"
    assert request.observations[0].artifact.artifact_id == "obs-artifact"
