import pytest

from cardiep import (
    ArtifactRef,
    calibration_request_from_electrotrace,
    observations_from_electrotrace,
)


def _handoff():
    return {
        "schema_version": "electrotrace-ep-calibration-v1",
        "observations": [
            {
                "observation_id": "ecg-1",
                "kind": "ecg",
                "artifact": {
                    "artifact_id": "artifact-1",
                    "kind": "electrotrace_ecg_calibration",
                    "uri": "file:///tmp/ecg.json",
                    "sha256": "a" * 64,
                    "metadata": {"sampling_rate_hz": 500.0},
                },
                "coordinate_frame": "clinical_ecg",
                "units": "mV",
            }
        ],
        "likelihood_hints": [
            {
                "term_id": "ecg-1:morphology",
                "model_output": "ecg",
                "discrepancy": "correlation",
                "weight": 1.0,
                "noise_parameters": {},
                "metadata": {"artifact_field": "beat_template"},
            }
        ],
    }


def test_electrotrace_handoff_becomes_ep_observation() -> None:
    observations = observations_from_electrotrace(_handoff())
    assert len(observations) == 1
    assert observations[0].kind == "ecg"
    assert observations[0].artifact.kind == "electrotrace_ecg_calibration"


def test_electrotrace_handoff_builds_calibration_request() -> None:
    request = calibration_request_from_electrotrace(
        _handoff(),
        subject_id="S1",
        anatomy_ref=ArtifactRef(
            artifact_id="mesh",
            kind="anatomy_bundle",
            uri="file:///tmp/anatomy.json",
        ),
        backend="ep-backend",
        parameter_bounds={"fibre_speed": (0.02, 0.15)},
    )
    assert request.subject_id == "S1"
    assert request.observations[0].observation_id == "ecg-1"
    assert request.settings["measurement_handoff_schema"] == "electrotrace-ep-calibration-v1"
    assert request.settings["likelihood_hints"][0]["model_output"] == "ecg"


def test_duplicate_handoff_observation_ids_fail_closed() -> None:
    handoff = _handoff()
    handoff["observations"].append(dict(handoff["observations"][0]))
    with pytest.raises(ValueError, match="duplicate observation IDs"):
        observations_from_electrotrace(handoff)
