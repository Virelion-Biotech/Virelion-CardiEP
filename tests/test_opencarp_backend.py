# ruff: noqa: I001

import pytest

from cardiep import (
    ArtifactRef,
    EPParameterSet,
    EPSimulationRequest,
    EPSimulationResult,
)
from cardiep.opencarp_backend import OpenCARPExternalBackend
from cardiep.provenance import sha256_json


SHA = "a" * 64
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _activation_reference() -> dict:
    return {
        "role": "activation_target_from_fast_calibration",
        "source_backend": "camps-cdt-external-v1",
        "anatomy_bundle_fingerprint": "b" * 64,
        "parameter_mapping_fingerprint": "e" * 64,
        "activation_ref": {
            "artifact_id": "camps-activation",
            "kind": "activation_map",
            "uri": "file:///camps-activation.json",
            "sha256": "f" * 64,
            "coordinate_frame": "model",
            "metadata": {},
        },
    }


def _request(**settings) -> EPSimulationRequest:
    defaults = {
        "ecg_forward_model": "lead_field",
        "torso_mode": "standardized",
        "electrode_mode": "protocol_derived",
        "expected_leads": LEADS,
    }
    defaults.update(settings)
    return EPSimulationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="mesh",
            kind="volume_mesh",
            uri="file:///mesh.vtu",
            sha256=SHA,
            coordinate_frame="model",
            metadata={"bundle_fingerprint": "b" * 64},
        ),
        backend="opencarp-external-v1",
        parameters=EPParameterSet(
            values={"fibre_speed": 0.06},
            units={"fibre_speed": "cm/ms"},
            source="calibrated",
        ),
        settings=defaults,
    )


def _result(request: EPSimulationRequest, **updates) -> EPSimulationResult:
    provenance = {
        "engine": "openCARP",
        "opencarp_version": "test-version",
        "wrapper_version": "test-wrapper",
        "request_sha256": sha256_json(request.model_dump(mode="json")),
    }
    calibration_reference = request.settings.get("hearttwin_calibration_reference")
    if calibration_reference is not None:
        activation_ref = calibration_reference["activation_ref"]
        provenance.update(
            {
                "calibration_reference_sha256": sha256_json(calibration_reference),
                "calibration_activation_artifact_id": activation_ref["artifact_id"],
                "calibration_activation_sha256": activation_ref["sha256"],
                "stimulation_translation_method": "target-activation root/time translation",
            }
        )
    payload = {
        "subject_id": request.subject_id,
        "backend": "opencarp-external-v1",
        "parameters": request.parameters,
        "outputs": [
            ArtifactRef(
                artifact_id="activation",
                kind="activation_map",
                uri="file:///activation.dat",
                sha256="c" * 64,
                coordinate_frame="model",
            ),
            ArtifactRef(
                artifact_id="ecg",
                kind="lead_field_ecg",
                uri="file:///ecg.csv",
                sha256="d" * 64,
                metadata={
                    "lead_names": LEADS,
                    "sample_rate_hz": 1000.0,
                    "forward_model": "lead_field",
                },
            ),
        ],
        "validation_status": "software_checked",
        "provenance": provenance,
    }
    payload.update(updates)
    return EPSimulationResult.model_validate(payload)


def test_backend_is_unavailable_when_configured_wrapper_is_missing() -> None:
    backend = OpenCARPExternalBackend(
        command=["definitely-not-installed-cardiep-wrapper", "{request}", "{output}"]
    )
    assert backend.available() is False


def test_wrapper_command_requires_request_and_output_placeholders() -> None:
    with pytest.raises(ValueError, match="placeholders"):
        OpenCARPExternalBackend(command=["wrapper", "--request", "request.json"])


def test_request_requires_lead_field_forward_model() -> None:
    request = _request(ecg_forward_model="pseudo_ecg")
    with pytest.raises(ValueError, match="lead_field"):
        OpenCARPExternalBackend.validate_request(request)


def test_patient_specific_torso_requires_pinned_artifact() -> None:
    request = _request(
        torso_mode="patient_specific",
        torso_ref={"artifact_id": "torso", "uri": "file:///torso.vtu"},
    )
    with pytest.raises(ValueError, match="SHA-256"):
        OpenCARPExternalBackend.validate_request(request)


def test_recorded_electrodes_require_pinned_artifact() -> None:
    request = _request(electrode_mode="recorded")
    with pytest.raises(TypeError, match="electrode_ref"):
        OpenCARPExternalBackend.validate_request(request)


def test_valid_lead_field_result_passes_contract_validation() -> None:
    request = _request()
    OpenCARPExternalBackend.validate_request(request)
    OpenCARPExternalBackend.validate_result(request, _result(request))


def test_calibration_activation_reference_is_hash_pinned_and_lineage_matched() -> None:
    request = _request(hearttwin_calibration_reference=_activation_reference())
    OpenCARPExternalBackend.validate_request(request)
    OpenCARPExternalBackend.validate_result(request, _result(request))


def test_calibration_activation_reference_rejects_wrong_anatomy_bundle() -> None:
    reference = _activation_reference()
    reference["anatomy_bundle_fingerprint"] = "0" * 64
    request = _request(hearttwin_calibration_reference=reference)
    with pytest.raises(ValueError, match="anatomy bundle"):
        OpenCARPExternalBackend.validate_request(request)


def test_calibration_activation_reference_requires_mapping_fingerprint() -> None:
    reference = _activation_reference()
    reference.pop("parameter_mapping_fingerprint")
    request = _request(hearttwin_calibration_reference=reference)
    with pytest.raises(ValueError, match="parameter_mapping_fingerprint"):
        OpenCARPExternalBackend.validate_request(request)


def test_calibration_activation_result_must_echo_reference_and_translation_method() -> None:
    request = _request(hearttwin_calibration_reference=_activation_reference())
    result = _result(request)
    result.provenance.pop("stimulation_translation_method")
    with pytest.raises(ValueError, match="stimulation_translation_method"):
        OpenCARPExternalBackend.validate_result(request, result)


def test_pseudo_ecg_result_is_rejected() -> None:
    request = _request()
    result = _result(
        request,
        outputs=[
            ArtifactRef(
                artifact_id="activation",
                kind="activation_map",
                uri="file:///activation.dat",
                sha256="c" * 64,
            ),
            ArtifactRef(
                artifact_id="pseudo",
                kind="pseudo_ecg",
                uri="file:///pseudo.json",
                sha256="d" * 64,
            ),
        ],
    )
    with pytest.raises(ValueError, match="pseudo_ecg"):
        OpenCARPExternalBackend.validate_result(request, result)


def test_result_must_match_exact_request_hash() -> None:
    request = _request()
    result = _result(request)
    result.provenance["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        OpenCARPExternalBackend.validate_result(request, result)
