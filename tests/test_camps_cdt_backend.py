from __future__ import annotations

from copy import deepcopy

import pytest

from cardiep.camps_cdt_backend import (
    CAMPS_CDT_BACKEND_NAME,
    UPSTREAM_DOI,
    UPSTREAM_REPOSITORY,
    CampsCardiacDigitalTwinExternalBackend,
)
from cardiep.camps_workspace import CampsWorkspaceManifest
from cardiep.models import (
    ArtifactRef,
    EPCalibrationRequest,
    EPObservation,
    EPParameterSet,
    EPSimulationRequest,
)
from cardiep.provenance import sha256_json
from cardiep.registry import discover_backends

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
UPSTREAM_COMMIT = "c" * 40
BUNDLE_FINGERPRINT = "b" * 64


def _artifact(name: str, kind: str, *, frame: str | None = "model-mm") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=name,
        kind=kind,
        uri=f"file:///{name}.csv",
        sha256="2" * 64,
        coordinate_frame=frame,
    )


def _workspace() -> CampsWorkspaceManifest:
    return CampsWorkspaceManifest(
        subject_id="P1",
        source_bundle_fingerprint=BUNDLE_FINGERPRINT,
        coordinate_frame="model-mm",
        coordinate_unit="mm",
        index_base=0,
        node_xyz=_artifact("xyz", "node_xyz"),
        tetra=_artifact("tetra", "tetra_connectivity", frame=None),
        lvendo_nodes=_artifact("lvendo", "lvendo_nodes", frame=None),
        rvendo_nodes=_artifact("rvendo", "rvendo_nodes", frame=None),
        fibre=_artifact("fibre", "fibre_field"),
        sheet=_artifact("sheet", "sheet_field"),
        normal=_artifact("normal", "normal_field"),
        material=_artifact("material", "tetra_material", frame=None),
        vc_ab_cut=_artifact("ab-cut", "ventricular_coordinate_ab_cut"),
        vc_rt=_artifact("rt", "ventricular_coordinate_rt"),
        electrode_xyz=_artifact("electrodes", "electrode_xyz"),
    )


def _anatomy() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="patient-volume",
        kind="volume_mesh",
        uri="file:///patient.vtu",
        sha256="a" * 64,
        coordinate_frame="model-mm",
        metadata={"bundle_fingerprint": BUNDLE_FINGERPRINT},
    )


def _observation(*, sha256: str | None = "d" * 64, sample_rate_hz: float = 500.0) -> EPObservation:
    return EPObservation(
        observation_id="fit-ecg",
        kind="ecg",
        artifact=ArtifactRef(
            artifact_id="fit-ecg-file",
            kind="measured_ecg",
            uri="file:///fit.xml",
            sha256=sha256,
            metadata={"lead_names": LEADS, "sample_rate_hz": sample_rate_hz},
        ),
        units="mV",
    )


def _settings() -> dict:
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "expected_leads": LEADS,
        "calibration_target": "activation_qrs",
        "inference_method": "smc_abc",
        "upstream_sample_rate_hz": 1000.0,
        "ecg_resampling_method": "polyphase",
        "workspace_manifest": _workspace().model_dump(mode="json"),
    }


def _calibration_request(*, observation: EPObservation | None = None) -> EPCalibrationRequest:
    return EPCalibrationRequest(
        subject_id="P1",
        anatomy_ref=_anatomy(),
        observations=[observation or _observation()],
        backend=CAMPS_CDT_BACKEND_NAME,
        parameter_bounds={"fibre_speed": (0.03, 0.10)},
        initial_parameters=EPParameterSet(
            values={"fibre_speed": 0.06},
            units={"fibre_speed": "cm/ms"},
            source="prior",
        ),
        settings=_settings(),
    )


def _simulation_result_payload(request: EPSimulationRequest) -> dict:
    workspace = CampsWorkspaceManifest.model_validate(request.settings["workspace_manifest"])
    return {
        "contract_version": "1.0",
        "subject_id": request.subject_id,
        "backend": CAMPS_CDT_BACKEND_NAME,
        "parameters": request.parameters.model_dump(mode="json"),
        "outputs": [
            {
                "artifact_id": "activation",
                "kind": "activation_map",
                "uri": "file:///activation.json",
                "sha256": "e" * 64,
                "coordinate_frame": "model-mm",
                "metadata": {"units": "ms"},
            },
            {
                "artifact_id": "ecg",
                "kind": "ecg_12lead",
                "uri": "file:///ecg.json",
                "sha256": "f" * 64,
                "metadata": {
                    "lead_names": LEADS,
                    "sample_rate_hz": 1000.0,
                    "role": "calibration_forward_model",
                },
            },
        ],
        "validation_status": "software_checked",
        "warnings": [],
        "provenance": {
            "engine": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "paper_doi": UPSTREAM_DOI,
            "wrapper_version": "test-wrapper-1",
            "workspace_manifest_fingerprint": workspace.fingerprint,
            "request_sha256": sha256_json(request.model_dump(mode="json")),
        },
    }


def _calibration_result_payload(request: EPCalibrationRequest) -> dict:
    parameters = EPParameterSet(
        values={"fibre_speed": 0.061},
        units={"fibre_speed": "cm/ms"},
        source="calibrated",
    )
    replay = EPSimulationRequest(
        subject_id=request.subject_id,
        anatomy_ref=request.anatomy_ref,
        backend=request.backend,
        parameters=parameters,
        observations=[],
        settings=request.settings,
    )
    workspace = CampsWorkspaceManifest.model_validate(request.settings["workspace_manifest"])
    return {
        "contract_version": "1.0",
        "subject_id": request.subject_id,
        "backend": CAMPS_CDT_BACKEND_NAME,
        "parameters": parameters.model_dump(mode="json"),
        "objective": 0.12,
        "converged": True,
        "posterior_ref": {
            "artifact_id": "posterior",
            "kind": "smc_abc_posterior",
            "uri": "file:///posterior.json",
            "sha256": "1" * 64,
            "metadata": {"population_size": 512},
        },
        "simulated": _simulation_result_payload(replay),
        "diagnostics": {
            "inference_method": "smc_abc",
            "posterior_population_size": 512,
            "discrepancy_metric": "qrs_ecg_discrepancy",
        },
        "provenance": {
            "engine": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "paper_doi": UPSTREAM_DOI,
            "wrapper_version": "test-wrapper-1",
            "workspace_manifest_fingerprint": workspace.fingerprint,
            "ecg_preprocessing": {
                "source_sample_rates_hz": [500.0],
                "target_sample_rate_hz": 1000.0,
                "method": "polyphase",
            },
            "request_sha256": sha256_json(request.model_dump(mode="json")),
        },
    }


class _FakeCampsBackend(CampsCardiacDigitalTwinExternalBackend):
    def __init__(self, result: dict) -> None:
        super().__init__(
            command=["true", "{request}", "{output}"],
            executable="true",
        )
        self.result = result

    def _invoke(self, request):
        return deepcopy(self.result)


def test_registry_exposes_camps_backend_even_when_wrapper_is_not_installed() -> None:
    assert CAMPS_CDT_BACKEND_NAME in discover_backends()


def test_calibration_requires_hash_pinned_measured_ecg() -> None:
    request = _calibration_request(observation=_observation(sha256=None))
    backend = _FakeCampsBackend({})
    with pytest.raises(ValueError, match="artifact.sha256"):
        backend.calibrate(request)


def test_calibration_requires_explicit_resampling_for_non_1000_hz_input() -> None:
    request = _calibration_request()
    request.settings["ecg_resampling_method"] = "none"
    backend = _FakeCampsBackend({})
    with pytest.raises(ValueError, match="declare ecg_resampling_method='polyphase'"):
        backend.calibrate(request)


def test_calibration_accepts_pinned_published_pipeline_contract() -> None:
    request = _calibration_request()
    backend = _FakeCampsBackend(_calibration_result_payload(request))
    result = backend.calibrate(request)
    assert result.parameters.source == "calibrated"
    assert result.posterior_ref is not None
    assert result.diagnostics["posterior_population_size"] == 512
    assert result.simulated is not None
    assert {item.kind for item in result.simulated.outputs} == {
        "activation_map",
        "ecg_12lead",
    }


def test_calibration_refuses_missing_posterior_population() -> None:
    request = _calibration_request()
    payload = _calibration_result_payload(request)
    payload["posterior_ref"] = None
    backend = _FakeCampsBackend(payload)
    with pytest.raises(ValueError, match="posterior_ref"):
        backend.calibrate(request)


def test_calibration_refuses_unpinned_or_changed_upstream_code() -> None:
    request = _calibration_request()
    payload = _calibration_result_payload(request)
    payload["provenance"]["upstream_commit"] = "9" * 40
    backend = _FakeCampsBackend(payload)
    with pytest.raises(ValueError, match="upstream_commit does not match"):
        backend.calibrate(request)


def test_forward_replay_must_not_receive_fit_or_holdout_observations() -> None:
    request = EPSimulationRequest(
        subject_id="P1",
        anatomy_ref=_anatomy(),
        backend=CAMPS_CDT_BACKEND_NAME,
        parameters=EPParameterSet(
            values={"fibre_speed": 0.061},
            units={"fibre_speed": "cm/ms"},
            source="calibrated",
        ),
        observations=[_observation()],
        settings=_settings(),
    )
    backend = _FakeCampsBackend({})
    with pytest.raises(ValueError, match="must be unconditional"):
        backend.simulate(request)
