"""Strict external adapter for the Camps Cardiac-Digital-Twin pipeline.

The upstream research code remains external and MIT-licensed. CardiEP exposes a
stable JSON boundary around a deliberately installed wrapper rather than copying
or silently modifying the scientific implementation.

This backend is intended for fast ventricular activation/QRS calibration from
patient anatomy and standard 12-lead ECG. It does not itself establish that a
particular wrapper, dataset, parameterization, or inferred patient model is
clinically valid.
"""
from __future__ import annotations

import os
import shlex
import string
from typing import Any

from .external import SubprocessEPBackend
from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPSimulationRequest,
    EPSimulationResult,
)
from .provenance import sha256_json

CAMPS_CDT_BACKEND_NAME = "camps-cdt-external-v1"
UPSTREAM_REPOSITORY = "juliacamps/Cardiac-Digital-Twin"
UPSTREAM_DOI = "10.1016/j.media.2024.103361"
_STANDARD_12_LEADS = {
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
}


def _require_sha256(value: str | None, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex string")
    return value.lower()


def _require_git_commit(value: Any, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 40 or any(character not in string.hexdigits for character in text):
        raise ValueError(f"{field} must be a pinned 40-character Git commit SHA")
    return text


def _expected_leads(settings: dict[str, Any]) -> list[str]:
    expected = settings.get("expected_leads", sorted(_STANDARD_12_LEADS))
    if not isinstance(expected, list) or set(map(str, expected)) != _STANDARD_12_LEADS:
        raise ValueError("Camps CDT calibration currently requires the standard 12 leads")
    return [str(item) for item in expected]


class CampsCardiacDigitalTwinExternalBackend(SubprocessEPBackend):
    """External wrapper for published ECG/CMR ventricular digital twinning.

    ``CARDIEP_CAMPS_CDT_WRAPPER`` may point to a command containing
    ``{request}`` and ``{output}`` placeholders. If unset, CardiEP looks for::

        cardiep-camps-cdt-wrapper --request {request} --output {output}

    The wrapper is expected to translate the stable CardiEP contract into the
    selected upstream scripts/environment and return a validated CardiEP result.
    """

    name = CAMPS_CDT_BACKEND_NAME

    def __init__(
        self,
        *,
        command: str | list[str] | None = None,
        executable: str | None = None,
        timeout_s: float = 14400.0,
    ) -> None:
        configured = command
        if configured is None:
            configured = os.environ.get("CARDIEP_CAMPS_CDT_WRAPPER")
        if configured is None:
            configured = [
                "cardiep-camps-cdt-wrapper",
                "--request",
                "{request}",
                "--output",
                "{output}",
            ]
        elif isinstance(configured, str):
            configured = shlex.split(configured)
        if "{request}" not in configured or "{output}" not in configured:
            raise ValueError(
                "Camps CDT wrapper command must contain {request} and {output} placeholders"
            )
        super().__init__(
            name=self.name,
            command=configured,
            executable=executable,
            timeout_s=timeout_s,
        )

    @staticmethod
    def _validate_common_request(request: EPSimulationRequest | EPCalibrationRequest) -> str:
        _require_sha256(request.anatomy_ref.sha256, field="anatomy_ref.sha256")
        if not request.anatomy_ref.coordinate_frame:
            raise ValueError("Camps CDT patient modeling requires an explicit anatomy coordinate frame")
        upstream_commit = _require_git_commit(
            request.settings.get("upstream_commit"),
            field="settings.upstream_commit",
        )
        _expected_leads(request.settings)
        target = str(request.settings.get("calibration_target", "activation_qrs")).strip().lower()
        if target != "activation_qrs":
            raise ValueError(
                "camps-cdt-external-v1 currently supports calibration_target='activation_qrs' only"
            )
        return upstream_commit

    @classmethod
    def validate_simulation_request(cls, request: EPSimulationRequest) -> str:
        upstream_commit = cls._validate_common_request(request)
        if request.observations:
            raise ValueError(
                "Camps CDT forward replay must be unconditional; observations belong in calibration/evaluation"
            )
        return upstream_commit

    @classmethod
    def validate_calibration_request(cls, request: EPCalibrationRequest) -> str:
        upstream_commit = cls._validate_common_request(request)
        if not request.observations:
            raise ValueError("Camps CDT calibration requires at least one ECG observation")
        for observation in request.observations:
            if observation.kind != "ecg":
                raise ValueError("Camps CDT activation calibration accepts ECG observations only")
            _require_sha256(
                observation.artifact.sha256,
                field=f"observation {observation.observation_id!r} artifact.sha256",
            )
        method = str(request.settings.get("inference_method", "smc_abc")).strip().lower()
        if method != "smc_abc":
            raise ValueError("camps-cdt-external-v1 currently requires inference_method='smc_abc'")
        return upstream_commit

    @staticmethod
    def _one_output(result: EPSimulationResult, kinds: set[str]):
        matches = [item for item in result.outputs if item.kind in kinds]
        if len(matches) != 1:
            raise ValueError(
                "Camps CDT result must contain exactly one artifact of kind(s): "
                + ", ".join(sorted(kinds))
            )
        return matches[0]

    @classmethod
    def _validate_simulation_result(
        cls,
        request: EPSimulationRequest,
        result: EPSimulationResult,
        *,
        upstream_commit: str,
    ) -> None:
        activation = cls._one_output(result, {"activation_map"})
        ecg = cls._one_output(result, {"ecg_12lead", "calibration_ecg"})
        _require_sha256(activation.sha256, field="activation_map.sha256")
        _require_sha256(ecg.sha256, field="ECG artifact sha256")

        leads = ecg.metadata.get("lead_names")
        if not isinstance(leads, list) or set(map(str, leads)) != _STANDARD_12_LEADS:
            raise ValueError("Camps CDT ECG artifact must declare all standard 12 lead_names")
        sample_rate = ecg.metadata.get("sample_rate_hz")
        if sample_rate is None or float(sample_rate) <= 0:
            raise ValueError("Camps CDT ECG artifact must declare a positive sample_rate_hz")
        role = str(ecg.metadata.get("role", "")).strip().lower()
        if role not in {"calibration_forward_model", "forward_prediction"}:
            raise ValueError(
                "Camps CDT ECG artifact must declare role calibration_forward_model or forward_prediction"
            )

        provenance = result.provenance
        if str(provenance.get("engine", "")).strip() != UPSTREAM_REPOSITORY:
            raise ValueError(
                f"Camps CDT result provenance must declare engine={UPSTREAM_REPOSITORY!r}"
            )
        observed_commit = _require_git_commit(
            provenance.get("upstream_commit"), field="provenance.upstream_commit"
        )
        if observed_commit != upstream_commit:
            raise ValueError("Camps CDT result upstream_commit does not match request")
        if not provenance.get("wrapper_version"):
            raise ValueError("Camps CDT result provenance requires wrapper_version")
        if str(provenance.get("paper_doi", "")).strip() != UPSTREAM_DOI:
            raise ValueError(f"Camps CDT result provenance requires paper_doi={UPSTREAM_DOI!r}")
        expected_request_sha = sha256_json(request.model_dump(mode="json"))
        if str(provenance.get("request_sha256", "")).lower() != expected_request_sha:
            raise ValueError("Camps CDT result request_sha256 does not match request")

    @classmethod
    def validate_calibration_result(
        cls,
        request: EPCalibrationRequest,
        result: EPCalibrationResult,
        *,
        upstream_commit: str,
    ) -> None:
        if result.parameters.source != "calibrated":
            raise ValueError("Camps CDT calibration result parameters must be marked calibrated")
        if result.posterior_ref is None:
            raise ValueError("Camps CDT calibration must return a posterior_ref")
        _require_sha256(result.posterior_ref.sha256, field="posterior_ref.sha256")
        if result.simulated is None:
            raise ValueError("Camps CDT calibration must return its best/representative simulation")

        simulation_request = EPSimulationRequest(
            subject_id=request.subject_id,
            anatomy_ref=request.anatomy_ref,
            backend=request.backend,
            parameters=result.parameters,
            observations=[],
            settings=request.settings,
        )
        cls._validate_simulation_result(
            simulation_request,
            result.simulated,
            upstream_commit=upstream_commit,
        )

        diagnostics = result.diagnostics
        population_size = diagnostics.get("posterior_population_size")
        if population_size is None or int(population_size) < 1:
            raise ValueError("Camps CDT calibration diagnostics require posterior_population_size >= 1")
        method = str(diagnostics.get("inference_method", "")).strip().lower()
        if method != "smc_abc":
            raise ValueError("Camps CDT calibration diagnostics must declare inference_method='smc_abc'")
        if not diagnostics.get("discrepancy_metric"):
            raise ValueError("Camps CDT calibration diagnostics require discrepancy_metric")

        provenance = result.provenance
        if str(provenance.get("engine", "")).strip() != UPSTREAM_REPOSITORY:
            raise ValueError(
                f"Camps CDT calibration provenance must declare engine={UPSTREAM_REPOSITORY!r}"
            )
        if _require_git_commit(
            provenance.get("upstream_commit"), field="provenance.upstream_commit"
        ) != upstream_commit:
            raise ValueError("Camps CDT calibration upstream_commit does not match request")
        if not provenance.get("wrapper_version"):
            raise ValueError("Camps CDT calibration provenance requires wrapper_version")
        if str(provenance.get("paper_doi", "")).strip() != UPSTREAM_DOI:
            raise ValueError(f"Camps CDT calibration provenance requires paper_doi={UPSTREAM_DOI!r}")
        expected_request_sha = sha256_json(request.model_dump(mode="json"))
        if str(provenance.get("request_sha256", "")).lower() != expected_request_sha:
            raise ValueError("Camps CDT calibration request_sha256 does not match request")

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        upstream_commit = self.validate_simulation_request(request)
        result = EPSimulationResult.model_validate(self._invoke(request))
        self._validate_simulation_result(
            request,
            result,
            upstream_commit=upstream_commit,
        )
        return result

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        upstream_commit = self.validate_calibration_request(request)
        result = EPCalibrationResult.model_validate(self._invoke(request))
        self.validate_calibration_result(
            request,
            result,
            upstream_commit=upstream_commit,
        )
        return result
