"""Strict external openCARP confirmation backend.

CardiEP does not vendor openCARP. This adapter executes a user-installed wrapper
that translates the stable CardiEP JSON contract into a reproducible openCARP /
carputils experiment and translates the resulting artifacts back into CardiEP.

The wrapper boundary is intentionally stricter than ``SubprocessEPBackend``:
scientific-grade ECG confirmation must use an explicit lead-field forward model,
return a standard 12-lead ECG artifact, preserve the exact request hash, and
report openCARP/wrapper versions. A pseudo-ECG is rejected.

When confirmation follows a heterogeneous fast calibration model, the caller may
also provide a hash-pinned activation-map reference. The wrapper must explicitly
report how that calibration-derived activation target was translated into the
openCARP stimulation setup; it may not silently reinterpret discrete root-site
structure as scalar conductivity parameters.
"""
from __future__ import annotations

import os
import shlex
import string
from typing import Any

from .backends import BackendUnavailable
from .external import SubprocessEPBackend
from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPSimulationRequest,
    EPSimulationResult,
)
from .provenance import sha256_json

OPENCARP_BACKEND_NAME = "opencarp-external-v1"
_CALIBRATION_REFERENCE_KEY = "hearttwin_calibration_reference"
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


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in string.hexdigits for character in text):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex string")
    return text


class OpenCARPExternalBackend(SubprocessEPBackend):
    """Confirmation-only openCARP backend using an explicit JSON wrapper.

    Configuration
    -------------
    ``CARDIEP_OPENCARP_WRAPPER`` may contain a command template with
    ``{request}`` and ``{output}`` placeholders, for example::

        /opt/virelion/bin/cardiep-opencarp-wrapper --request {request} --output {output}

    The default command name is ``cardiep-opencarp-wrapper`` so the backend is
    visible in backend status but remains unavailable unless a wrapper is
    deliberately installed on ``PATH``.
    """

    name = OPENCARP_BACKEND_NAME

    def __init__(
        self,
        *,
        command: str | list[str] | None = None,
        executable: str | None = None,
        timeout_s: float = 7200.0,
    ) -> None:
        configured = command
        if configured is None:
            configured = os.environ.get("CARDIEP_OPENCARP_WRAPPER")
        if configured is None:
            configured = [
                "cardiep-opencarp-wrapper",
                "--request",
                "{request}",
                "--output",
                "{output}",
            ]
        elif isinstance(configured, str):
            configured = shlex.split(configured)
        if "{request}" not in configured or "{output}" not in configured:
            raise ValueError(
                "openCARP wrapper command must contain {request} and {output} placeholders"
            )
        super().__init__(
            name=self.name,
            command=configured,
            executable=executable,
            timeout_s=timeout_s,
        )

    @staticmethod
    def _validate_artifact_ref(value: Any, *, setting: str) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"{setting} must be an artifact reference object")
        if not value.get("artifact_id") or not value.get("uri"):
            raise ValueError(f"{setting} requires artifact_id and uri")
        _require_sha256(value.get("sha256"), field=f"{setting}.sha256")

    @classmethod
    def _validate_calibration_reference(
        cls,
        request: EPSimulationRequest,
    ) -> dict[str, Any] | None:
        value = request.settings.get(_CALIBRATION_REFERENCE_KEY)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"settings.{_CALIBRATION_REFERENCE_KEY} must be an object")
        if value.get("role") != "activation_target_from_fast_calibration":
            raise ValueError(
                f"settings.{_CALIBRATION_REFERENCE_KEY}.role must be "
                "'activation_target_from_fast_calibration'"
            )
        source_backend = str(value.get("source_backend") or "").strip()
        if not source_backend or source_backend == OPENCARP_BACKEND_NAME:
            raise ValueError(
                "calibration activation reference requires a different explicit source_backend"
            )
        activation_ref = value.get("activation_ref")
        cls._validate_artifact_ref(
            activation_ref,
            setting=f"settings.{_CALIBRATION_REFERENCE_KEY}.activation_ref",
        )
        if activation_ref.get("kind") != "activation_map":
            raise ValueError("calibration activation reference must have kind='activation_map'")
        expected_bundle = request.anatomy_ref.metadata.get("bundle_fingerprint")
        reference_bundle = value.get("anatomy_bundle_fingerprint")
        if not isinstance(expected_bundle, str) or not expected_bundle:
            raise ValueError("openCARP anatomy_ref requires metadata.bundle_fingerprint")
        if str(reference_bundle) != expected_bundle:
            raise ValueError(
                "calibration activation reference anatomy bundle does not match openCARP anatomy"
            )
        activation_frame = activation_ref.get("coordinate_frame")
        anatomy_frame = request.anatomy_ref.coordinate_frame
        if activation_frame and anatomy_frame and str(activation_frame) != str(anatomy_frame):
            raise ValueError(
                "calibration activation reference coordinate frame does not match openCARP anatomy"
            )
        _require_sha256(
            value.get("parameter_mapping_fingerprint"),
            field=(
                f"settings.{_CALIBRATION_REFERENCE_KEY}.parameter_mapping_fingerprint"
            ),
        )
        return value

    @classmethod
    def validate_request(cls, request: EPSimulationRequest) -> None:
        settings = request.settings
        model = str(settings.get("ecg_forward_model", "")).strip().lower()
        if model != "lead_field":
            raise ValueError(
                "opencarp-external-v1 requires settings.ecg_forward_model='lead_field'"
            )

        torso_mode = str(settings.get("torso_mode", "")).strip().lower()
        if torso_mode not in {"patient_specific", "standardized"}:
            raise ValueError(
                "openCARP lead-field confirmation requires torso_mode patient_specific or standardized"
            )
        if torso_mode == "patient_specific":
            cls._validate_artifact_ref(settings.get("torso_ref"), setting="settings.torso_ref")

        electrode_mode = str(settings.get("electrode_mode", "")).strip().lower()
        if electrode_mode not in {"recorded", "protocol_derived"}:
            raise ValueError(
                "openCARP lead-field confirmation requires electrode_mode recorded or protocol_derived"
            )
        if electrode_mode == "recorded":
            cls._validate_artifact_ref(
                settings.get("electrode_ref"), setting="settings.electrode_ref"
            )

        expected = settings.get("expected_leads")
        if expected is None:
            expected = sorted(_STANDARD_12_LEADS)
        if not isinstance(expected, list) or set(map(str, expected)) != _STANDARD_12_LEADS:
            raise ValueError("openCARP patient ECG confirmation currently requires standard 12 leads")

        cls._validate_calibration_reference(request)

    @staticmethod
    def _output_of_kind(result: EPSimulationResult, kinds: set[str]):
        matches = [item for item in result.outputs if item.kind in kinds]
        if len(matches) != 1:
            raise ValueError(
                "openCARP result must contain exactly one artifact of kind(s): "
                + ", ".join(sorted(kinds))
            )
        return matches[0]

    @classmethod
    def validate_result(
        cls,
        request: EPSimulationRequest,
        result: EPSimulationResult,
    ) -> None:
        if any(item.kind == "pseudo_ecg" for item in result.outputs):
            raise ValueError("openCARP scientific confirmation may not return pseudo_ecg")

        cls._output_of_kind(result, {"activation_map"})
        ecg = cls._output_of_kind(result, {"ecg_12lead", "lead_field_ecg"})
        leads = ecg.metadata.get("lead_names")
        if not isinstance(leads, list) or set(map(str, leads)) != _STANDARD_12_LEADS:
            raise ValueError("openCARP ECG artifact must declare all standard 12 lead_names")
        sample_rate = ecg.metadata.get("sample_rate_hz")
        if sample_rate is None or float(sample_rate) <= 0:
            raise ValueError("openCARP ECG artifact must declare a positive sample_rate_hz")
        forward_model = str(ecg.metadata.get("forward_model", "")).strip().lower()
        if forward_model != "lead_field":
            raise ValueError("openCARP ECG artifact must declare forward_model='lead_field'")

        provenance = result.provenance
        engine = str(provenance.get("engine", "")).strip().lower()
        if engine != "opencarp":
            raise ValueError("openCARP result provenance must declare engine='openCARP'")
        for field in ("opencarp_version", "wrapper_version", "request_sha256"):
            if not provenance.get(field):
                raise ValueError(f"openCARP result provenance requires {field}")
        expected_sha = sha256_json(request.model_dump(mode="json"))
        if str(provenance["request_sha256"]).lower() != expected_sha:
            raise ValueError("openCARP wrapper result request_sha256 does not match request")

        calibration_reference = cls._validate_calibration_reference(request)
        if calibration_reference is not None:
            expected_reference_sha = sha256_json(calibration_reference)
            observed_reference_sha = _require_sha256(
                provenance.get("calibration_reference_sha256"),
                field="provenance.calibration_reference_sha256",
            )
            if observed_reference_sha != expected_reference_sha:
                raise ValueError(
                    "openCARP calibration_reference_sha256 does not match request"
                )
            method = str(provenance.get("stimulation_translation_method") or "").strip()
            if not method:
                raise ValueError(
                    "openCARP result provenance requires stimulation_translation_method "
                    "when a fast-calibration activation target is supplied"
                )
            activation_ref = calibration_reference["activation_ref"]
            if str(provenance.get("calibration_activation_artifact_id")) != str(
                activation_ref["artifact_id"]
            ):
                raise ValueError(
                    "openCARP result does not identify the requested calibration activation artifact"
                )
            if _require_sha256(
                provenance.get("calibration_activation_sha256"),
                field="provenance.calibration_activation_sha256",
            ) != str(activation_ref["sha256"]).lower():
                raise ValueError(
                    "openCARP result calibration activation hash does not match request"
                )

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        self.validate_request(request)
        result = EPSimulationResult.model_validate(self._invoke(request))
        self.validate_result(request, result)
        return result

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        raise BackendUnavailable(
            "opencarp-external-v1 is confirmation-only; perform inverse inference around "
            "a fast forward model and use openCARP for independent high-fidelity replay"
        )
