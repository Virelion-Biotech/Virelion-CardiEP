"""Patient-specific recorded-electrode openCARP QRS confirmation profile.

This profile transfers a hash-pinned activation field produced by an explicit
fast calibration backend into openCARP as a prescribed activation sequence.
The openCARP workspace must contain precomputed lead fields generated from the
same hash-pinned patient torso and recorded electrode geometry supplied in the
request. No scalar conduction parameters are reinterpreted across backends.

Scientific claim: patient-specific lead-field ECG confirmation conditioned on
the calibrated activation sequence. Propagation is not independently solved by
openCARP in this profile.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import stat
import zipfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import write_json_artifact
from .models import EPSimulationRequest, EPSimulationResult
from .opencarp_backend import OPENCARP_BACKEND_NAME, OpenCARPExternalBackend
from .opencarp_experiment import OpenCARPExperimentSpec
from .opencarp_patient_qrs_v1 import (
    PatientQRSProfileError,
    _REQUIRED_ELECTRODES,
    _STANDARD_12,
    _load_activation,
    _mesh_node_count,
    _opencarp_version,
    _parse_ecg_files,
    _run_command,
    _staged,
    _write_activation_igb,
)
from .opencarp_runtime import WRAPPER_VERSION
from .provenance import file_sha256, sha256_json

EXPERIMENT_ID = "virelion-patient-qrs-recorded-lf12"
EXPERIMENT_VERSION = "1.0.0"
DRIVER_VERSION = "cardiep-opencarp-qrs-recorded-lf12-v1"
WORKSPACE_SCHEMA = "cardiep-opencarp-recorded-workspace-v1"


class RecordedOpenCARPWorkspaceManifest(BaseModel):
    """Hash inventory for one patient heart/torso/recorded-electrode workspace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = WORKSPACE_SCHEMA
    experiment_id: str
    experiment_version: str
    experiment_fingerprint: str = Field(min_length=64, max_length=64)
    source_anatomy_sha256: str = Field(min_length=64, max_length=64)
    source_torso_sha256: str = Field(min_length=64, max_length=64)
    source_electrode_sha256: str = Field(min_length=64, max_length=64)
    coordinate_frame: str
    mesh_basename: str
    electrodes_dir: str
    leadfield_dir: str
    leadfield_basis: str
    files: dict[str, str]

    @model_validator(mode="after")
    def validate_manifest(self) -> RecordedOpenCARPWorkspaceManifest:
        if self.experiment_id != EXPERIMENT_ID or self.experiment_version != EXPERIMENT_VERSION:
            raise ValueError("workspace experiment identity does not match recorded-LF12 profile")
        if not self.coordinate_frame.strip():
            raise ValueError("workspace coordinate_frame must be explicit")
        if self.leadfield_basis != "patient_specific_torso_and_recorded_electrodes":
            raise ValueError(
                "workspace leadfield_basis must declare patient-specific torso and recorded electrodes"
            )
        for field_name in ("mesh_basename", "electrodes_dir", "leadfield_dir"):
            value = str(getattr(self, field_name)).strip()
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"workspace {field_name} must be a safe relative path")
        if not self.files:
            raise ValueError("workspace must inventory all scientific files")
        for relative, digest in self.files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("workspace file paths must be relative and traversal-free")
            lowered = str(digest).lower()
            if len(lowered) != 64 or any(char not in "0123456789abcdef" for char in lowered):
                raise ValueError(f"workspace file {relative!r} has invalid SHA-256")
        return self


def profile_experiment_spec() -> OpenCARPExperimentSpec:
    """Return the immutable patient-specific recorded-electrode declaration."""
    return OpenCARPExperimentSpec.model_validate(
        {
            "contract_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "experiment_version": EXPERIMENT_VERSION,
            "intended_use": (
                "Research-only patient-specific 12-lead lead-field ECG confirmation "
                "conditioned on one hash-pinned calibrated ventricular activation sequence."
            ),
            "field_model": "monodomain",
            "activation_model": "prescribed_activation",
            "tissue_regions": [
                {
                    "region_id": "ventricular_myocardium",
                    "element_tags": [1],
                    "ionic_model": "carputils_fake_ep_standard_pulse",
                    "ionic_parameters": {},
                }
            ],
            "fibre_field_required": False,
            "parameter_bindings": [],
            "activation_translation": {
                "strategy_id": "fast_activation_to_shifted_igb",
                "strategy_version": "1.0",
                "settings": {
                    "input_units": "ms",
                    "ecg_reference_shift_ms": 250.0,
                    "preserve_relative_activation": True,
                },
                "rationale": (
                    "Preserve the calibrated activation sequence exactly while adding "
                    "a fixed pre-QRS baseline before the openCARP fake-EP waveform."
                ),
            },
            "time_grid": {
                "start_ms": -250.0,
                "end_ms": 450.0,
                "integration_dt_ms": 0.02,
                "output_dt_ms": 1.0,
            },
            "ecg": {
                "forward_model": "lead_field",
                "torso_mode": "patient_specific",
                "electrode_mode": "recorded",
                "expected_leads": _STANDARD_12,
                "output_voltage_unit": "mV",
                "output_sample_rate_hz": 1000.0,
                "time_axis_start_ms": -250.0,
                "time_axis_end_ms": 450.0,
                "simulation_zero_relative_to_ecg_reference_ms": -250.0,
                "volume_conductor_regions": [
                    {
                        "region_id": "patient_torso",
                        "element_tags": [100],
                        "conductivity_s_per_m": 0.2,
                    }
                ],
            },
            "solver_options": {
                "leadfield_configuration": 12,
                "leadfield_post_processing_opts": 64,
                "fake_ep_pulse": "standard_shaped_foot",
                "leadfields_precomputed_from_patient_volume_conductor": True,
            },
            "provenance_reference": (
                "Virelion patient-specific recorded-electrode prescribed-activation LF12 profile v1.0.0"
            ),
        }
    )


def _safe_extract_workspace(
    archive: Path,
    destination: Path,
) -> RecordedOpenCARPWorkspaceManifest:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive, "r") as handle:
            infos = handle.infolist()
            for info in infos:
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise PatientQRSProfileError("workspace ZIP contains path traversal")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise PatientQRSProfileError("workspace ZIP may not contain symbolic links")
                target = (destination / relative).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise PatientQRSProfileError("workspace ZIP escapes extraction root")
            handle.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PatientQRSProfileError("openCARP workspace is not a valid ZIP archive") from exc

    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise PatientQRSProfileError("openCARP workspace has no manifest.json")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = RecordedOpenCARPWorkspaceManifest.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise PatientQRSProfileError(f"invalid openCARP workspace manifest: {exc}") from exc

    expected_files = set(manifest.files)
    observed_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if observed_files != expected_files:
        raise PatientQRSProfileError(
            "workspace file inventory differs from manifest; refusing untracked scientific inputs"
        )
    for relative, expected in manifest.files.items():
        if file_sha256(destination / relative) != expected.lower():
            raise PatientQRSProfileError(f"workspace hash mismatch for {relative}")
    return manifest


def _request_ref_sha(ref: dict, *, name: str) -> str:
    value = str(ref.get("sha256") or "").lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PatientQRSProfileError(f"request {name} requires a valid SHA-256")
    return value


class PatientQRSRecordedLF12Driver:
    """Exact patient-specific recorded-electrode prescribed-activation driver."""

    experiment_id = EXPERIMENT_ID
    experiment_version = EXPERIMENT_VERSION
    driver_version = DRIVER_VERSION

    def available(self) -> bool:
        executable = os.environ.get("CARDIEP_OPENCARP_EXECUTABLE", "openCARP")
        return shutil.which(executable) is not None and importlib.util.find_spec("carputils") is not None

    def run(
        self,
        *,
        plan: dict,
        experiment: OpenCARPExperimentSpec,
        output_path: str,
    ) -> None:
        expected = profile_experiment_spec()
        if experiment.fingerprint != expected.fingerprint:
            raise PatientQRSProfileError(
                "declared experiment differs from the fingerprint-locked recorded-LF12 profile"
            )

        request = EPSimulationRequest.model_validate(plan.get("request"))
        if request.backend != OPENCARP_BACKEND_NAME:
            raise PatientQRSProfileError("profile can run only opencarp-external-v1 requests")
        if request.parameters.values or request.parameters.units:
            raise PatientQRSProfileError(
                "prescribed-activation profile accepts no scalar parameter replay; "
                "the calibrated activation field is the transferred state"
            )
        if str(request.settings.get("torso_mode", "")).lower() != "patient_specific":
            raise PatientQRSProfileError("recorded-LF12 profile requires patient_specific torso_mode")
        if str(request.settings.get("electrode_mode", "")).lower() != "recorded":
            raise PatientQRSProfileError("recorded-LF12 profile requires recorded electrode_mode")

        torso_ref = request.settings.get("torso_ref")
        electrode_ref = request.settings.get("electrode_ref")
        if not isinstance(torso_ref, dict) or not isinstance(electrode_ref, dict):
            raise PatientQRSProfileError("recorded-LF12 profile requires torso_ref and electrode_ref")

        workspace_artifact = _staged(plan, "opencarp_workspace")
        activation_artifact = _staged(plan, "calibration_activation")
        staged_torso = _staged(plan, "torso")
        staged_electrodes = _staged(plan, "electrodes")
        workspace_archive = Path(str(workspace_artifact["staged_path"])).resolve()
        activation_path = Path(str(activation_artifact["staged_path"])).resolve()
        run_root = Path(str(plan["runtime_root"])).resolve() / "recorded-lf12-v1"
        workspace_root = run_root / "workspace"
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
        manifest = _safe_extract_workspace(workspace_archive, workspace_root)

        if manifest.experiment_fingerprint != experiment.fingerprint:
            raise PatientQRSProfileError("workspace was not built for this experiment fingerprint")
        if manifest.source_anatomy_sha256.lower() != str(request.anatomy_ref.sha256).lower():
            raise PatientQRSProfileError("workspace anatomy hash does not match patient request")
        expected_torso_sha = _request_ref_sha(torso_ref, name="torso_ref")
        expected_electrode_sha = _request_ref_sha(electrode_ref, name="electrode_ref")
        if manifest.source_torso_sha256.lower() != expected_torso_sha:
            raise PatientQRSProfileError("workspace torso hash does not match patient request")
        if manifest.source_electrode_sha256.lower() != expected_electrode_sha:
            raise PatientQRSProfileError("workspace electrode hash does not match patient request")
        if str(staged_torso["sha256"]).lower() != expected_torso_sha:
            raise PatientQRSProfileError("staged torso hash does not match patient request")
        if str(staged_electrodes["sha256"]).lower() != expected_electrode_sha:
            raise PatientQRSProfileError("staged electrode hash does not match patient request")
        if manifest.coordinate_frame != request.anatomy_ref.coordinate_frame:
            raise PatientQRSProfileError("workspace coordinate frame does not match anatomy")
        for ref_name, ref in (("torso", torso_ref), ("electrodes", electrode_ref)):
            frame = ref.get("coordinate_frame")
            if frame is not None and str(frame) != manifest.coordinate_frame:
                raise PatientQRSProfileError(
                    f"{ref_name} coordinate frame does not match openCARP workspace"
                )

        mesh_basename = workspace_root / manifest.mesh_basename
        node_count = _mesh_node_count(mesh_basename)
        activation = _load_activation(activation_path, node_count)
        shift_ms = float(experiment.activation_translation.settings["ecg_reference_shift_ms"])
        shifted = activation + shift_ms
        duration_ms = experiment.time_grid.end_ms - experiment.time_grid.start_ms
        if float(np.max(shifted)) >= duration_ms:
            raise PatientQRSProfileError("shifted activation sequence extends beyond simulation window")

        run_root.mkdir(parents=True, exist_ok=True)
        act_igb = run_root / "prescribed_activation.igb"
        _write_activation_igb(act_igb, shifted)

        try:
            from carputils import ep
            from carputils.leadfield import leadfield_setup
        except ImportError as exc:  # pragma: no cover - external runtime
            raise PatientQRSProfileError("carputils is required by this profile") from exc

        executable = str(plan["opencarp_executable"])
        ep_dir = run_root / "ep"
        ep_dir.mkdir(exist_ok=True)
        dt_us = experiment.time_grid.integration_dt_ms * 1000.0
        ep_cmd = [
            executable,
            "-simID",
            str(ep_dir),
            "-meshname",
            str(mesh_basename),
            "-dt",
            f"{dt_us:.12g}",
            "-tend",
            f"{duration_ms:.12g}",
            "-spacedt",
            f"{experiment.time_grid.output_dt_ms:.12g}",
            "-timedt",
            "10",
        ]
        ep_cmd.extend(
            map(
                str,
                ep.fake_ep_opts(
                    stimIdx=0,
                    act_sequence=str(act_igb),
                    pulse_file=True,
                ),
            )
        )
        _run_command(ep_cmd, cwd=run_root)
        vm_file = ep_dir / "vm.igb"
        if not vm_file.is_file():
            raise PatientQRSProfileError("prescribed-activation run did not produce vm.igb")

        electrode_dir = workspace_root / manifest.electrodes_dir
        for name in _REQUIRED_ELECTRODES:
            if not (electrode_dir / f"{name}.vtx").is_file():
                raise PatientQRSProfileError(f"workspace is missing electrode {name}.vtx")
        leadfield_dir = workspace_root / manifest.leadfield_dir
        if not leadfield_dir.is_dir():
            raise PatientQRSProfileError("workspace leadfield directory is missing")

        stim_opts, electrode_opts, run_opts = leadfield_setup(
            12,
            str(electrode_dir),
            post_processing_opts=64,
            ecg_timedt=experiment.time_grid.output_dt_ms,
            lf_dir=str(leadfield_dir),
            lf_vmfile=str(vm_file),
        )
        lf_dir = run_root / "leadfield"
        lf_dir.mkdir(exist_ok=True)
        lf_cmd = [
            executable,
            "-simID",
            str(lf_dir),
            "-meshname",
            str(mesh_basename),
            "-dt",
            f"{dt_us:.12g}",
            "-tend",
            f"{duration_ms:.12g}",
        ]
        lf_cmd.extend(map(str, stim_opts))
        lf_cmd.extend(map(str, electrode_opts))
        lf_cmd.extend(map(str, run_opts))
        _run_command(lf_cmd, cwd=run_root)

        relative_time_ms, ecg_values = _parse_ecg_files(lf_dir, experiment)
        artifact_root = run_root / "artifacts"
        ecg_ref = write_json_artifact(
            artifact_root,
            artifact_id=f"{request.subject_id}-opencarp-recorded-lf12-ecg",
            kind="lead_field_ecg",
            payload={
                "schema_version": "cardiep-lead-field-ecg-v1",
                "subject_id": request.subject_id,
                "lead_names": list(experiment.ecg.expected_leads),
                "sample_rate_hz": experiment.ecg.output_sample_rate_hz,
                "time_ms": relative_time_ms.tolist(),
                "values": ecg_values.tolist(),
                "units": experiment.ecg.output_voltage_unit,
            },
            metadata={
                "lead_names": list(experiment.ecg.expected_leads),
                "sample_rate_hz": experiment.ecg.output_sample_rate_hz,
                "forward_model": "lead_field",
                "units": experiment.ecg.output_voltage_unit,
                "experiment_id": experiment.experiment_id,
                "experiment_version": experiment.experiment_version,
                "experiment_fingerprint": experiment.fingerprint,
                "torso_mode": "patient_specific",
                "electrode_mode": "recorded",
            },
        )
        activation_ref = write_json_artifact(
            artifact_root,
            artifact_id=f"{request.subject_id}-opencarp-prescribed-activation",
            kind="activation_map",
            payload={
                "schema_version": "cardiep-field-v1",
                "subject_id": request.subject_id,
                "units": "ms",
                "activation_ms": activation.tolist(),
                "source": "prescribed_from_fast_calibration",
            },
            metadata={
                "units": "ms",
                "source": "prescribed_from_fast_calibration",
                "independent_propagation_solved": False,
            },
        ).model_copy(update={"coordinate_frame": request.anatomy_ref.coordinate_frame})

        calibration_reference = request.settings["hearttwin_calibration_reference"]
        result = EPSimulationResult(
            subject_id=request.subject_id,
            backend=request.backend,
            parameters=request.parameters,
            outputs=[activation_ref, ecg_ref],
            validation_status="software_checked",
            warnings=[
                "Propagation was prescribed from a calibrated activation field and was not independently re-solved by openCARP.",
                "The lead fields are patient-specific only to the extent represented by the hash-pinned torso and recorded-electrode workspace inputs.",
            ],
            provenance={
                "engine": "openCARP",
                "opencarp_version": _opencarp_version(executable),
                "wrapper_version": WRAPPER_VERSION,
                "driver_version": DRIVER_VERSION,
                "request_sha256": sha256_json(request.model_dump(mode="json")),
                "experiment_id": experiment.experiment_id,
                "experiment_version": experiment.experiment_version,
                "experiment_fingerprint": experiment.fingerprint,
                "workspace_sha256": str(workspace_artifact["sha256"]).lower(),
                "workspace_manifest_sha256": sha256_json(manifest.model_dump(mode="json")),
                "torso_sha256": expected_torso_sha,
                "electrode_sha256": expected_electrode_sha,
                "calibration_reference_sha256": sha256_json(calibration_reference),
                "stimulation_translation_method": (
                    "prescribed_activation_sequence:fast_activation_to_shifted_igb:v1"
                ),
                "calibration_activation_artifact_id": calibration_reference["activation_ref"]["artifact_id"],
                "calibration_activation_sha256": str(
                    calibration_reference["activation_ref"]["sha256"]
                ).lower(),
                "propagation_mode": "prescribed_activation_not_independent_propagation",
                "torso_mode": "patient_specific",
                "electrode_mode": "recorded",
            },
        )
        OpenCARPExternalBackend.validate_result(request, result)
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
