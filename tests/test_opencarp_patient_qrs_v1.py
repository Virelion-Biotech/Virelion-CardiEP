from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from cardiep.models import ArtifactRef, EPParameterSet, EPSimulationRequest, EPSimulationResult
from cardiep.opencarp_backend import OPENCARP_BACKEND_NAME, OpenCARPExternalBackend
from cardiep.opencarp_patient_qrs_v1 import (
    EXPERIMENT_ID,
    EXPERIMENT_VERSION,
    PatientQRSPrescribedLF12Driver,
    PatientQRSProfileError,
    _safe_extract_workspace,
    profile_experiment_spec,
)
from cardiep.provenance import file_sha256, sha256_json

LEADS = [
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
]
ELECTRODES = ["RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"]


def _workspace_zip(tmp_path: Path, *, anatomy_sha: str, fingerprint: str) -> Path:
    root = tmp_path / "workspace-source"
    (root / "mesh").mkdir(parents=True)
    (root / "electrodes").mkdir()
    (root / "leadfields").mkdir()
    (root / "mesh" / "heart.pts").write_text(
        "3\n0 0 0\n1 0 0\n0 1 0\n", encoding="utf-8"
    )
    (root / "mesh" / "heart.elem").write_text("1\nTr 0 1 2 1\n", encoding="utf-8")
    for name in ELECTRODES:
        (root / "electrodes" / f"{name}.vtx").write_text("1\nextra\n0\n", encoding="utf-8")
    for lead in LEADS:
        (root / "leadfields" / f"LF_{lead}.igb").write_bytes(f"lf-{lead}".encode())

    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": "cardiep-opencarp-workspace-v1",
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_fingerprint": fingerprint,
        "source_anatomy_sha256": anatomy_sha,
        "coordinate_frame": "model",
        "mesh_basename": "mesh/heart",
        "electrodes_dir": "electrodes",
        "leadfield_dir": "leadfields",
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive = tmp_path / "workspace.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in root.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    return archive


def _activation(tmp_path: Path) -> Path:
    path = tmp_path / "activation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "cardiep-field-v1",
                "subject_id": "P1",
                "units": "ms",
                "activation_ms": [0.0, 15.0, 30.0],
                "selection_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _request(workspace: Path, activation: Path) -> EPSimulationRequest:
    spec = profile_experiment_spec()
    anatomy_sha = "a" * 64
    bundle = "b" * 64
    calibration = {
        "role": "activation_target_from_fast_calibration",
        "source_backend": "camps-cdt-external-v1",
        "anatomy_bundle_fingerprint": bundle,
        "parameter_mapping_fingerprint": "c" * 64,
        "activation_ref": {
            "artifact_id": "activation",
            "kind": "activation_map",
            "uri": activation.as_uri(),
            "sha256": file_sha256(activation),
            "coordinate_frame": "model",
        },
    }
    return EPSimulationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="anatomy",
            kind="volume_mesh",
            uri=(workspace.parent / "anatomy.vtu").as_uri(),
            sha256=anatomy_sha,
            coordinate_frame="model",
            metadata={"bundle_fingerprint": bundle},
        ),
        backend=OPENCARP_BACKEND_NAME,
        parameters=EPParameterSet(values={}, units={}, source="calibrated"),
        observations=[],
        settings={
            "ecg_forward_model": "lead_field",
            "torso_mode": "standardized",
            "electrode_mode": "protocol_derived",
            "expected_leads": LEADS,
            "hearttwin_calibration_reference": calibration,
            "opencarp_experiment_spec": spec.model_dump(mode="json"),
            "opencarp_workspace_ref": {
                "artifact_id": "workspace",
                "kind": "opencarp_workspace",
                "uri": workspace.as_uri(),
                "sha256": file_sha256(workspace),
            },
        },
    )


def test_profile_is_prescribed_activation_and_has_no_scalar_bindings() -> None:
    spec = profile_experiment_spec()
    assert spec.experiment_id == EXPERIMENT_ID
    assert spec.activation_model == "prescribed_activation"
    assert spec.parameter_bindings == []
    assert spec.ecg.output_sample_rate_hz == 1000.0
    assert spec.ecg.time_axis_start_ms == -250.0
    assert spec.ecg.time_axis_end_ms == 450.0
    assert len(spec.fingerprint) == 64


def test_workspace_rejects_untracked_scientific_file(tmp_path: Path) -> None:
    spec = profile_experiment_spec()
    archive = _workspace_zip(tmp_path, anatomy_sha="a" * 64, fingerprint=spec.fingerprint)
    expanded = tmp_path / "expanded"
    manifest = _safe_extract_workspace(archive, expanded)
    assert manifest.experiment_fingerprint == spec.fingerprint

    bad_root = tmp_path / "bad"
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(bad_root)
    (bad_root / "untracked.dat").write_text("bad", encoding="utf-8")
    bad_archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_archive, "w") as handle:
        for path in bad_root.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(bad_root).as_posix())
    with pytest.raises(PatientQRSProfileError, match="inventory"):
        _safe_extract_workspace(bad_archive, tmp_path / "bad-expanded")


def test_full_profile_contract_with_fake_external_runner(tmp_path: Path, monkeypatch) -> None:
    spec = profile_experiment_spec()
    workspace = _workspace_zip(tmp_path, anatomy_sha="a" * 64, fingerprint=spec.fingerprint)
    activation = _activation(tmp_path)
    request = _request(workspace, activation)

    staged_workspace = tmp_path / "staged-workspace.zip"
    staged_workspace.write_bytes(workspace.read_bytes())
    staged_activation = tmp_path / "staged-activation.json"
    staged_activation.write_bytes(activation.read_bytes())
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    plan = {
        "runtime_root": runtime_root.as_posix(),
        "opencarp_executable": "/fake/openCARP",
        "request_sha256": sha256_json(request.model_dump(mode="json")),
        "request": request.model_dump(mode="json"),
        "staged_artifacts": [
            {
                "role": "opencarp_workspace",
                "artifact_id": "workspace",
                "kind": "opencarp_workspace",
                "sha256": file_sha256(staged_workspace),
                "staged_path": staged_workspace.as_posix(),
            },
            {
                "role": "calibration_activation",
                "artifact_id": "activation",
                "kind": "activation_map",
                "sha256": file_sha256(staged_activation),
                "staged_path": staged_activation.as_posix(),
            },
        ],
    }

    ep_module = types.ModuleType("carputils.ep")
    ep_module.fake_ep_opts = lambda **kwargs: ["-fake_ep", str(kwargs["act_sequence"])]
    leadfield_module = types.ModuleType("carputils.leadfield")
    leadfield_module.leadfield_setup = lambda *args, **kwargs: (
        [],
        [],
        ["-experiment", "4"],
    )
    package = types.ModuleType("carputils")
    package.ep = ep_module
    monkeypatch.setitem(sys.modules, "carputils", package)
    monkeypatch.setitem(sys.modules, "carputils.ep", ep_module)
    monkeypatch.setitem(sys.modules, "carputils.leadfield", leadfield_module)

    def fake_write_igb(path: Path, values: np.ndarray) -> None:
        assert values.tolist() == [250.0, 265.0, 280.0]
        path.write_bytes(b"activation-igb")

    def fake_run(argv: list[str], *, cwd: Path) -> None:
        sim_id = Path(argv[argv.index("-simID") + 1])
        sim_id.mkdir(parents=True, exist_ok=True)
        if "-experiment" in argv:
            time = np.arange(0.0, 701.0, 1.0)
            for index, lead in enumerate(LEADS):
                voltage = (index + 1) * np.sin(time / 30.0) * 1e-3
                np.savetxt(sim_id / f"ECG_{lead}.dat", np.column_stack([time, voltage]))
        else:
            (sim_id / "vm.igb").write_bytes(b"vm")

    monkeypatch.setattr("cardiep.opencarp_patient_qrs_v1._write_activation_igb", fake_write_igb)
    monkeypatch.setattr("cardiep.opencarp_patient_qrs_v1._run_command", fake_run)
    monkeypatch.setattr(
        "cardiep.opencarp_patient_qrs_v1._opencarp_version",
        lambda executable: "openCARP-test",
    )

    output = tmp_path / "result.json"
    PatientQRSPrescribedLF12Driver().run(
        plan=plan,
        experiment=spec,
        output_path=str(output),
    )
    result = EPSimulationResult.model_validate_json(output.read_text(encoding="utf-8"))
    OpenCARPExternalBackend.validate_result(request, result)
    assert result.validation_status == "software_checked"
    assert result.provenance["propagation_mode"] == (
        "prescribed_activation_not_independent_propagation"
    )
    assert result.provenance["experiment_fingerprint"] == spec.fingerprint
    assert "not independently re-solved" in result.warnings[0]


def test_profile_rejects_scientific_setting_drift(tmp_path: Path) -> None:
    spec = profile_experiment_spec().model_copy(deep=True)
    changed = spec.model_copy(
        update={
            "solver_options": {
                **spec.solver_options,
                "leadfield_post_processing_opts": 66,
            }
        }
    )
    with pytest.raises(PatientQRSProfileError, match="fingerprint-locked"):
        PatientQRSPrescribedLF12Driver().run(
            plan={"request": {}, "runtime_root": str(tmp_path)},
            experiment=changed,
            output_path=str(tmp_path / "out.json"),
        )
