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
from cardiep.opencarp_patient_qrs_v1 import PatientQRSProfileError
from cardiep.opencarp_patient_recorded_qrs_v1 import (
    EXPERIMENT_ID,
    EXPERIMENT_VERSION,
    PatientQRSRecordedLF12Driver,
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


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    anatomy = tmp_path / "anatomy.vtu"
    torso = tmp_path / "torso.json"
    electrodes = tmp_path / "electrodes.json"
    anatomy.write_text("anatomy", encoding="utf-8")
    torso.write_text('{"kind":"patient_torso"}\n', encoding="utf-8")
    electrodes.write_text('{"kind":"recorded_electrodes"}\n', encoding="utf-8")
    return anatomy, torso, electrodes


def _workspace_zip(
    tmp_path: Path,
    *,
    anatomy_sha: str,
    torso_sha: str,
    electrode_sha: str,
    fingerprint: str,
) -> Path:
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
        "schema_version": "cardiep-opencarp-recorded-workspace-v1",
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_fingerprint": fingerprint,
        "source_anatomy_sha256": anatomy_sha,
        "source_torso_sha256": torso_sha,
        "source_electrode_sha256": electrode_sha,
        "coordinate_frame": "case3_model",
        "mesh_basename": "mesh/heart",
        "electrodes_dir": "electrodes",
        "leadfield_dir": "leadfields",
        "leadfield_basis": "patient_specific_torso_and_recorded_electrodes",
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _request(
    workspace: Path,
    activation: Path,
    anatomy: Path,
    torso: Path,
    electrodes: Path,
) -> EPSimulationRequest:
    spec = profile_experiment_spec()
    bundle = "b" * 64
    calibration = {
        "role": "activation_target_from_fast_calibration",
        "source_backend": "numpy-eikonal-v1",
        "anatomy_bundle_fingerprint": bundle,
        "parameter_mapping_fingerprint": "c" * 64,
        "activation_ref": {
            "artifact_id": "activation",
            "kind": "activation_map",
            "uri": activation.as_uri(),
            "sha256": file_sha256(activation),
            "coordinate_frame": "case3_model",
        },
    }
    return EPSimulationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="anatomy",
            kind="volume_mesh",
            uri=anatomy.as_uri(),
            sha256=file_sha256(anatomy),
            coordinate_frame="case3_model",
            metadata={"bundle_fingerprint": bundle},
        ),
        backend=OPENCARP_BACKEND_NAME,
        parameters=EPParameterSet(values={}, units={}, source="calibrated"),
        observations=[],
        settings={
            "ecg_forward_model": "lead_field",
            "torso_mode": "patient_specific",
            "torso_ref": {
                "artifact_id": "torso",
                "kind": "torso_mesh",
                "uri": torso.as_uri(),
                "sha256": file_sha256(torso),
                "coordinate_frame": "case3_model",
            },
            "electrode_mode": "recorded",
            "electrode_ref": {
                "artifact_id": "electrodes",
                "kind": "electrode_coordinates",
                "uri": electrodes.as_uri(),
                "sha256": file_sha256(electrodes),
                "coordinate_frame": "case3_model",
            },
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


def test_profile_declares_patient_specific_recorded_forward_model() -> None:
    spec = profile_experiment_spec()
    assert spec.experiment_id == EXPERIMENT_ID
    assert spec.activation_model == "prescribed_activation"
    assert spec.parameter_bindings == []
    assert spec.ecg.torso_mode == "patient_specific"
    assert spec.ecg.electrode_mode == "recorded"
    assert spec.ecg.expected_leads == LEADS
    assert spec.ecg.output_sample_rate_hz == 1000.0
    assert len(spec.fingerprint) == 64


def test_workspace_hash_binds_patient_torso_and_electrodes(tmp_path: Path) -> None:
    anatomy, torso, electrodes = _write_inputs(tmp_path)
    spec = profile_experiment_spec()
    archive = _workspace_zip(
        tmp_path,
        anatomy_sha=file_sha256(anatomy),
        torso_sha=file_sha256(torso),
        electrode_sha=file_sha256(electrodes),
        fingerprint=spec.fingerprint,
    )
    manifest = _safe_extract_workspace(archive, tmp_path / "expanded")
    assert manifest.source_torso_sha256 == file_sha256(torso)
    assert manifest.source_electrode_sha256 == file_sha256(electrodes)
    assert manifest.leadfield_basis == "patient_specific_torso_and_recorded_electrodes"


def test_full_recorded_profile_contract_with_fake_external_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    anatomy, torso, electrodes = _write_inputs(tmp_path)
    spec = profile_experiment_spec()
    activation = _activation(tmp_path)
    workspace = _workspace_zip(
        tmp_path,
        anatomy_sha=file_sha256(anatomy),
        torso_sha=file_sha256(torso),
        electrode_sha=file_sha256(electrodes),
        fingerprint=spec.fingerprint,
    )
    request = _request(workspace, activation, anatomy, torso, electrodes)

    staged_workspace = tmp_path / "staged-workspace.zip"
    staged_workspace.write_bytes(workspace.read_bytes())
    staged_activation = tmp_path / "staged-activation.json"
    staged_activation.write_bytes(activation.read_bytes())
    staged_torso = tmp_path / "staged-torso.json"
    staged_torso.write_bytes(torso.read_bytes())
    staged_electrodes = tmp_path / "staged-electrodes.json"
    staged_electrodes.write_bytes(electrodes.read_bytes())
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
            {
                "role": "torso",
                "artifact_id": "torso",
                "kind": "torso_mesh",
                "sha256": file_sha256(staged_torso),
                "staged_path": staged_torso.as_posix(),
            },
            {
                "role": "electrodes",
                "artifact_id": "electrodes",
                "kind": "electrode_coordinates",
                "sha256": file_sha256(staged_electrodes),
                "staged_path": staged_electrodes.as_posix(),
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

    monkeypatch.setattr(
        "cardiep.opencarp_patient_recorded_qrs_v1._write_activation_igb",
        fake_write_igb,
    )
    monkeypatch.setattr(
        "cardiep.opencarp_patient_recorded_qrs_v1._run_command",
        fake_run,
    )
    monkeypatch.setattr(
        "cardiep.opencarp_patient_recorded_qrs_v1._opencarp_version",
        lambda executable: "openCARP-test",
    )

    output = tmp_path / "result.json"
    PatientQRSRecordedLF12Driver().run(
        plan=plan,
        experiment=spec,
        output_path=str(output),
    )
    result = EPSimulationResult.model_validate_json(output.read_text(encoding="utf-8"))
    OpenCARPExternalBackend.validate_result(request, result)
    assert result.validation_status == "software_checked"
    assert result.provenance["torso_sha256"] == file_sha256(torso)
    assert result.provenance["electrode_sha256"] == file_sha256(electrodes)
    assert result.provenance["propagation_mode"] == (
        "prescribed_activation_not_independent_propagation"
    )
    assert result.provenance["experiment_fingerprint"] == spec.fingerprint


def test_recorded_profile_rejects_workspace_torso_hash_drift(tmp_path: Path) -> None:
    anatomy, torso, electrodes = _write_inputs(tmp_path)
    spec = profile_experiment_spec()
    activation = _activation(tmp_path)
    workspace = _workspace_zip(
        tmp_path,
        anatomy_sha=file_sha256(anatomy),
        torso_sha="d" * 64,
        electrode_sha=file_sha256(electrodes),
        fingerprint=spec.fingerprint,
    )
    request = _request(workspace, activation, anatomy, torso, electrodes)
    plan = {
        "runtime_root": (tmp_path / "runtime").as_posix(),
        "opencarp_executable": "/fake/openCARP",
        "request": request.model_dump(mode="json"),
        "staged_artifacts": [
            {
                "role": "opencarp_workspace",
                "sha256": file_sha256(workspace),
                "staged_path": workspace.as_posix(),
            },
            {
                "role": "calibration_activation",
                "sha256": file_sha256(activation),
                "staged_path": activation.as_posix(),
            },
            {
                "role": "torso",
                "sha256": file_sha256(torso),
                "staged_path": torso.as_posix(),
            },
            {
                "role": "electrodes",
                "sha256": file_sha256(electrodes),
                "staged_path": electrodes.as_posix(),
            },
        ],
    }
    with pytest.raises(PatientQRSProfileError, match="torso hash"):
        PatientQRSRecordedLF12Driver().run(
            plan=plan,
            experiment=spec,
            output_path=str(tmp_path / "out.json"),
        )
