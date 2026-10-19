from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cardiep.models import ArtifactRef, EPParameterSet, EPSimulationRequest
from cardiep.opencarp_backend import OPENCARP_BACKEND_NAME
from cardiep.opencarp_probed_backend import ProbedOpenCARPExternalBackend
from cardiep.opencarp_runtime import (
    OpenCARPRuntimeError,
    prepare_opencarp_runtime_plan,
    probe_opencarp_runtime,
)
from cardiep.provenance import sha256_json

STANDARD_12 = [
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


def _file(tmp_path: Path, name: str, content: bytes) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _request(tmp_path: Path) -> EPSimulationRequest:
    anatomy, anatomy_sha = _file(tmp_path, "heart.vtu", b"anatomy")
    torso, torso_sha = _file(tmp_path, "torso.vtu", b"torso")
    electrodes, electrodes_sha = _file(tmp_path, "electrodes.json", b"electrodes")
    activation, activation_sha = _file(tmp_path, "activation.json", b"activation")
    bundle_fingerprint = "b" * 64
    mapping_fingerprint = "c" * 64

    return EPSimulationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="anatomy",
            kind="volume_mesh",
            uri=anatomy.resolve().as_uri(),
            sha256=anatomy_sha,
            coordinate_frame="model",
            metadata={"bundle_fingerprint": bundle_fingerprint},
        ),
        backend=OPENCARP_BACKEND_NAME,
        parameters=EPParameterSet(
            values={"bulk_conductivity": 1.0},
            units={"bulk_conductivity": "1"},
            source="calibrated",
        ),
        observations=[],
        settings={
            "ecg_forward_model": "lead_field",
            "torso_mode": "patient_specific",
            "torso_ref": {
                "artifact_id": "torso",
                "kind": "torso_mesh",
                "uri": torso.resolve().as_uri(),
                "sha256": torso_sha,
            },
            "electrode_mode": "recorded",
            "electrode_ref": {
                "artifact_id": "electrodes",
                "kind": "electrode_coordinates",
                "uri": electrodes.resolve().as_uri(),
                "sha256": electrodes_sha,
            },
            "expected_leads": STANDARD_12,
            "hearttwin_calibration_reference": {
                "role": "activation_target_from_fast_calibration",
                "source_backend": "camps-cdt-external-v1",
                "anatomy_bundle_fingerprint": bundle_fingerprint,
                "parameter_mapping_fingerprint": mapping_fingerprint,
                "activation_ref": {
                    "artifact_id": "activation",
                    "kind": "activation_map",
                    "uri": activation.resolve().as_uri(),
                    "sha256": activation_sha,
                    "coordinate_frame": "model",
                },
            },
        },
    )


def test_plan_stages_all_patient_artifacts_by_hash(tmp_path: Path) -> None:
    request = _request(tmp_path)
    opencarp = tmp_path / "openCARP"
    opencarp.write_text("fixture", encoding="utf-8")

    plan = prepare_opencarp_runtime_plan(
        request,
        opencarp_executable=str(opencarp),
        runtime_root=tmp_path / "runtime",
    )

    assert plan.request_sha256 == sha256_json(request.model_dump(mode="json"))
    assert {item.role for item in plan.staged_artifacts} == {
        "anatomy",
        "torso",
        "electrodes",
        "calibration_activation",
    }
    for item in plan.staged_artifacts:
        staged = Path(item.staged_path)
        assert staged.is_file()
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == item.sha256
        assert staged.name.startswith(item.sha256[:16])
    assert Path(plan.driver_request_path).is_file()


def test_plan_rejects_tampered_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)
    anatomy_path = Path(request.anatomy_ref.uri.removeprefix("file://"))
    anatomy_path.write_bytes(b"tampered")
    opencarp = tmp_path / "openCARP"
    opencarp.write_text("fixture", encoding="utf-8")

    with pytest.raises((OpenCARPRuntimeError, ValueError), match="SHA-256|hash"):
        prepare_opencarp_runtime_plan(
            request,
            opencarp_executable=str(opencarp),
            runtime_root=tmp_path / "runtime",
        )


def test_plan_rejects_remote_artifact_for_package_runtime(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = request.model_dump(mode="json")
    payload["settings"]["torso_ref"]["uri"] = "s3://bucket/torso.vtu"
    request = EPSimulationRequest.model_validate(payload)
    opencarp = tmp_path / "openCARP"
    opencarp.write_text("fixture", encoding="utf-8")

    with pytest.raises(OpenCARPRuntimeError, match="local/file"):
        prepare_opencarp_runtime_plan(
            request,
            opencarp_executable=str(opencarp),
            runtime_root=tmp_path / "runtime",
        )


def test_probe_requires_open_carp_and_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARDIEP_OPENCARP_EXECUTABLE", raising=False)
    monkeypatch.delenv("CARDIEP_OPENCARP_DRIVER", raising=False)
    monkeypatch.setattr("cardiep.opencarp_runtime.shutil.which", lambda _: None)
    result = probe_opencarp_runtime()
    assert result.ready is False
    assert any("openCARP executable" in reason for reason in result.reasons)
    assert any("OPENCARP_DRIVER" in reason for reason in result.reasons)


def test_probe_accepts_explicit_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opencarp = tmp_path / "openCARP"
    driver = tmp_path / "driver"
    for executable in (opencarp, driver):
        executable.write_text("fixture", encoding="utf-8")
        executable.chmod(0o755)

    monkeypatch.setenv("CARDIEP_OPENCARP_EXECUTABLE", str(opencarp))
    monkeypatch.setenv("CARDIEP_OPENCARP_DRIVER", f'"{driver}" {{plan}} {{output}}')
    result = probe_opencarp_runtime()
    assert result.ready is True
    assert result.opencarp_executable == str(opencarp.resolve())
    assert result.driver_executable == str(driver.resolve())


def test_packaged_backend_uses_probe_for_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ProbedOpenCARPExternalBackend()
    monkeypatch.setattr("cardiep.opencarp_probed_backend.shutil.which", lambda _: "/tmp/wrapper")

    class _Process:
        def __init__(self, returncode: int):
            self.returncode = returncode

    monkeypatch.setattr(
        "cardiep.opencarp_probed_backend.subprocess.run",
        lambda *args, **kwargs: _Process(1),
    )
    assert backend.available() is False

    monkeypatch.setattr(
        "cardiep.opencarp_probed_backend.subprocess.run",
        lambda *args, **kwargs: _Process(0),
    )
    assert backend.available() is True


def test_custom_wrapper_keeps_executable_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ProbedOpenCARPExternalBackend(
        command=["custom-opencarp-wrapper", "{request}", "{output}"],
        executable="custom-opencarp-wrapper",
    )
    monkeypatch.setattr("cardiep.opencarp_probed_backend.shutil.which", lambda _: "/tmp/custom")

    def _unexpected(*args, **kwargs):
        raise AssertionError("custom wrapper must not be forced to implement package probe")

    monkeypatch.setattr("cardiep.opencarp_probed_backend.subprocess.run", _unexpected)
    assert backend.available() is True
