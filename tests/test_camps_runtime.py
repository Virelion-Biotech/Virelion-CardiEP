from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cardiep.camps_cdt_backend import CAMPS_CDT_BACKEND_NAME
from cardiep.camps_probed_backend import ProbedCampsCardiacDigitalTwinExternalBackend
from cardiep.camps_runtime import (
    CampsRuntimeError,
    UPSTREAM_ECG_LEADS,
    _materialize_worktree,
    _stage_calibration_ecg,
    probe_camps_runtime,
)
from cardiep.models import ArtifactRef, EPCalibrationRequest, EPObservation

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _electrotrace_artifact(tmp_path: Path, *, sample_rate_hz: float = 1000.0) -> Path:
    n = 64
    payload = {
        "schema_version": "electrotrace-ep-calibration-v1",
        "kind": "ecg",
        "representation": "median_beat_template",
        "sampling_rate_hz": sample_rate_hz,
        "relative_time_s": (np.arange(n) / sample_rate_hz).tolist(),
        "lead_names": STANDARD_12,
        "lead_units": {lead: "mV" for lead in STANDARD_12},
        "beat_template": {
            lead: (np.arange(n, dtype=float) + index * 100.0).tolist()
            for index, lead in enumerate(STANDARD_12)
        },
    }
    path = tmp_path / "ecg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request(tmp_path: Path, *, sample_rate_hz: float = 1000.0) -> EPCalibrationRequest:
    ecg = _electrotrace_artifact(tmp_path, sample_rate_hz=sample_rate_hz)
    observation = EPObservation(
        observation_id="fit",
        kind="ecg",
        artifact=ArtifactRef(
            artifact_id="fit-ecg",
            kind="electrotrace_ecg_calibration",
            uri=ecg.resolve().as_uri(),
            sha256=_sha(ecg),
            metadata={"sample_rate_hz": sample_rate_hz, "lead_names": STANDARD_12},
        ),
        units="mV",
    )
    return EPCalibrationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="anatomy",
            kind="volume_mesh",
            uri="file:///unused.vtu",
            sha256="a" * 64,
            coordinate_frame="model",
            metadata={"bundle_fingerprint": "b" * 64},
        ),
        observations=[observation],
        backend=CAMPS_CDT_BACKEND_NAME,
        parameter_bounds={"sheet_speed": (0.02, 0.08)},
        settings={"ecg_resampling_method": "none"},
    )


def test_stage_ecg_uses_only_upstream_independent_leads(tmp_path: Path) -> None:
    request = _request(tmp_path)
    staged = _stage_calibration_ecg(request, data_root=tmp_path / "data", alias="UKB_1097970")

    matrix = np.loadtxt(staged.staged_path, delimiter=",")
    assert matrix.shape == (8, 64)
    assert staged.upstream_leads == list(UPSTREAM_ECG_LEADS)
    assert staged.source_sample_rate_hz == 1000.0
    assert staged.target_sample_rate_hz == 1000.0
    assert staged.staged_sha256 == _sha(Path(staged.staged_path))

    # Lead order is the audited upstream order, not the source 12-lead order.
    source_indices = [STANDARD_12.index(lead) for lead in UPSTREAM_ECG_LEADS]
    assert np.allclose(matrix[:, 0], np.asarray(source_indices, dtype=float) * 100.0)


def test_stage_ecg_refuses_implicit_resampling(tmp_path: Path) -> None:
    request = _request(tmp_path, sample_rate_hz=500.0)
    with pytest.raises(CampsRuntimeError, match="polyphase"):
        _stage_calibration_ecg(request, data_root=tmp_path / "data", alias="UKB_1097970")


def test_stage_ecg_polyphase_resamples_when_scipy_is_installed(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    request = _request(tmp_path, sample_rate_hz=500.0)
    request.settings["ecg_resampling_method"] = "polyphase"
    staged = _stage_calibration_ecg(request, data_root=tmp_path / "data", alias="UKB_1097970")
    matrix = np.loadtxt(staged.staged_path, delimiter=",")
    assert staged.target_sample_rate_hz == 1000.0
    assert staged.resampling_method == "polyphase"
    assert matrix.shape[1] in {127, 128}


def _git(cwd: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def test_materialize_worktree_is_pinned_to_exact_commit(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "v1")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("v2\n", encoding="utf-8")
    _git(repo, "commit", "-am", "v2")

    destination = tmp_path / "run" / "upstream"
    materialized = _materialize_worktree(repo, first, destination)
    assert materialized == destination
    assert _git(destination, "rev-parse", "HEAD") == first
    assert (destination / "README.md").read_text(encoding="utf-8") == "v1\n"

    with pytest.raises(CampsRuntimeError, match="not present"):
        _materialize_worktree(repo, "f" * 40, tmp_path / "missing")


def test_probe_requires_upstream_and_scientific_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARDIEP_CAMPS_UPSTREAM_ROOT", raising=False)
    monkeypatch.delenv("CARDIEP_CAMPS_DRIVER", raising=False)
    result = probe_camps_runtime()
    assert result.ready is False
    assert any("UPSTREAM_ROOT" in reason for reason in result.reasons)
    assert any("CAMPS_DRIVER" in reason for reason in result.reasons)


def test_probe_accepts_explicit_ready_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"
    (upstream / ".git").mkdir(parents=True)
    (upstream / "src").mkdir()
    (upstream / "src" / "personalise_to_QRS.py").write_text("# fixture\n", encoding="utf-8")
    driver = tmp_path / "driver"
    driver.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    driver.chmod(0o755)

    monkeypatch.setenv("CARDIEP_CAMPS_UPSTREAM_ROOT", str(upstream))
    monkeypatch.setenv("CARDIEP_CAMPS_DRIVER", f"{driver} {{plan}} {{output}}")
    result = probe_camps_runtime()
    assert result.ready is True
    assert result.driver_executable == str(driver)


def test_packaged_backend_uses_probe_for_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ProbedCampsCardiacDigitalTwinExternalBackend()
    monkeypatch.setattr("cardiep.camps_probed_backend.shutil.which", lambda _: "/tmp/wrapper")

    class _Process:
        def __init__(self, returncode: int):
            self.returncode = returncode

    monkeypatch.setattr(
        "cardiep.camps_probed_backend.subprocess.run",
        lambda *args, **kwargs: _Process(1),
    )
    assert backend.available() is False

    monkeypatch.setattr(
        "cardiep.camps_probed_backend.subprocess.run",
        lambda *args, **kwargs: _Process(0),
    )
    assert backend.available() is True


def test_custom_wrapper_retains_executable_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ProbedCampsCardiacDigitalTwinExternalBackend(
        command=["custom-camps-wrapper", "{request}", "{output}"],
        executable="custom-camps-wrapper",
    )
    monkeypatch.setattr("cardiep.camps_probed_backend.shutil.which", lambda _: "/tmp/custom")

    def _unexpected(*args, **kwargs):
        raise AssertionError("custom wrapper must not be forced to implement package probe")

    monkeypatch.setattr("cardiep.camps_probed_backend.subprocess.run", _unexpected)
    assert backend.available() is True
