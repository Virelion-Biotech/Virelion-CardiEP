"""Reproducible runtime staging for the external Camps QRS calibration backend.

This module provides the package-owned ``cardiep-camps-cdt-wrapper`` boundary.
It does not reimplement or patch the upstream Cardiac-Digital-Twin science.
Instead it:

* verifies that a local git checkout contains the exact request-pinned commit;
* creates an isolated detached worktree for the run;
* materializes the hash-pinned Camps geometry workspace;
* converts ElectroTrace median-beat ECG artifacts into the audited upstream
  eight-independent-lead CSV layout, with explicit polyphase resampling to
  1000 Hz when requested;
* writes the upstream path mapping inside the isolated worktree; and
* invokes a separately configured scientific driver which must produce the
  complete CardiEP calibration result contract.

The extra scientific driver is intentional.  The audited upstream
``personalise_to_QRS.py`` is a research script with a fixed subject alias and it
saves a posterior population without a per-particle discrepancy or a complete
representative simulation artifact.  Selecting an arbitrary row as a MAP/best
patient would be scientifically unjustified.  A qualified driver must therefore
run the pinned workflow and explicitly export the posterior, representative
activation map, representative ECG, parameter set, diagnostics, and provenance
required by :mod:`cardiep.camps_cdt_backend`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .camps_cdt_backend import (
    CAMPS_CDT_BACKEND_NAME,
    UPSTREAM_QRS_SAMPLE_RATE_HZ,
    CampsCardiacDigitalTwinExternalBackend,
)
from .camps_staging import CampsStagingResult, materialize_camps_geometry_workspace
from .models import EPCalibrationRequest
from .provenance import file_sha256, sha256_json, uri_to_path, verify_file_sha256

UPSTREAM_ALIAS = "UKB_1097970"
UPSTREAM_ECG_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
WRAPPER_VERSION = "cardiep-camps-runtime-v1"


class CampsRuntimeError(RuntimeError):
    pass


class CampsRuntimeProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    wrapper_version: str = WRAPPER_VERSION
    upstream_root: str | None = None
    driver_executable: str | None = None
    reasons: list[str] = Field(default_factory=list)


class CampsECGStaging(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    source_artifact_id: str
    source_sha256: str
    source_sample_rate_hz: float
    target_sample_rate_hz: float
    resampling_method: str
    upstream_leads: list[str]
    staged_path: str
    staged_sha256: str
    n_samples: int


class CampsRuntimePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    wrapper_version: str = WRAPPER_VERSION
    mode: Literal["calibration"] = "calibration"
    subject_id: str
    upstream_alias: str
    upstream_commit: str
    request_sha256: str
    workspace_manifest_fingerprint: str
    runtime_root: str
    upstream_worktree: str
    path_mapping_file: str
    geometry: CampsStagingResult
    ecg: CampsECGStaging
    driver_request_path: str
    driver_result_path: str
    request: dict[str, Any]


def _driver_tokens() -> list[str] | None:
    raw = os.environ.get("CARDIEP_CAMPS_DRIVER")
    if not raw:
        return None
    tokens = shlex.split(raw)
    if not tokens:
        return None
    required = {"{plan}", "{output}"}
    if not required.issubset(tokens):
        return None
    return tokens


def probe_camps_runtime() -> CampsRuntimeProbe:
    reasons: list[str] = []
    upstream_raw = os.environ.get("CARDIEP_CAMPS_UPSTREAM_ROOT")
    upstream_root: Path | None = None
    if not upstream_raw:
        reasons.append("CARDIEP_CAMPS_UPSTREAM_ROOT is not set")
    else:
        upstream_root = Path(upstream_raw).expanduser().resolve()
        if not (upstream_root / ".git").exists():
            reasons.append("CARDIEP_CAMPS_UPSTREAM_ROOT is not a git checkout")
        if not (upstream_root / "src" / "personalise_to_QRS.py").is_file():
            reasons.append("pinned Camps personalise_to_QRS.py is missing")

    tokens = _driver_tokens()
    executable: str | None = None
    if tokens is None:
        reasons.append(
            "CARDIEP_CAMPS_DRIVER must contain a command with {plan} and {output} placeholders"
        )
    else:
        executable = tokens[0]
        if shutil.which(executable) is None and not Path(executable).expanduser().is_file():
            reasons.append(f"Camps scientific driver executable is unavailable: {executable}")

    return CampsRuntimeProbe(
        ready=not reasons,
        upstream_root=None if upstream_root is None else upstream_root.as_posix(),
        driver_executable=executable,
        reasons=reasons,
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=None if cwd is None else str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise CampsRuntimeError(
            process.stderr.strip()
            or process.stdout.strip()
            or f"git {' '.join(args)} exited with {process.returncode}"
        )
    return process.stdout.strip()


def _verify_upstream_commit(upstream_root: Path, commit: str) -> None:
    if not shutil.which("git"):
        raise CampsRuntimeError("git is required for pinned Camps runtime worktrees")
    try:
        resolved = _git("rev-parse", f"{commit}^{{commit}}", cwd=upstream_root)
    except CampsRuntimeError as exc:
        raise CampsRuntimeError(
            f"Pinned upstream commit {commit} is not present in CARDIEP_CAMPS_UPSTREAM_ROOT"
        ) from exc
    if resolved.lower() != commit.lower():
        raise CampsRuntimeError("git resolved a different upstream commit")


def _materialize_worktree(upstream_root: Path, commit: str, destination: Path) -> Path:
    _verify_upstream_commit(upstream_root, commit)
    if destination.exists():
        try:
            existing = _git("rev-parse", "HEAD", cwd=destination)
        except CampsRuntimeError as exc:
            raise CampsRuntimeError(
                f"Existing Camps worktree is not a valid git checkout: {destination}"
            ) from exc
        if existing.lower() != commit.lower():
            raise CampsRuntimeError(
                "Existing Camps runtime worktree is pinned to a different commit"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(
        "worktree",
        "add",
        "--detach",
        str(destination),
        commit,
        cwd=upstream_root,
    )
    resolved = _git("rev-parse", "HEAD", cwd=destination)
    if resolved.lower() != commit.lower():
        raise CampsRuntimeError("Created Camps worktree does not match requested commit")
    return destination


def _load_electrotrace_template(observation: Any) -> tuple[dict[str, Any], Path]:
    artifact = observation.artifact
    source = uri_to_path(artifact.uri)
    if not source.is_file():
        raise CampsRuntimeError(f"ElectroTrace ECG artifact is missing: {source}")
    verify_file_sha256(source, artifact.sha256)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampsRuntimeError("ElectroTrace calibration artifact must contain a JSON object")
    if payload.get("schema_version") != "electrotrace-ep-calibration-v1":
        raise CampsRuntimeError("Unsupported ElectroTrace calibration schema")
    if payload.get("representation") != "median_beat_template":
        raise CampsRuntimeError("Camps runtime requires an ElectroTrace median-beat template")
    return payload, source


def _polyphase_resample(signals: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if math.isclose(source_hz, target_hz, rel_tol=0.0, abs_tol=1e-9):
        return signals
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - depends on optional wrapper environment
        raise CampsRuntimeError(
            "SciPy is required for declared polyphase ECG resampling; install virelion-cardiep[wrappers]"
        ) from exc

    ratio = Fraction(target_hz / source_hz).limit_denominator(10000)
    if ratio.numerator <= 0 or ratio.denominator <= 0:
        raise CampsRuntimeError("Invalid ECG resampling ratio")
    resampled = resample_poly(signals, ratio.numerator, ratio.denominator, axis=1)
    if not np.isfinite(resampled).all():
        raise CampsRuntimeError("Polyphase ECG resampling produced non-finite values")
    return np.asarray(resampled, dtype=float)


def _stage_calibration_ecg(
    request: EPCalibrationRequest,
    *,
    data_root: Path,
    alias: str,
) -> CampsECGStaging:
    if len(request.observations) != 1:
        raise CampsRuntimeError(
            "The audited Camps QRS script consumes exactly one clinical ECG; "
            "multi-acquisition calibration requires a separately qualified driver"
        )
    observation = request.observations[0]
    payload, source = _load_electrotrace_template(observation)
    lead_names = [str(item) for item in payload.get("lead_names") or []]
    beat_template = payload.get("beat_template")
    if not isinstance(beat_template, dict):
        raise CampsRuntimeError("ElectroTrace artifact has no beat_template object")
    missing = [lead for lead in UPSTREAM_ECG_LEADS if lead not in lead_names or lead not in beat_template]
    if missing:
        raise CampsRuntimeError(
            "ElectroTrace template is missing Camps independent lead(s): " + ", ".join(missing)
        )

    rows: list[np.ndarray] = []
    lengths: set[int] = set()
    for lead in UPSTREAM_ECG_LEADS:
        values = np.asarray(beat_template[lead], dtype=float).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise CampsRuntimeError(f"ElectroTrace lead {lead} is empty or non-finite")
        rows.append(values)
        lengths.add(int(values.size))
    if len(lengths) != 1:
        raise CampsRuntimeError("ElectroTrace beat-template leads have unequal lengths")
    matrix = np.stack(rows, axis=0)

    source_rate = float(payload.get("sampling_rate_hz", 0.0))
    if not math.isfinite(source_rate) or source_rate <= 0:
        raise CampsRuntimeError("ElectroTrace template has an invalid sampling rate")
    method = str(request.settings.get("ecg_resampling_method", "none")).strip().lower()
    if not math.isclose(
        source_rate,
        UPSTREAM_QRS_SAMPLE_RATE_HZ,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        if method != "polyphase":
            raise CampsRuntimeError(
                "Non-1000-Hz Camps calibration ECG requires ecg_resampling_method='polyphase'"
            )
        matrix = _polyphase_resample(
            matrix,
            source_hz=source_rate,
            target_hz=UPSTREAM_QRS_SAMPLE_RATE_HZ,
        )
    elif method not in {"none", "polyphase"}:
        raise CampsRuntimeError("Unsupported Camps ECG resampling method")

    clinical_dir = (data_root / "clinical_data").resolve()
    clinical_dir.mkdir(parents=True, exist_ok=True)
    destination = clinical_dir / f"{alias}_clinical_qrs_ecg.csv"
    np.savetxt(destination, matrix, delimiter=",", fmt="%.17g")
    return CampsECGStaging(
        observation_id=observation.observation_id,
        source_artifact_id=observation.artifact.artifact_id,
        source_sha256=file_sha256(source),
        source_sample_rate_hz=source_rate,
        target_sample_rate_hz=UPSTREAM_QRS_SAMPLE_RATE_HZ,
        resampling_method=method,
        upstream_leads=list(UPSTREAM_ECG_LEADS),
        staged_path=destination.as_posix(),
        staged_sha256=file_sha256(destination),
        n_samples=int(matrix.shape[1]),
    )


def _runtime_root(request_sha: str, subject_id: str) -> Path:
    configured = os.environ.get("CARDIEP_RUNTIME_ROOT")
    base = Path(configured).expanduser() if configured else Path.cwd() / "cardiep_runs" / "runtime"
    safe_subject = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in subject_id
    )
    return (base / "camps" / safe_subject / request_sha[:16]).resolve()


def prepare_camps_runtime_plan(
    request: EPCalibrationRequest,
    *,
    upstream_root: str | Path,
    runtime_root: str | Path | None = None,
    upstream_alias: str = UPSTREAM_ALIAS,
) -> CampsRuntimePlan:
    upstream_commit, workspace, _rates, _method = (
        CampsCardiacDigitalTwinExternalBackend.validate_calibration_request(request)
    )
    request_json = request.model_dump(mode="json")
    request_sha = sha256_json(request_json)
    root = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root is not None
        else _runtime_root(request_sha, request.subject_id)
    )
    root.mkdir(parents=True, exist_ok=True)

    upstream_path = Path(upstream_root).expanduser().resolve()
    worktree = _materialize_worktree(upstream_path, upstream_commit, root / "upstream")

    alias_manifest = workspace.model_copy(update={"subject_id": upstream_alias})
    geometry = materialize_camps_geometry_workspace(
        alias_manifest,
        staging_root=root / "workspace",
    )
    ecg = _stage_calibration_ecg(
        request,
        data_root=Path(geometry.data_path),
        alias=upstream_alias,
    )

    config_dir = worktree / ".custom_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path_mapping = config_dir / ".your_path_mapping.txt"
    path_mapping.write_text(geometry.path_mapping_json, encoding="utf-8")

    driver_request = root / "driver-request.json"
    driver_result = root / "driver-result.json"
    plan_payload = {
        "subject_id": request.subject_id,
        "upstream_alias": upstream_alias,
        "upstream_commit": upstream_commit,
        "request_sha256": request_sha,
        "workspace_manifest_fingerprint": workspace.fingerprint,
        "upstream_worktree": worktree.as_posix(),
        "upstream_entrypoint": (worktree / "src" / "personalise_to_QRS.py").as_posix(),
        "data_path": geometry.data_path,
        "results_path": geometry.results_path,
        "path_mapping_file": path_mapping.as_posix(),
        "clinical_ecg_path": ecg.staged_path,
        "ecg": ecg.model_dump(mode="json"),
        "request": request_json,
        "expected_result_path": driver_result.as_posix(),
    }
    driver_request.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return CampsRuntimePlan(
        subject_id=request.subject_id,
        upstream_alias=upstream_alias,
        upstream_commit=upstream_commit,
        request_sha256=request_sha,
        workspace_manifest_fingerprint=workspace.fingerprint,
        runtime_root=root.as_posix(),
        upstream_worktree=worktree.as_posix(),
        path_mapping_file=path_mapping.as_posix(),
        geometry=geometry,
        ecg=ecg,
        driver_request_path=driver_request.as_posix(),
        driver_result_path=driver_result.as_posix(),
        request=request_json,
    )


def _invoke_driver(plan: CampsRuntimePlan, output_path: Path) -> None:
    tokens = _driver_tokens()
    if tokens is None:
        raise CampsRuntimeError(
            "CARDIEP_CAMPS_DRIVER must name a qualified command containing {plan} and {output} placeholders"
        )
    argv = [
        token.replace("{plan}", plan.driver_request_path).replace("{output}", str(output_path))
        for token in tokens
    ]
    process = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        cwd=plan.upstream_worktree,
        env={**os.environ, "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg")},
    )
    if process.returncode:
        raise CampsRuntimeError(
            process.stderr.strip()
            or process.stdout.strip()
            or f"Camps scientific driver exited with {process.returncode}"
        )
    if not output_path.is_file():
        raise CampsRuntimeError("Camps scientific driver did not create the requested result JSON")
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CampsRuntimeError("Camps scientific driver result is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise CampsRuntimeError("Camps scientific driver result must be a JSON object")


def run_camps_wrapper(request_path: str | Path, output_path: str | Path) -> None:
    probe = probe_camps_runtime()
    if not probe.ready or probe.upstream_root is None:
        raise CampsRuntimeError("Camps runtime is not ready: " + "; ".join(probe.reasons))

    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampsRuntimeError("Camps wrapper request must be a JSON object")
    if "parameter_bounds" not in payload:
        raise CampsRuntimeError(
            "The stock Camps wrapper currently supports calibration only; "
            "standalone observation-free replay requires a separately qualified driver contract"
        )
    request = EPCalibrationRequest.model_validate(payload)
    if request.backend != CAMPS_CDT_BACKEND_NAME:
        raise CampsRuntimeError(
            f"Camps wrapper received backend {request.backend!r}, expected {CAMPS_CDT_BACKEND_NAME!r}"
        )
    plan = prepare_camps_runtime_plan(
        request,
        upstream_root=probe.upstream_root,
    )
    _invoke_driver(plan, Path(output_path).expanduser().resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CardiEP Camps external runtime wrapper")
    parser.add_argument("--probe", action="store_true", help="check upstream/runtime availability")
    parser.add_argument("--request")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    if args.probe:
        result = probe_camps_runtime()
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return 0 if result.ready else 1
    if not args.request or not args.output:
        parser.error("--request and --output are required unless --probe is used")
    try:
        run_camps_wrapper(args.request, args.output)
    except (CampsRuntimeError, OSError, TypeError, ValueError) as exc:
        print(f"cardiep-camps-cdt-wrapper: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
