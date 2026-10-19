"""Deterministic runtime staging for the external openCARP confirmation backend.

The package-owned ``cardiep-opencarp-wrapper`` does not guess an openCARP
experiment from a generic patient request.  It validates the existing strict
CardiEP request contract, stages every local artifact into a hash-addressed run
directory, preserves the original request hash, and delegates the
experiment-specific conversion to an explicitly configured driver.

The driver is responsible for translating the staged anatomy, calibrated
activation target, scalar parameter mapping, torso/electrode assumptions, and
lead-field request into a reproducible openCARP/carputils experiment.  The
existing :class:`cardiep.opencarp_backend.OpenCARPExternalBackend` validates the
result after this wrapper returns, including request hash, openCARP/wrapper
versions, standard 12-lead output, activation output, and calibration-reference
lineage.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from .models import ArtifactRef, EPSimulationRequest
from .opencarp_backend import OPENCARP_BACKEND_NAME, OpenCARPExternalBackend
from .provenance import file_sha256, sha256_json, uri_to_path, verify_file_sha256

WRAPPER_VERSION = "cardiep-opencarp-runtime-v1"


class OpenCARPRuntimeError(RuntimeError):
    pass


class OpenCARPRuntimeProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    wrapper_version: str = WRAPPER_VERSION
    opencarp_executable: str | None = None
    driver_executable: str | None = None
    reasons: list[str] = Field(default_factory=list)


class StagedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    artifact_id: str
    kind: str
    original_uri: str
    sha256: str
    staged_path: str
    coordinate_frame: str | None = None


class OpenCARPRuntimePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    wrapper_version: str = WRAPPER_VERSION
    mode: Literal["confirmation"] = "confirmation"
    subject_id: str
    request_sha256: str
    runtime_root: str
    opencarp_executable: str
    staged_artifacts: list[StagedArtifact]
    driver_request_path: str
    driver_result_path: str
    request: dict[str, Any]


def _strip_token_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _command_tokens(env_name: str) -> list[str] | None:
    raw = os.environ.get(env_name)
    if not raw:
        return None
    return [
        _strip_token_quotes(token)
        for token in shlex.split(raw, posix=os.name != "nt")
    ]


def _driver_tokens() -> list[str] | None:
    tokens = _command_tokens("CARDIEP_OPENCARP_DRIVER")
    if not tokens or not {"{plan}", "{output}"}.issubset(tokens):
        return None
    return tokens


def _resolve_executable(value: str) -> str | None:
    resolved = shutil.which(value)
    if resolved:
        return str(Path(resolved).resolve())
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return None


def probe_opencarp_runtime() -> OpenCARPRuntimeProbe:
    reasons: list[str] = []
    open_carp_name = os.environ.get("CARDIEP_OPENCARP_EXECUTABLE", "openCARP")
    open_carp = _resolve_executable(open_carp_name)
    if open_carp is None:
        reasons.append(
            "openCARP executable is unavailable; set CARDIEP_OPENCARP_EXECUTABLE"
        )

    tokens = _driver_tokens()
    driver: str | None = None
    if tokens is None:
        reasons.append(
            "CARDIEP_OPENCARP_DRIVER must contain a command with {plan} and {output} placeholders"
        )
    else:
        driver = _resolve_executable(tokens[0])
        if driver is None:
            reasons.append(
                f"openCARP scientific driver executable is unavailable: {tokens[0]}"
            )

    return OpenCARPRuntimeProbe(
        ready=not reasons,
        opencarp_executable=open_carp,
        driver_executable=driver,
        reasons=reasons,
    )


def _runtime_root(request_sha: str, subject_id: str) -> Path:
    configured = os.environ.get("CARDIEP_RUNTIME_ROOT")
    base = (
        Path(configured).expanduser()
        if configured
        else Path.cwd() / "cardiep_runs" / "runtime"
    )
    safe_subject = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in subject_id
    )
    return (base / "opencarp" / safe_subject / request_sha[:16]).resolve()


def _local_artifact_path(ref: ArtifactRef | dict[str, Any], *, role: str) -> tuple[Path, str]:
    if isinstance(ref, ArtifactRef):
        uri = ref.uri
        digest = ref.sha256
    else:
        uri = str(ref.get("uri") or "")
        digest = ref.get("sha256")
    if not uri or not digest:
        raise OpenCARPRuntimeError(f"{role} requires uri and sha256")
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise OpenCARPRuntimeError(
            f"package-owned openCARP runtime requires local/file artifacts for {role}; "
            f"got scheme {parsed.scheme!r}"
        )
    path = uri_to_path(uri)
    if not path.is_file():
        raise OpenCARPRuntimeError(f"{role} artifact is missing: {path}")
    verify_file_sha256(path, str(digest))
    return path, str(digest).lower()


def _stage_artifact(
    ref: ArtifactRef | dict[str, Any],
    *,
    role: str,
    destination_root: Path,
) -> StagedArtifact:
    source, digest = _local_artifact_path(ref, role=role)
    if isinstance(ref, ArtifactRef):
        artifact_id = ref.artifact_id
        kind = ref.kind
        uri = ref.uri
        frame = ref.coordinate_frame
    else:
        artifact_id = str(ref.get("artifact_id") or "")
        kind = str(ref.get("kind") or "")
        uri = str(ref.get("uri") or "")
        frame = ref.get("coordinate_frame")
    if not artifact_id or not kind:
        raise OpenCARPRuntimeError(f"{role} requires artifact_id and kind")

    destination_root.mkdir(parents=True, exist_ok=True)
    suffix = "".join(source.suffixes) or ".bin"
    destination = (destination_root / f"{digest[:16]}-{role}{suffix}").resolve()
    if destination.exists():
        if file_sha256(destination) != digest:
            raise OpenCARPRuntimeError(
                f"existing staged {role} artifact fails SHA-256 verification"
            )
    else:
        shutil.copyfile(source, destination)
    if file_sha256(destination) != digest:
        raise OpenCARPRuntimeError(f"staged {role} artifact hash changed during copy")

    return StagedArtifact(
        role=role,
        artifact_id=artifact_id,
        kind=kind,
        original_uri=uri,
        sha256=digest,
        staged_path=destination.as_posix(),
        coordinate_frame=None if frame is None else str(frame),
    )


def _request_artifacts(request: EPSimulationRequest) -> list[tuple[str, ArtifactRef | dict[str, Any]]]:
    artifacts: list[tuple[str, ArtifactRef | dict[str, Any]]] = [
        ("anatomy", request.anatomy_ref)
    ]
    settings = request.settings
    if str(settings.get("torso_mode", "")).lower() == "patient_specific":
        artifacts.append(("torso", settings["torso_ref"]))
    if str(settings.get("electrode_mode", "")).lower() == "recorded":
        artifacts.append(("electrodes", settings["electrode_ref"]))
    calibration = settings.get("hearttwin_calibration_reference")
    if isinstance(calibration, dict):
        activation = calibration.get("activation_ref")
        if not isinstance(activation, dict):
            raise OpenCARPRuntimeError(
                "hearttwin_calibration_reference.activation_ref must be an artifact object"
            )
        artifacts.append(("calibration_activation", activation))
    return artifacts


def prepare_opencarp_runtime_plan(
    request: EPSimulationRequest,
    *,
    opencarp_executable: str,
    runtime_root: str | Path | None = None,
) -> OpenCARPRuntimePlan:
    OpenCARPExternalBackend.validate_request(request)
    if request.backend != OPENCARP_BACKEND_NAME:
        raise OpenCARPRuntimeError(
            f"openCARP runtime received backend {request.backend!r}, "
            f"expected {OPENCARP_BACKEND_NAME!r}"
        )

    request_json = request.model_dump(mode="json")
    request_sha = sha256_json(request_json)
    root = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root is not None
        else _runtime_root(request_sha, request.subject_id)
    )
    root.mkdir(parents=True, exist_ok=True)
    staged_root = root / "inputs"
    staged = [
        _stage_artifact(ref, role=role, destination_root=staged_root)
        for role, ref in _request_artifacts(request)
    ]

    driver_request = root / "driver-request.json"
    driver_result = root / "driver-result.json"
    plan_payload = {
        "wrapper_version": WRAPPER_VERSION,
        "subject_id": request.subject_id,
        "request_sha256": request_sha,
        "opencarp_executable": str(Path(opencarp_executable).resolve()),
        "staged_artifacts": [item.model_dump(mode="json") for item in staged],
        "request": request_json,
        "expected_result_path": driver_result.as_posix(),
    }
    driver_request.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return OpenCARPRuntimePlan(
        subject_id=request.subject_id,
        request_sha256=request_sha,
        runtime_root=root.as_posix(),
        opencarp_executable=str(Path(opencarp_executable).resolve()),
        staged_artifacts=staged,
        driver_request_path=driver_request.as_posix(),
        driver_result_path=driver_result.as_posix(),
        request=request_json,
    )


def _invoke_driver(plan: OpenCARPRuntimePlan, output_path: Path) -> None:
    tokens = _driver_tokens()
    if tokens is None:
        raise OpenCARPRuntimeError(
            "CARDIEP_OPENCARP_DRIVER must contain {plan} and {output} placeholders"
        )
    argv = [
        token.replace("{plan}", plan.driver_request_path).replace(
            "{output}", str(output_path)
        )
        for token in tokens
    ]
    process = subprocess.run(
        argv,
        cwd=plan.runtime_root,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CARDIEP_OPENCARP_EXECUTABLE": plan.opencarp_executable,
        },
    )
    if process.returncode:
        raise OpenCARPRuntimeError(
            process.stderr.strip()
            or process.stdout.strip()
            or f"openCARP scientific driver exited with {process.returncode}"
        )
    if not output_path.is_file():
        raise OpenCARPRuntimeError(
            "openCARP scientific driver did not create the requested result JSON"
        )
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenCARPRuntimeError(
            "openCARP scientific driver result is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise OpenCARPRuntimeError(
            "openCARP scientific driver result must be a JSON object"
        )


def run_opencarp_wrapper(request_path: str | Path, output_path: str | Path) -> None:
    probe = probe_opencarp_runtime()
    if not probe.ready or probe.opencarp_executable is None:
        raise OpenCARPRuntimeError(
            "openCARP runtime is not ready: " + "; ".join(probe.reasons)
        )
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OpenCARPRuntimeError("openCARP wrapper request must be a JSON object")
    request = EPSimulationRequest.model_validate(payload)
    plan = prepare_opencarp_runtime_plan(
        request,
        opencarp_executable=probe.opencarp_executable,
    )
    _invoke_driver(plan, Path(output_path).expanduser().resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CardiEP openCARP external runtime wrapper"
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--request")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    if args.probe:
        result = probe_opencarp_runtime()
        print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return 0 if result.ready else 1
    if not args.request or not args.output:
        parser.error("--request and --output are required unless --probe is used")
    try:
        run_opencarp_wrapper(args.request, args.output)
    except (OpenCARPRuntimeError, OSError, TypeError, ValueError) as exc:
        print(
            f"cardiep-opencarp-wrapper: {type(exc).__name__}: {exc}",
            file=os.sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
