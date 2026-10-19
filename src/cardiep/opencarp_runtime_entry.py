"""Strict package entrypoint for declared openCARP patient experiments.

Third-party ``CARDIEP_OPENCARP_WRAPPER`` implementations remain free to own
their own experiment contract.  The package-owned wrapper is stricter: every
simulation request must include a complete ``opencarp_experiment_spec`` and that
spec must agree with the immutable CardiEP request before any external driver is
invoked.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .models import EPSimulationRequest
from .opencarp_experiment import OpenCARPExperimentSpec, validate_experiment_request
from .opencarp_runtime import (
    OpenCARPRuntimeError,
    probe_opencarp_runtime,
    run_opencarp_wrapper,
)


def declared_experiment(request: EPSimulationRequest) -> OpenCARPExperimentSpec:
    raw = request.settings.get("opencarp_experiment_spec")
    if not isinstance(raw, dict):
        raise OpenCARPRuntimeError(
            "package-owned openCARP runtime requires settings.opencarp_experiment_spec"
        )
    try:
        return validate_experiment_request(request, raw)
    except (TypeError, ValueError, ValidationError) as exc:
        raise OpenCARPRuntimeError(f"invalid openCARP experiment spec: {exc}") from exc


def validate_declared_request_file(request_path: str | Path) -> OpenCARPExperimentSpec:
    path = Path(request_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenCARPRuntimeError("openCARP request file is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise OpenCARPRuntimeError("openCARP request must be a JSON object")
    try:
        request = EPSimulationRequest.model_validate(payload)
    except ValidationError as exc:
        raise OpenCARPRuntimeError(f"invalid CardiEP simulation request: {exc}") from exc
    return declared_experiment(request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CardiEP declared openCARP patient runtime wrapper"
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
        spec = validate_declared_request_file(args.request)
        os.environ["CARDIEP_OPENCARP_EXPERIMENT_ID"] = spec.experiment_id
        os.environ["CARDIEP_OPENCARP_EXPERIMENT_VERSION"] = spec.experiment_version
        os.environ["CARDIEP_OPENCARP_EXPERIMENT_FINGERPRINT"] = spec.fingerprint
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
