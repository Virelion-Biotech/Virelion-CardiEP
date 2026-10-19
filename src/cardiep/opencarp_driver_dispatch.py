"""Dispatch a staged openCARP runtime plan to one registered experiment driver."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .opencarp_driver_registry import (
    OpenCARPExperimentDriverError,
    discover_opencarp_experiment_drivers,
)
from .opencarp_experiment import OpenCARPExperimentSpec


class OpenCARPDriverDispatchError(RuntimeError):
    pass


def _load_plan(plan_path: str | Path) -> tuple[dict, OpenCARPExperimentSpec]:
    path = Path(plan_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenCARPDriverDispatchError("openCARP driver plan is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise OpenCARPDriverDispatchError("openCARP driver plan must be a JSON object")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise OpenCARPDriverDispatchError("openCARP driver plan has no request object")
    settings = request.get("settings")
    if not isinstance(settings, dict):
        raise OpenCARPDriverDispatchError("openCARP request has no settings object")
    raw_experiment = settings.get("opencarp_experiment_spec")
    if not isinstance(raw_experiment, dict):
        raise OpenCARPDriverDispatchError(
            "openCARP driver plan has no declared opencarp_experiment_spec"
        )
    try:
        experiment = OpenCARPExperimentSpec.model_validate(raw_experiment)
    except ValidationError as exc:
        raise OpenCARPDriverDispatchError(f"invalid openCARP experiment spec: {exc}") from exc
    return payload, experiment


def dispatch_registered_experiment(
    plan_path: str | Path,
    output_path: str | Path,
) -> None:
    plan, experiment = _load_plan(plan_path)
    registry = discover_opencarp_experiment_drivers()
    try:
        driver = registry.resolve(experiment)
    except OpenCARPExperimentDriverError as exc:
        raise OpenCARPDriverDispatchError(str(exc)) from exc

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    driver.run(
        plan=plan,
        experiment=experiment,
        output_path=str(output),
    )
    if not output.is_file():
        raise OpenCARPDriverDispatchError(
            "registered openCARP experiment driver returned without creating result JSON"
        )
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenCARPDriverDispatchError(
            "registered openCARP experiment driver produced invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise OpenCARPDriverDispatchError(
            "registered openCARP experiment driver result must be a JSON object"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dispatch one declared openCARP experiment to its registered implementation"
    )
    parser.add_argument("plan")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    try:
        dispatch_registered_experiment(args.plan, args.output)
    except (OpenCARPDriverDispatchError, OSError, TypeError, ValueError) as exc:
        print(
            f"cardiep-opencarp-experiment-driver: {type(exc).__name__}: {exc}",
            file=os.sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
