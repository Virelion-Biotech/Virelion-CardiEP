from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import EPAPI
from .scientific_validation import (
    ActivationProfile,
    AgreementThresholds,
    activation_profile_from_csv,
    compare_activation_profiles,
    fenicsx_beat_niederer_reference,
    load_convergence_manifest,
    mesh_convergence_report,
    niederer_2011_spec,
)


def _load_payload(path: str) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Request JSON must contain an object")
    return raw


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


def _write_or_print(value: dict, output: str | None) -> None:
    if output is None:
        _print(value)
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cardiep",
        description="Patient-specific cardiac electrophysiology engine and backend gateway",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report package and backend availability")
    sub.add_parser("backends", help="List discovered EP backends")
    sub.add_parser("ecosystem", help="List supported external EP ecosystem targets")
    sub.add_parser(
        "validate-reference",
        help="Run deterministic reference checks for the native fast EP engine",
    )

    spec = sub.add_parser(
        "niederer-spec",
        help="Emit the canonical Niederer 2011 cross-solver benchmark specification",
    )
    spec.add_argument("--output")

    fixture = sub.add_parser(
        "niederer-fenicsx-reference",
        help="Emit the pinned FEniCSx-beat Niederer interoperability profile",
    )
    fixture.add_argument("--output")

    compare = sub.add_parser(
        "compare-activation",
        help="Compare two canonical activation profiles",
    )
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument(
        "--alignment",
        choices=["absolute", "p1-relative", "minimum-relative"],
        default="absolute",
    )
    compare.add_argument("--rmse-ms-max", type=float)
    compare.add_argument("--max-abs-ms-max", type=float)
    compare.add_argument("--correlation-min", type=float)
    compare.add_argument("--abs-bias-ms-max", type=float)
    compare.add_argument("--point-tolerance-cm", type=float, default=1e-6)
    compare.add_argument("--output")


    profile_csv = sub.add_parser(
        "profile-from-csv",
        help="Convert an external solver activation CSV into the canonical profile contract",
    )
    profile_csv.add_argument("input")
    profile_csv.add_argument("output")
    profile_csv.add_argument("--benchmark-id", required=True)
    profile_csv.add_argument("--solver-name", required=True)
    profile_csv.add_argument("--solver-version")
    profile_csv.add_argument("--solver-commit")
    profile_csv.add_argument("--equation")
    profile_csv.add_argument("--ionic-model")
    profile_csv.add_argument("--coordinate-unit", default="cm")
    profile_csv.add_argument("--time-unit", default="ms")
    profile_csv.add_argument("--id-column", default="id")
    profile_csv.add_argument("--x-column", default="x")
    profile_csv.add_argument("--y-column", default="y")
    profile_csv.add_argument("--z-column", default="z")
    profile_csv.add_argument("--activation-column", default="activation")

    convergence = sub.add_parser(
        "validate-convergence",
        help="Evaluate a cardiep-convergence-manifest-v1 mesh-convergence study",
    )
    convergence.add_argument("manifest")
    convergence.add_argument("--output")

    simulate = sub.add_parser("simulate", help="Run a typed EPSimulationRequest JSON")
    simulate.add_argument("request")
    calibrate = sub.add_parser("calibrate", help="Run a typed EPCalibrationRequest JSON")
    calibrate.add_argument("request")
    args = parser.parse_args(argv)
    api = EPAPI()

    if args.command == "doctor":
        _print(api.health())
        return 0
    if args.command == "backends":
        _print(api.backends())
        return 0
    if args.command == "ecosystem":
        _print(api.ecosystem())
        return 0
    if args.command == "validate-reference":
        result = api.validate_reference()
        _print(result)
        return 0 if result["passed"] else 2
    if args.command == "niederer-spec":
        _write_or_print(niederer_2011_spec(), args.output)
        return 0
    if args.command == "niederer-fenicsx-reference":
        _write_or_print(fenicsx_beat_niederer_reference().to_dict(), args.output)
        return 0
    if args.command == "compare-activation":
        threshold_values = [
            args.rmse_ms_max,
            args.max_abs_ms_max,
            args.correlation_min,
            args.abs_bias_ms_max,
        ]
        if any(value is not None for value in threshold_values) and not all(
            value is not None for value in threshold_values
        ):
            parser.error(
                "Provide all four agreement thresholds or none: "
                "--rmse-ms-max, --max-abs-ms-max, --correlation-min, --abs-bias-ms-max"
            )
        thresholds = (
            None
            if all(value is None for value in threshold_values)
            else AgreementThresholds(
                rmse_ms_max=args.rmse_ms_max,
                max_abs_ms_max=args.max_abs_ms_max,
                correlation_min=args.correlation_min,
                abs_bias_ms_max=args.abs_bias_ms_max,
            )
        )
        report = compare_activation_profiles(
            ActivationProfile.from_json(args.reference),
            ActivationProfile.from_json(args.candidate),
            alignment=args.alignment,
            thresholds=thresholds,
            point_tolerance_cm=args.point_tolerance_cm,
        )
        _write_or_print(report, args.output)
        return 2 if report["status"] == "fail" else 0

    if args.command == "profile-from-csv":
        profile = activation_profile_from_csv(
            args.input,
            benchmark_id=args.benchmark_id,
            solver_name=args.solver_name,
            solver_version=args.solver_version,
            solver_commit=args.solver_commit,
            equation=args.equation,
            ionic_model=args.ionic_model,
            coordinate_unit=args.coordinate_unit,
            time_unit=args.time_unit,
            id_column=args.id_column,
            x_column=args.x_column,
            y_column=args.y_column,
            z_column=args.z_column,
            activation_column=args.activation_column,
        )
        _write_or_print(profile.to_dict(), args.output)
        return 0
    if args.command == "validate-convergence":
        levels, exact = load_convergence_manifest(args.manifest)
        report = mesh_convergence_report(levels, exact=exact)
        _write_or_print(report, args.output)
        return 0
    if args.command == "simulate":
        _print(api.simulate(_load_payload(args.request)))
        return 0
    if args.command == "calibrate":
        _print(api.calibrate(_load_payload(args.request)))
        return 0
    return 2
