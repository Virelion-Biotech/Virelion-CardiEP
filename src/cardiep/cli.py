from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import EPAPI


def _load_payload(path: str) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Request JSON must contain an object")
    return raw


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str))


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
    if args.command == "simulate":
        _print(api.simulate(_load_payload(args.request)))
        return 0
    if args.command == "calibrate":
        _print(api.calibrate(_load_payload(args.request)))
        return 0
    return 2
