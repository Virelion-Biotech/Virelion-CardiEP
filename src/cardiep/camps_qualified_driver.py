"""Launcher for the qualified Camps posterior-selection instrumentation hook."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .camps_scientific_driver import (
    AUDITED_UPSTREAM_COMMIT,
    CampsScientificDriverError,
    _instrument_source,
)

_INSTRUMENTED_NAME = "_virelion_personalise_to_QRS.py"


def run_qualified_driver(plan_path: str | Path, output_path: str | Path) -> None:
    plan_file = Path(plan_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise CampsScientificDriverError("Camps runtime plan must be a JSON object")
    if str(plan.get("upstream_commit", "")) != AUDITED_UPSTREAM_COMMIT:
        raise CampsScientificDriverError(
            "qualified driver supports only the audited upstream Camps commit"
        )
    source_path = Path(plan["upstream_entrypoint"]).expanduser().resolve()
    if not source_path.is_file():
        raise CampsScientificDriverError(f"upstream entrypoint is missing: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    instrumented = _instrument_source(
        source,
        plan_path=plan_file,
        output_path=output_file,
    )

    # The pinned upstream script intentionally derives its import/config roots
    # from __file__. Keep the temporary copy in the isolated worktree's src/
    # directory so those semantics are unchanged.
    instrumented_path = source_path.parent / _INSTRUMENTED_NAME
    instrumented_path.write_text(instrumented, encoding="utf-8")
    try:
        process = subprocess.run(
            [
                os.environ.get("CARDIEP_CAMPS_PYTHON", os.sys.executable),
                str(instrumented_path),
            ],
            cwd=str(source_path.parent),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg")},
        )
    finally:
        instrumented_path.unlink(missing_ok=True)

    if process.returncode:
        raise CampsScientificDriverError(
            process.stderr.strip()
            or process.stdout.strip()
            or f"instrumented Camps workflow exited with {process.returncode}"
        )
    if not output_file.is_file():
        raise CampsScientificDriverError("instrumented Camps workflow produced no result JSON")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Qualified Camps QRS posterior-selection driver"
    )
    parser.add_argument("plan")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    try:
        run_qualified_driver(args.plan, args.output)
    except (CampsScientificDriverError, OSError, TypeError, ValueError) as exc:
        print(
            f"cardiep-camps-qualified-driver: {type(exc).__name__}: {exc}",
            file=os.sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
