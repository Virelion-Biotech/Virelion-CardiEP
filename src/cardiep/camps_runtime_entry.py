"""Console entrypoint that defaults Camps runtime to the package-qualified driver."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .camps_runtime import main as runtime_main

_QUALIFIED_DRIVER = "cardiep-camps-qualified-driver"


def _configure_default_driver() -> None:
    if os.environ.get("CARDIEP_CAMPS_DRIVER"):
        return
    resolved = shutil.which(_QUALIFIED_DRIVER)
    if resolved is None:
        candidate = Path(os.sys.executable).resolve().parent / _QUALIFIED_DRIVER
        if os.name == "nt":
            candidate = candidate.with_suffix(".exe")
        if candidate.is_file():
            resolved = str(candidate)
    if resolved is not None:
        os.environ["CARDIEP_CAMPS_DRIVER"] = f'"{resolved}" {{plan}} {{output}}'


def main(argv: list[str] | None = None) -> int:
    _configure_default_driver()
    return runtime_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
