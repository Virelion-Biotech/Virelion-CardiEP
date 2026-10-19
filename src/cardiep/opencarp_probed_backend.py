"""Readiness-aware package-owned openCARP backend.

Installing the ``cardiep-opencarp-wrapper`` console script must not make
``opencarp-external-v1`` look operational when openCARP itself or the configured
experiment driver is missing.  The package-owned wrapper therefore implements a
``--probe`` handshake.  Deliberately configured third-party wrappers retain the
historical executable-presence behavior.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .opencarp_backend import OpenCARPExternalBackend

_PACKAGED_WRAPPER = "cardiep-opencarp-wrapper"


class ProbedOpenCARPExternalBackend(OpenCARPExternalBackend):
    """openCARP backend whose package-owned wrapper must pass ``--probe``."""

    def available(self) -> bool:
        if not super().available():
            return False
        executable = str(self.executable or "")
        if not executable:
            return False
        configured = os.environ.get("CARDIEP_OPENCARP_WRAPPER")
        should_probe = Path(executable).name == _PACKAGED_WRAPPER and (
            configured is None or Path(str(self.command[0])).name == _PACKAGED_WRAPPER
        )
        if not should_probe:
            return True

        resolved = shutil.which(executable) or (
            str(Path(executable).expanduser().resolve())
            if Path(executable).expanduser().is_file()
            else None
        )
        if resolved is None:
            return False
        try:
            process = subprocess.run(
                [resolved, "--probe"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return process.returncode == 0
