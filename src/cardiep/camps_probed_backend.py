"""Readiness-aware wrapper around the packaged Camps external backend.

Installing CardiEP may install the ``cardiep-camps-cdt-wrapper`` console script,
but that alone must not make the scientific backend look operational.  The
packaged wrapper also requires a pinned upstream checkout and a qualified
scientific driver.  This subclass therefore probes the packaged wrapper before
reporting availability.

Custom external wrappers retain the historical executable-presence behaviour;
they are assumed to own their own readiness contract unless they deliberately
use the packaged wrapper command name.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .camps_cdt_backend import CampsCardiacDigitalTwinExternalBackend

_PACKAGED_WRAPPER = "cardiep-camps-cdt-wrapper"


class ProbedCampsCardiacDigitalTwinExternalBackend(
    CampsCardiacDigitalTwinExternalBackend
):
    """Camps backend whose packaged wrapper must pass ``--probe``."""

    def available(self) -> bool:
        if not super().available():
            return False

        executable = str(self.executable or "")
        if not executable:
            return False
        executable_name = Path(executable).name
        configured = os.environ.get("CARDIEP_CAMPS_CDT_WRAPPER")

        # A deliberately configured third-party wrapper is not required to
        # implement CardiEP's package-owned probe protocol. The package-owned
        # command, including an explicit env override back to that command,
        # must prove its upstream/driver readiness.
        should_probe = executable_name == _PACKAGED_WRAPPER and (
            configured is None
            or Path(str(self.command[0])).name == _PACKAGED_WRAPPER
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
