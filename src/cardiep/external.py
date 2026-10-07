from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPSimulationRequest,
    EPSimulationResult,
)
from .provenance import file_sha256, sha256_json, uri_to_path, verify_file_sha256


@dataclass(frozen=True)
class ExternalBackendDescriptor:
    project: str
    role: str
    integration: str
    license: str
    url: str
    notes: str


EXTERNAL_ECOSYSTEM = (
    ExternalBackendDescriptor(
        project="Cardiac-Digital-Twin",
        role="Eikonal/reaction-Eikonal, ECG personalization and digital-twinning architecture",
        integration="architectural reference; optional plugin candidate",
        license="MIT",
        url="https://github.com/juliacamps/Cardiac-Digital-Twin",
        notes="CardiEP independently implements the modular geometry/conduction/propagation/ECG separation.",
    ),
    ExternalBackendDescriptor(
        project="fenicsx-beat",
        role="Monodomain PDE + ODE splitting on FEniCSx",
        integration="recommended plugin backend for finite-element monodomain studies",
        license="MIT",
        url="https://github.com/finsberg/fenicsx-beat",
        notes="Keep its heavy FEniCSx/MPI dependency outside CardiEP core.",
    ),
    ExternalBackendDescriptor(
        project="MonoAlg3D_C",
        role="High-performance finite-volume monodomain solver with GPU support",
        integration="recommended command/plugin backend for large organ simulations",
        license="MIT",
        url="https://github.com/rsachetto/MonoAlg3D_C",
        notes="Useful for high-throughput organ-scale verification of fast Eikonal twins.",
    ),
    ExternalBackendDescriptor(
        project="openCARP",
        role="Mature monodomain/bidomain, ionic models, lead-field ECG and experiment tooling",
        integration="external academic backend; do not vendor into CardiEP",
        license="academic-use terms",
        url="https://opencarp.org/",
        notes="Integration should remain an explicit user-installed adapter because distribution terms differ.",
    ),
    ExternalBackendDescriptor(
        project="pyCEPS",
        role="CARTO/EnSite EAM import and openCARP conversion",
        integration="measurement preprocessing adapter, not vendored",
        license="GPL-3.0+",
        url="https://github.com/medunigraz/pyCEPS",
        notes="ElectroTrace/CardiEP should consume exported calibration artifacts rather than duplicate vendor parsers.",
    ),
    ExternalBackendDescriptor(
        project="Cobiveco",
        role="Consistent biventricular coordinates",
        integration="CardiAnatomy coordinate-field producer",
        license="Apache-2.0",
        url="https://github.com/KIT-IBT/Cobiveco",
        notes="CardiEP consumes coordinate fields for APD gradients and regional parameterization.",
    ),
)


class SubprocessEPBackend:
    """Adapter for user-installed heavy EP engines through a JSON contract.

    The command is executed without a shell. Tokens may contain {request} and
    {output} placeholders. The external wrapper must write a CardiEP result JSON.
    """

    def __init__(
        self,
        *,
        name: str,
        command: str | list[str],
        executable: str | None = None,
        timeout_s: float = 3600.0,
    ) -> None:
        self.name = name
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.executable = executable or (self.command[0] if self.command else None)
        self.timeout_s = float(timeout_s)
        if not self.command or not self.name.strip() or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("External backend requires a name, command and finite positive timeout")

    def available(self) -> bool:
        return bool(self.executable and shutil.which(self.executable))

    def _invoke(self, request: Any) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError(f"External EP backend executable is unavailable: {self.executable}")
        with tempfile.TemporaryDirectory(prefix="cardiep-") as tmp:
            root = Path(tmp).resolve()
            request_path = root / "request.json"
            output_path = root / "result.json"
            request_path.write_text(
                json.dumps(request.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            argv = [
                token.replace("{request}", str(request_path)).replace("{output}", str(output_path))
                for token in self.command
            ]
            process = subprocess.run(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            if process.returncode:
                raise RuntimeError(
                    process.stderr.strip()
                    or process.stdout.strip()
                    or f"External EP backend exited with {process.returncode}"
                )
            if output_path.is_file():
                result = json.loads(output_path.read_text(encoding="utf-8"))
                return self._persist_artifacts(result, request, root)
            if process.stdout.strip():
                return self._persist_artifacts(json.loads(process.stdout), request, root)
            raise RuntimeError("External EP backend produced no result JSON")

    @staticmethod
    def _persist_artifacts(result, request, temporary_root):
        """Keep wrapper outputs alive after its temporary exchange directory closes."""
        from urllib.parse import urlparse
        temporary_root = Path(temporary_root).resolve()
        configured = request.settings.get("output_dir")
        destination_root = Path(configured).expanduser().resolve() if configured else Path.cwd() / "cardiep_runs" / "external" / sha256_json(request.model_dump(mode="json"))[:16]
        def visit(value):
            if isinstance(value, list):
                return [visit(item) for item in value]
            if not isinstance(value, dict):
                return value
            value = {key: visit(item) for key, item in value.items()}
            if {"artifact_id", "kind", "uri"} <= set(value) and urlparse(value["uri"]).scheme in {"", "file"}:
                source = uri_to_path(value["uri"], relative_to=temporary_root)
                if not source.is_file():
                    raise FileNotFoundError(f"External artifact does not exist: {source}")
                verify_file_sha256(source, value.get("sha256"))
                digest = file_sha256(source)
                if source.is_relative_to(temporary_root):
                    destination_root.mkdir(parents=True, exist_ok=True)
                    destination = destination_root / f"{digest[:16]}-{source.name}"
                    shutil.copyfile(source, destination)
                    value["uri"] = destination.resolve().as_uri()
                else:
                    value["uri"] = source.as_uri()
                value["sha256"] = digest
            return value
        return visit(result)

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        return EPSimulationResult.model_validate(self._invoke(request))

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        return EPCalibrationResult.model_validate(self._invoke(request))
