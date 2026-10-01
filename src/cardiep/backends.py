from __future__ import annotations

from typing import Protocol

from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPSimulationRequest,
    EPSimulationResult,
)


class EPBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult: ...

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult: ...


class BackendUnavailable(RuntimeError):
    pass
