from __future__ import annotations

from .backends import BackendUnavailable, EPBackend
from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPSimulationRequest,
    EPSimulationResult,
)


class ReadinessError(RuntimeError):
    pass


class CardiEPService:
    def __init__(self) -> None:
        self._backends: dict[str, EPBackend] = {}

    def register_backend(self, backend: EPBackend) -> None:
        self._backends[backend.name] = backend

    def backends(self) -> list[str]:
        return sorted(self._backends)

    def _backend(self, name: str) -> EPBackend:
        backend = self._backends.get(name)
        if backend is None or not backend.available():
            raise BackendUnavailable(f"CardiEP backend unavailable: {name}")
        return backend

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        result = self._backend(request.backend).simulate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned a different subject")
        return result

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        result = self._backend(request.backend).calibrate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned a different subject")
        return result
