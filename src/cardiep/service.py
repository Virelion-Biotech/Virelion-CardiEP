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
    def __init__(
        self,
        backends: list[EPBackend] | None = None,
        *,
        register_defaults: bool = True,
    ) -> None:
        self._backends: dict[str, EPBackend] = {}
        if register_defaults:
            from .registry import discover_backends

            for backend in discover_backends().values():
                self.register_backend(backend)
        for backend in backends or []:
            self.register_backend(backend)

    def register_backend(self, backend: EPBackend) -> None:
        name = str(backend.name).strip()
        if not name:
            raise ValueError("EP backend name must be non-empty")
        self._backends[name] = backend

    def backends(self) -> list[str]:
        return sorted(self._backends)

    def backend_status(self) -> list[dict]:
        from .registry import backend_status

        return backend_status(self._backends)

    def _backend(self, name: str) -> EPBackend:
        backend = self._backends.get(name)
        if backend is None or not backend.available():
            raise BackendUnavailable(f"CardiEP backend unavailable: {name}")
        return backend

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        result = self._backend(request.backend).simulate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned a different subject")
        if result.backend != request.backend:
            raise ReadinessError(
                f"Backend identity mismatch: request={request.backend!r}, result={result.backend!r}"
            )

        self._preserve_anatomy(request, result)
        return result

    @staticmethod
    def _preserve_anatomy(request, result):
        # Preserve the exact anatomy identity at the service boundary so every
        # backend participates in the same HeartTwin lineage contract.
        expected = {"anatomy_artifact_id": request.anatomy_ref.artifact_id}
        if request.anatomy_ref.sha256 is not None:
            expected["anatomy_sha256"] = request.anatomy_ref.sha256
        if request.anatomy_ref.coordinate_frame is not None:
            expected["anatomy_coordinate_frame"] = request.anatomy_ref.coordinate_frame
        fingerprint = request.anatomy_ref.metadata.get("bundle_fingerprint")
        if fingerprint is not None:
            expected["anatomy_bundle_fingerprint"] = str(fingerprint)
        for key, value in expected.items():
            if key in result.provenance and result.provenance[key] != value:
                raise ReadinessError(f"Backend anatomy identity conflict: {key}")
            result.provenance[key] = value

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        result = self._backend(request.backend).calibrate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned inference for a different subject")
        if result.backend != request.backend:
            raise ReadinessError(
                f"Backend identity mismatch: request={request.backend!r}, result={result.backend!r}"
            )
        if result.simulated is not None:
            if result.simulated.subject_id != request.subject_id or result.simulated.backend != request.backend:
                raise ReadinessError("Calibration returned a mismatched nested simulation")
            self._preserve_anatomy(request, result.simulated)
        self._preserve_anatomy(request, result)
        return result
