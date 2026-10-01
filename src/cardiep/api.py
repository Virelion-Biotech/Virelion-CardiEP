from __future__ import annotations

from typing import Any

from .models import EPCalibrationRequest, EPSimulationRequest
from .service import CardiEPService


class EPAPI:
    capabilities = ("ep.health", "ep.simulate", "ep.calibrate")

    def __init__(self, service: CardiEPService | None = None) -> None:
        self.service = service or CardiEPService()

    def health(self) -> dict[str, Any]:
        return {
            "service": "CardiEP",
            "status": "ok",
            "backends": self.service.backends(),
            "capabilities": list(self.capabilities),
        }

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = EPSimulationRequest.model_validate(payload)
        return self.service.simulate(request).model_dump(mode="json")

    def calibrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = EPCalibrationRequest.model_validate(payload)
        return self.service.calibrate(request).model_dump(mode="json")
