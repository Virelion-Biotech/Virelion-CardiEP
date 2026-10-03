from __future__ import annotations

from typing import Any

from .external import EXTERNAL_ECOSYSTEM
from .models import EPCalibrationRequest, EPSimulationRequest
from .service import CardiEPService
from .validation import run_reference_validation


class EPAPI:
    capabilities = (
        "ep.health",
        "ep.backends",
        "ep.simulate",
        "ep.calibrate",
        "ep.validate.reference",
        "ep.ecosystem",
    )

    def __init__(self, service: CardiEPService | None = None) -> None:
        self.service = service or CardiEPService()

    def health(self) -> dict[str, Any]:
        statuses = self.service.backend_status()
        return {
            "service": "CardiEP",
            "status": "ok" if any(item["available"] for item in statuses) else "degraded",
            "contract_version": "1.0",
            "backends": statuses,
            "capabilities": list(self.capabilities),
            "scientific_status": "research software; no clinical-device claim",
        }

    def backends(self) -> dict[str, Any]:
        return {"backends": self.service.backend_status()}

    def ecosystem(self) -> dict[str, Any]:
        return {
            "external_projects": [item.__dict__ for item in EXTERNAL_ECOSYSTEM],
            "policy": (
                "Heavy numerical engines remain optional plugins or explicit external "
                "commands. CardiEP does not silently vendor incompatible or restricted code."
            ),
        }

    def validate_reference(self) -> dict[str, Any]:
        return run_reference_validation()

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = EPSimulationRequest.model_validate(payload)
        return self.service.simulate(request).model_dump(mode="json")

    def calibrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = EPCalibrationRequest.model_validate(payload)
        return self.service.calibrate(request).model_dump(mode="json")
