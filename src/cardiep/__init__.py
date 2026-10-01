"""Public API for Virelion-CardiEP."""

from .handoff import calibration_request_from_electrotrace, observations_from_electrotrace
from .models import (
    ArtifactRef,
    EPCalibrationRequest,
    EPCalibrationResult,
    EPObservation,
    EPParameterSet,
    EPSimulationRequest,
    EPSimulationResult,
)
from .service import CardiEPService, ReadinessError

__all__ = [
    "ArtifactRef",
    "observations_from_electrotrace",
    "calibration_request_from_electrotrace",
    "EPObservation",
    "EPParameterSet",
    "EPSimulationRequest",
    "EPSimulationResult",
    "EPCalibrationRequest",
    "EPCalibrationResult",
    "CardiEPService",
    "ReadinessError",
]

__version__ = "0.1.0"
