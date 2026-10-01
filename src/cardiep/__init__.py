"""Public API for Virelion-CardiEP."""

from .conduction import RootSchedule, resolve_root_schedule
from .discrepancy import DiscrepancyReport, DiscrepancyTerm, evaluate_observations
from .ecg import ECGResult, pseudo_ecg
from .external import EXTERNAL_ECOSYSTEM, ExternalBackendDescriptor, SubprocessEPBackend
from .geometry import EPGeometry, load_ep_geometry
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
from .native_backend import NATIVE_BACKEND_NAME, NativeEikonalBackend
from .propagation import PropagationResult, anisotropic_eikonal
from .repolarization import RepolarizationResult, apd_map
from .service import CardiEPService, ReadinessError
from .validation import run_reference_validation, synthetic_tetra_geometry

__all__ = [
    "EXTERNAL_ECOSYSTEM",
    "NATIVE_BACKEND_NAME",
    "ArtifactRef",
    "CardiEPService",
    "DiscrepancyReport",
    "DiscrepancyTerm",
    "ECGResult",
    "EPCalibrationRequest",
    "EPCalibrationResult",
    "EPGeometry",
    "EPObservation",
    "EPParameterSet",
    "EPSimulationRequest",
    "EPSimulationResult",
    "ExternalBackendDescriptor",
    "NativeEikonalBackend",
    "PropagationResult",
    "ReadinessError",
    "RepolarizationResult",
    "RootSchedule",
    "SubprocessEPBackend",
    "anisotropic_eikonal",
    "apd_map",
    "calibration_request_from_electrotrace",
    "evaluate_observations",
    "load_ep_geometry",
    "observations_from_electrotrace",
    "pseudo_ecg",
    "resolve_root_schedule",
    "run_reference_validation",
    "synthetic_tetra_geometry",
]

__version__ = "0.2.0"
