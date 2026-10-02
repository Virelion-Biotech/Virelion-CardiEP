"""Public API for Virelion-CardiEP."""

from .api import EPAPI
from .conduction import RootSchedule, resolve_root_schedule
from .configuration import validate_native_configuration
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
from .scientific_validation import (
    ActivationProfile,
    AgreementThresholds,
    activation_profile_from_csv,
    ConvergenceLevel,
    compare_activation_profiles,
    fenicsx_beat_niederer_reference,
    load_convergence_manifest,
    mesh_convergence_report,
    niederer_2011_spec,
)
from .validation import run_reference_validation, synthetic_tetra_geometry

__all__ = [
    "activation_profile_from_csv",
    "ActivationProfile",
    "AgreementThresholds",
    "anisotropic_eikonal",
    "apd_map",
    "ArtifactRef",
    "calibration_request_from_electrotrace",
    "CardiEPService",
    "compare_activation_profiles",
    "ConvergenceLevel",
    "DiscrepancyReport",
    "DiscrepancyTerm",
    "ECGResult",
    "EPAPI",
    "EPCalibrationRequest",
    "EPCalibrationResult",
    "EPGeometry",
    "EPObservation",
    "EPParameterSet",
    "EPSimulationRequest",
    "EPSimulationResult",
    "evaluate_observations",
    "EXTERNAL_ECOSYSTEM",
    "ExternalBackendDescriptor",
    "fenicsx_beat_niederer_reference",
    "load_convergence_manifest",
    "load_ep_geometry",
    "mesh_convergence_report",
    "NATIVE_BACKEND_NAME",
    "NativeEikonalBackend",
    "niederer_2011_spec",
    "observations_from_electrotrace",
    "PropagationResult",
    "pseudo_ecg",
    "ReadinessError",
    "RepolarizationResult",
    "resolve_root_schedule",
    "RootSchedule",
    "run_reference_validation",
    "SubprocessEPBackend",
    "synthetic_tetra_geometry",
    "validate_native_configuration",
]

__version__ = "0.2.0"
