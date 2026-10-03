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
from .scientific_validation import (
    ActivationProfile,
    AgreementThresholds,
    ConvergenceLevel,
    activation_profile_from_csv,
    compare_activation_profiles,
    fenicsx_beat_niederer_reference,
    load_convergence_manifest,
    mesh_convergence_report,
    niederer_2011_spec,
)
from .service import CardiEPService, ReadinessError
from .surface_backend import SURFACE_BACKEND_NAME, SurfaceEikonalBackend, load_surface_geometry
from .validation import (
    run_eikonal_refinement_validation,
    run_reference_validation,
    synthetic_tetra_geometry,
)

__all__ = [
    "EPAPI",
    "EXTERNAL_ECOSYSTEM",
    "NATIVE_BACKEND_NAME",
    "SURFACE_BACKEND_NAME",
    "ActivationProfile",
    "AgreementThresholds",
    "ArtifactRef",
    "CardiEPService",
    "ConvergenceLevel",
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
    "SurfaceEikonalBackend",
    "activation_profile_from_csv",
    "anisotropic_eikonal",
    "apd_map",
    "calibration_request_from_electrotrace",
    "compare_activation_profiles",
    "evaluate_observations",
    "fenicsx_beat_niederer_reference",
    "load_convergence_manifest",
    "load_ep_geometry",
    "load_surface_geometry",
    "mesh_convergence_report",
    "niederer_2011_spec",
    "observations_from_electrotrace",
    "pseudo_ecg",
    "resolve_root_schedule",
    "run_eikonal_refinement_validation",
    "run_reference_validation",
    "synthetic_tetra_geometry",
    "validate_native_configuration",
]

__version__ = "0.2.0"
