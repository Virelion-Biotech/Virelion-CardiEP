from __future__ import annotations

import pytest
from pydantic import ValidationError

from cardiep.models import ArtifactRef, EPParameterSet, EPSimulationRequest
from cardiep.opencarp_backend import OPENCARP_BACKEND_NAME
from cardiep.opencarp_driver_registry import OpenCARPExperimentDriverRegistry
from cardiep.opencarp_experiment import OpenCARPExperimentSpec, validate_experiment_request
from cardiep.opencarp_runtime import OpenCARPRuntimeError
from cardiep.opencarp_runtime_entry import declared_experiment

LEADS = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def experiment_payload() -> dict:
    return {
        "contract_version": "1.0",
        "experiment_id": "patient-ventricular-qrs-confirmation",
        "experiment_version": "1.0.0",
        "intended_use": (
            "High-fidelity research confirmation of one calibrated ventricular "
            "activation and QRS model."
        ),
        "field_model": "monodomain",
        "activation_model": "reaction_eikonal_plus",
        "tissue_regions": [
            {
                "region_id": "ventricular_myocardium",
                "element_tags": [1],
                "ionic_model": "explicit-study-ionic-model",
                "ionic_parameters": {"parameter_scale": 1.0},
                "conduction_velocity_m_per_s": [0.6, 0.25, 0.2],
                "intracellular_conductivity_s_per_m": [0.2, 0.08, 0.06],
            }
        ],
        "fibre_field_required": True,
        "parameter_bindings": [
            {
                "parameter_name": "bulk_conductivity",
                "parameter_unit": "S/m",
                "target_path": (
                    "tissue_regions.ventricular_myocardium."
                    "intracellular_conductivity_s_per_m.0"
                ),
                "rationale": "Bind the mapped longitudinal conductivity used by confirmation.",
            }
        ],
        "activation_translation": {
            "strategy_id": "declared-stimulation-translation",
            "strategy_version": "1.0",
            "settings": {"max_sites": 32, "temporal_tolerance_ms": 1.0},
            "rationale": (
                "Translate the hash-pinned calibrated activation target into explicit stimuli."
            ),
        },
        "time_grid": {
            "start_ms": -100.0,
            "end_ms": 500.0,
            "integration_dt_ms": 0.02,
            "output_dt_ms": 2.0,
        },
        "ecg": {
            "forward_model": "lead_field",
            "torso_mode": "standardized",
            "electrode_mode": "protocol_derived",
            "expected_leads": LEADS,
            "output_voltage_unit": "mV",
            "output_sample_rate_hz": 500.0,
            "time_axis_start_ms": -100.0,
            "time_axis_end_ms": 500.0,
            "simulation_zero_relative_to_ecg_reference_ms": 0.0,
            "volume_conductor_regions": [
                {
                    "region_id": "torso_bulk",
                    "element_tags": [100],
                    "conductivity_s_per_m": 0.2,
                }
            ],
        },
        "solver_options": {"linear_solver": "explicit-study-choice"},
        "provenance_reference": "Protocol PATIENT-QRS-CONFIRMATION-001",
    }


def simulation_request(payload: dict | None = None) -> EPSimulationRequest:
    settings = {
        "ecg_forward_model": "lead_field",
        "torso_mode": "standardized",
        "electrode_mode": "protocol_derived",
        "expected_leads": LEADS,
        "hearttwin_calibration_reference": {
            "activation_ref": {
                "artifact_id": "activation",
                "kind": "activation_map",
                "uri": "file:///activation.json",
                "sha256": "d" * 64,
                "coordinate_frame": "model",
            }
        },
    }
    if payload is not None:
        settings["opencarp_experiment_spec"] = payload
    return EPSimulationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="anatomy",
            kind="volume_mesh",
            uri="file:///anatomy.vtu",
            sha256="a" * 64,
            coordinate_frame="model",
            metadata={"bundle_fingerprint": "b" * 64},
        ),
        backend=OPENCARP_BACKEND_NAME,
        parameters=EPParameterSet(
            values={"bulk_conductivity": 0.2},
            units={"bulk_conductivity": "S/m"},
            source="calibrated",
        ),
        observations=[],
        settings=settings,
    )


def test_registry_imports_against_restored_contract() -> None:
    registry = OpenCARPExperimentDriverRegistry()
    assert registry.status() == []


def test_declared_experiment_binds_request_and_fingerprints() -> None:
    payload = experiment_payload()
    request = simulation_request(payload)
    selected = validate_experiment_request(request, payload)
    assert selected.experiment_id == "patient-ventricular-qrs-confirmation"
    assert len(selected.fingerprint) == 64
    assert declared_experiment(request).fingerprint == selected.fingerprint


def test_declared_entrypoint_rejects_missing_spec() -> None:
    with pytest.raises(OpenCARPRuntimeError, match="opencarp_experiment_spec"):
        declared_experiment(simulation_request())


def test_reaction_eikonal_plus_requires_velocity_and_conductivity() -> None:
    payload = experiment_payload()
    payload["tissue_regions"][0].pop("conduction_velocity_m_per_s")
    with pytest.raises(ValidationError, match="both velocities and intracellular"):
        OpenCARPExperimentSpec.model_validate(payload)


def test_parameter_units_must_match_request() -> None:
    payload = experiment_payload()
    payload["parameter_bindings"][0]["parameter_unit"] = "mS/cm"
    with pytest.raises(ValueError, match="unit"):
        validate_experiment_request(simulation_request(payload), payload)
