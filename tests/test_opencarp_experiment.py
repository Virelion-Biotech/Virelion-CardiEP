from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from cardiep.models import ArtifactRef, EPParameterSet, EPSimulationRequest
from cardiep.opencarp_backend import OPENCARP_BACKEND_NAME
from cardiep.opencarp_experiment import (
    OpenCARPExperimentSpec,
    validate_experiment_request,
)

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def _spec() -> dict:
    return {
        "contract_version": "1.0",
        "experiment_id": "patient-ventricular-qrs-confirmation",
        "experiment_version": "1.0.0",
        "intended_use": "High-fidelity confirmation of one calibrated ventricular activation/QRS model.",
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
                "target_path": "tissue_regions.ventricular_myocardium.intracellular_conductivity_s_per_m.0",
                "rationale": "Bind the explicitly mapped longitudinal conductivity used by confirmation.",
            }
        ],
        "activation_translation": {
            "strategy_id": "declared-stimulation-translation",
            "strategy_version": "1.0",
            "settings": {"max_sites": 32, "temporal_tolerance_ms": 1.0},
            "rationale": "Translate the hash-pinned calibrated activation target into explicit stimuli.",
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
        "solver_options": {
            "linear_solver": "explicit-study-choice",
            "relative_tolerance": 1e-8,
        },
        "provenance_reference": "Protocol PATIENT-QRS-CONFIRMATION-001",
    }


def _request(spec: dict | None = None) -> EPSimulationRequest:
    settings = {
        "ecg_forward_model": "lead_field",
        "torso_mode": "standardized",
        "electrode_mode": "protocol_derived",
        "expected_leads": LEADS,
        "hearttwin_calibration_reference": {
            "role": "activation_target_from_fast_calibration",
            "source_backend": "camps-cdt-external-v1",
            "anatomy_bundle_fingerprint": "b" * 64,
            "parameter_mapping_fingerprint": "c" * 64,
            "activation_ref": {
                "artifact_id": "activation",
                "kind": "activation_map",
                "uri": "file:///activation.json",
                "sha256": "d" * 64,
                "coordinate_frame": "model",
            },
        },
    }
    if spec is not None:
        settings["opencarp_experiment_spec"] = spec
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


def test_valid_experiment_binds_request_and_has_stable_fingerprint() -> None:
    request = _request(_spec())
    selected = validate_experiment_request(request, request.settings["opencarp_experiment_spec"])
    assert selected.field_model == "monodomain"
    assert selected.activation_model == "reaction_eikonal_plus"
    assert selected.fingerprint == OpenCARPExperimentSpec.model_validate(_spec()).fingerprint
    assert len(selected.fingerprint) == 64


def test_reaction_eikonal_plus_requires_velocity_and_conductivity() -> None:
    payload = _spec()
    payload["tissue_regions"][0].pop("conduction_velocity_m_per_s")
    with pytest.raises(ValidationError, match="both velocities and intracellular"):
        OpenCARPExperimentSpec.model_validate(payload)


def test_bidomain_requires_extracellular_conductivity() -> None:
    payload = _spec()
    payload["field_model"] = "bidomain"
    with pytest.raises(ValidationError, match="extracellular conductivity"):
        OpenCARPExperimentSpec.model_validate(payload)


def test_solver_options_cannot_be_implicit() -> None:
    payload = _spec()
    payload["solver_options"] = {}
    with pytest.raises(ValidationError, match="solver_options"):
        OpenCARPExperimentSpec.model_validate(payload)


def test_parameter_binding_must_cover_request_exactly() -> None:
    payload = _spec()
    payload["parameter_bindings"][0]["parameter_name"] = "different_parameter"
    with pytest.raises(ValueError, match="unbound request parameters"):
        validate_experiment_request(_request(payload), payload)


def test_parameter_units_must_match_binding() -> None:
    payload = _spec()
    payload["parameter_bindings"][0]["parameter_unit"] = "mS/cm"
    with pytest.raises(ValueError, match="unit"):
        validate_experiment_request(_request(payload), payload)


def test_ecg_modes_must_match_immutable_request() -> None:
    payload = _spec()
    payload["ecg"]["torso_mode"] = "patient_specific"
    with pytest.raises(ValueError, match="torso_mode"):
        validate_experiment_request(_request(payload), payload)


def test_calibration_activation_reference_is_mandatory() -> None:
    payload = _spec()
    request_payload = _request(payload).model_dump(mode="json")
    request_payload["settings"].pop("hearttwin_calibration_reference")
    request = EPSimulationRequest.model_validate(request_payload)
    with pytest.raises(ValueError, match="calibration activation reference"):
        validate_experiment_request(request, payload)


def test_overlapping_tissue_tags_are_rejected() -> None:
    payload = deepcopy(_spec())
    second = deepcopy(payload["tissue_regions"][0])
    second["region_id"] = "second"
    payload["tissue_regions"].append(second)
    with pytest.raises(ValidationError, match="cannot overlap"):
        OpenCARPExperimentSpec.model_validate(payload)
