from __future__ import annotations

import json
from pathlib import Path

import pytest

from cardiep.opencarp_driver_dispatch import (
    OpenCARPDriverDispatchError,
    dispatch_registered_experiment,
)
from cardiep.opencarp_driver_registry import (
    OpenCARPExperimentDriverError,
    OpenCARPExperimentDriverRegistry,
)
from cardiep.opencarp_experiment import OpenCARPExperimentSpec

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def _experiment() -> OpenCARPExperimentSpec:
    return OpenCARPExperimentSpec.model_validate(
        {
            "contract_version": "1.0",
            "experiment_id": "patient-qrs-confirmation",
            "experiment_version": "1.0.0",
            "intended_use": "Confirm one calibrated patient ventricular activation/QRS model.",
            "field_model": "monodomain",
            "activation_model": "reaction_eikonal_plus",
            "tissue_regions": [
                {
                    "region_id": "myocardium",
                    "element_tags": [1],
                    "ionic_model": "explicit-model",
                    "conduction_velocity_m_per_s": [0.6, 0.25, 0.2],
                    "intracellular_conductivity_s_per_m": [0.2, 0.08, 0.06],
                }
            ],
            "fibre_field_required": True,
            "parameter_bindings": [
                {
                    "parameter_name": "bulk_conductivity",
                    "parameter_unit": "S/m",
                    "target_path": "tissue_regions.myocardium.intracellular_conductivity_s_per_m.0",
                    "rationale": "Bind the mapped confirmation conductivity explicitly.",
                }
            ],
            "activation_translation": {
                "strategy_id": "activation-to-stimuli",
                "strategy_version": "1.0",
                "settings": {"max_sites": 32},
                "rationale": "Translate calibrated activation into explicit stimuli.",
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
                        "region_id": "torso",
                        "element_tags": [100],
                        "conductivity_s_per_m": 0.2,
                    }
                ],
            },
            "solver_options": {"linear_solver": "declared"},
            "provenance_reference": "Protocol QRS-001",
        }
    )


class _Driver:
    experiment_id = "patient-qrs-confirmation"
    experiment_version = "1.0.0"
    driver_version = "driver-v1"

    def __init__(self, available: bool = True):
        self._available = available
        self.calls = []

    def available(self) -> bool:
        return self._available

    def run(self, *, plan, experiment, output_path):
        self.calls.append((plan, experiment.experiment_id, output_path))
        Path(output_path).write_text(json.dumps({"status": "ok"}), encoding="utf-8")


class _OtherDriver(_Driver):
    experiment_version = "2.0.0"


def test_registry_resolves_exact_id_and_version() -> None:
    driver = _Driver()
    registry = OpenCARPExperimentDriverRegistry([driver])
    assert registry.resolve(_experiment()) is driver
    status = registry.status()
    assert status[0]["experiment_id"] == driver.experiment_id
    assert status[0]["experiment_version"] == driver.experiment_version
    assert status[0]["available"] is True


def test_registry_rejects_duplicate_exact_driver_identity() -> None:
    with pytest.raises(OpenCARPExperimentDriverError, match="duplicate"):
        OpenCARPExperimentDriverRegistry([_Driver(), _Driver()])


def test_registry_rejects_missing_or_unavailable_driver() -> None:
    with pytest.raises(OpenCARPExperimentDriverError, match="no registered"):
        OpenCARPExperimentDriverRegistry([_OtherDriver()]).resolve(_experiment())
    with pytest.raises(OpenCARPExperimentDriverError, match="unavailable"):
        OpenCARPExperimentDriverRegistry([_Driver(False)]).resolve(_experiment())


def _plan(tmp_path: Path) -> Path:
    experiment = _experiment().model_dump(mode="json")
    payload = {
        "request": {
            "settings": {"opencarp_experiment_spec": experiment}
        }
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dispatcher_fails_closed_when_no_driver_is_discovered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cardiep.opencarp_driver_dispatch.discover_opencarp_experiment_drivers",
        lambda: OpenCARPExperimentDriverRegistry(),
    )
    with pytest.raises(OpenCARPDriverDispatchError, match="no registered"):
        dispatch_registered_experiment(_plan(tmp_path), tmp_path / "result.json")


def test_dispatcher_calls_exact_registered_driver(tmp_path: Path, monkeypatch) -> None:
    driver = _Driver()
    monkeypatch.setattr(
        "cardiep.opencarp_driver_dispatch.discover_opencarp_experiment_drivers",
        lambda: OpenCARPExperimentDriverRegistry([driver]),
    )
    output = tmp_path / "result.json"
    dispatch_registered_experiment(_plan(tmp_path), output)
    assert output.is_file()
    assert driver.calls[0][1] == "patient-qrs-confirmation"
    assert Path(driver.calls[0][2]) == output.resolve()


def test_dispatcher_rejects_driver_that_returns_no_json(tmp_path: Path, monkeypatch) -> None:
    driver = _Driver()

    def _no_output(*, plan, experiment, output_path):
        return None

    driver.run = _no_output  # type: ignore[method-assign]
    monkeypatch.setattr(
        "cardiep.opencarp_driver_dispatch.discover_opencarp_experiment_drivers",
        lambda: OpenCARPExperimentDriverRegistry([driver]),
    )
    with pytest.raises(OpenCARPDriverDispatchError, match="without creating result JSON"):
        dispatch_registered_experiment(_plan(tmp_path), tmp_path / "missing.json")
