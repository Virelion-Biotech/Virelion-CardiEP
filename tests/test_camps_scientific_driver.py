from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardiep.camps_cdt_backend import CAMPS_CDT_BACKEND_NAME
from cardiep.camps_scientific_driver import (
    AUDITED_UPSTREAM_COMMIT,
    CampsScientificDriverError,
    _instrument_source,
    expand_independent_ecg_to_standard_12,
    export_from_upstream_namespace,
    select_minimum_finite_discrepancy,
)
from cardiep.camps_workspace import CampsWorkspaceManifest
from cardiep.models import ArtifactRef, EPCalibrationRequest, EPObservation

LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
BUNDLE = "b" * 64


def _artifact(name: str, kind: str, *, frame: str | None = "model-mm") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=name,
        kind=kind,
        uri=f"file:///{name}.csv",
        sha256="2" * 64,
        coordinate_frame=frame,
    )


def _workspace() -> CampsWorkspaceManifest:
    return CampsWorkspaceManifest(
        subject_id="P1",
        source_bundle_fingerprint=BUNDLE,
        coordinate_frame="model-mm",
        coordinate_unit="mm",
        index_base=0,
        node_xyz=_artifact("xyz", "node_xyz"),
        tetra=_artifact("tetra", "tetra_connectivity", frame=None),
        lvendo_nodes=_artifact("lvendo", "lvendo_nodes", frame=None),
        rvendo_nodes=_artifact("rvendo", "rvendo_nodes", frame=None),
        fibre=_artifact("fibre", "fibre_field"),
        sheet=_artifact("sheet", "sheet_field"),
        normal=_artifact("normal", "normal_field"),
        material=_artifact("material", "tetra_material", frame=None),
        vc_ab_cut=_artifact("ab", "ventricular_coordinate_ab_cut"),
        vc_rt=_artifact("rt", "ventricular_coordinate_rt"),
        electrode_xyz=_artifact("electrodes", "electrode_xyz"),
    )


def _request() -> EPCalibrationRequest:
    return EPCalibrationRequest(
        subject_id="P1",
        anatomy_ref=ArtifactRef(
            artifact_id="anatomy",
            kind="volume_mesh",
            uri="file:///patient.vtu",
            sha256="a" * 64,
            coordinate_frame="model-mm",
            metadata={"bundle_fingerprint": BUNDLE},
        ),
        observations=[
            EPObservation(
                observation_id="fit",
                kind="ecg",
                artifact=ArtifactRef(
                    artifact_id="fit-ecg",
                    kind="electrotrace_ecg_calibration",
                    uri="file:///fit.json",
                    sha256="d" * 64,
                    metadata={"lead_names": LEADS, "sample_rate_hz": 1000.0},
                ),
                units="mV",
            )
        ],
        backend=CAMPS_CDT_BACKEND_NAME,
        parameter_bounds={"fibre_speed": (0.03, 0.10)},
        settings={
            "upstream_commit": AUDITED_UPSTREAM_COMMIT,
            "expected_leads": LEADS,
            "calibration_target": "activation_qrs",
            "inference_method": "smc_abc",
            "upstream_sample_rate_hz": 1000.0,
            "ecg_resampling_method": "none",
            "workspace_manifest": _workspace().model_dump(mode="json"),
        },
    )


class _EPModel:
    def simulate_electrophysiology(self, *, parameter_particle_modules_dict):
        assert "propagation_module" in parameter_particle_modules_dict
        return np.asarray([0.0, 2.0, 5.0]), np.zeros((3, 4))


class _Simulator:
    def __init__(self):
        self.electrophysiology_model = _EPModel()

    def simulate_particle(self, *, parameter_particle_modules_dict):
        assert "propagation_module" in parameter_particle_modules_dict
        base = np.arange(5, dtype=float)
        return np.vstack([base + offset for offset in range(8)])


class _Adapter:
    parameter_name_list_in_order = ["fibre_speed", "root_0"]

    def get_theta_names(self):
        return ["fibre_speed", "root_0"]

    def distribute_parameter_to_modules(self, *, parameter_particle):
        return {"propagation_module": np.asarray(parameter_particle, dtype=float)}


class _Evaluator:
    def __init__(self):
        self.adapter = _Adapter()
        self.simulator = _Simulator()

    def evaluate_parameter_population(self, *, parameter_population):
        assert np.asarray(parameter_population).shape == (3, 2)
        return np.asarray([0.45, np.nan, 0.12])


class _Inference:
    def __init__(self):
        self.evaluator = _Evaluator()


def test_selects_minimum_finite_discrepancy() -> None:
    assert select_minimum_finite_discrepancy([0.4, np.nan, 0.2, np.inf]) == (2, 0.2)
    with pytest.raises(CampsScientificDriverError, match="no finite"):
        select_minimum_finite_discrepancy([np.nan, np.inf])


def test_expands_eight_independent_leads_to_standard_twelve() -> None:
    base = np.arange(4, dtype=float)
    ecg = np.vstack([base + offset for offset in range(8)])
    expanded = expand_independent_ecg_to_standard_12(ecg)
    assert expanded.shape == (12, 4)
    lead_i, lead_ii = ecg[0], ecg[1]
    assert np.allclose(expanded[2], lead_ii - lead_i)
    assert np.allclose(expanded[3], -(lead_i + lead_ii) / 2.0)
    assert np.allclose(expanded[4], lead_i - lead_ii / 2.0)
    assert np.allclose(expanded[5], lead_ii - lead_i / 2.0)
    assert np.allclose(expanded[6:], ecg[2:])


def test_instrumentation_requires_exact_audited_marker(tmp_path: Path) -> None:
    source = (
        "if __name__ == '__main__':\n"
        "        population_parameter = inference_method.get_parameter_from_theta(population_theta)\n"
        "        inference_method = None\n"
    )
    instrumented = _instrument_source(
        source,
        plan_path=tmp_path / "plan.json",
        output_path=tmp_path / "result.json",
    )
    assert "export_from_upstream_namespace" in instrumented
    assert "raise SystemExit(0)" in instrumented
    with pytest.raises(CampsScientificDriverError, match="exactly once"):
        _instrument_source(
            "population_parameter = inference_method.get_parameter_from_theta(population_theta)\n",
            plan_path=tmp_path / "plan.json",
            output_path=tmp_path / "result.json",
        )


def test_full_export_uses_same_evaluator_and_validates_contract(tmp_path: Path) -> None:
    request = _request()
    workspace = _workspace()
    plan = {
        "request": request.model_dump(mode="json"),
        "upstream_commit": AUDITED_UPSTREAM_COMMIT,
        "workspace_manifest_fingerprint": workspace.fingerprint,
        "runtime_root": str(tmp_path / "runtime"),
    }
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "result.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    namespace = {
        "inference_method": _Inference(),
        "population_parameter": np.asarray(
            [[0.05, 0.0], [0.06, 1.0], [0.07, 1.0]], dtype=float
        ),
        "population_theta": np.asarray(
            [[0.05, 0.0], [0.06, 1.0], [0.07, 1.0]], dtype=float
        ),
        "inference_success": True,
    }
    export_from_upstream_namespace(namespace, plan_path, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["objective"] == pytest.approx(0.12)
    assert payload["diagnostics"]["selected_index"] == 2
    assert payload["diagnostics"]["representative_selection"] == (
        "minimum_finite_discrepancy_same_inference_evaluator"
    )
    assert payload["parameters"]["values"] == {
        "fibre_speed": pytest.approx(0.07),
        "root_0": pytest.approx(1.0),
    }
    assert payload["parameters"]["units"] == {
        "fibre_speed": "cm/ms",
        "root_0": "1",
    }
    outputs = payload["simulated"]["outputs"]
    activation = next(item for item in outputs if item["kind"] == "activation_map")
    ecg = next(item for item in outputs if item["kind"] == "calibration_ecg")
    assert activation["sha256"]
    assert activation["coordinate_frame"] == "model-mm"
    assert ecg["metadata"]["lead_names"] == LEADS
    assert ecg["metadata"]["sample_rate_hz"] == 1000.0
    assert ecg["metadata"]["role"] == "calibration_forward_model"
    assert payload["provenance"]["upstream_commit"] == AUDITED_UPSTREAM_COMMIT
    assert payload["provenance"]["request_sha256"]
