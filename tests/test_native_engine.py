import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from cardiep import EPAPI, NATIVE_BACKEND_NAME, synthetic_tetra_geometry


def _geometry_file(tmp_path: Path) -> Path:
    geo = synthetic_tetra_geometry(with_fibres=True)
    path = tmp_path / "ep_geometry.json"
    path.write_text(
        json.dumps(
            {
                "units": "cm",
                "node_xyz": geo.node_xyz_cm.tolist(),
                "tetrahedra": geo.tetrahedra.tolist(),
                "fibre": geo.fibre.tolist(),
                "sheet": geo.sheet.tolist(),
                "normal": geo.normal.tolist(),
                "ventricular_coordinates": {
                    key: value.tolist() for key, value in geo.ventricular_coordinates.items()
                },
                "electrodes": {
                    key: value.tolist() for key, value in geo.electrodes_cm.items()
                },
                "root_nodes": [0],
                "endocardial_nodes": [0, 1, 2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_default_api_has_native_backend() -> None:
    health = EPAPI().health()
    status = {item["name"]: item for item in health["backends"]}
    assert status[NATIVE_BACKEND_NAME]["available"] is True


def test_native_simulation_writes_typed_artifacts(tmp_path: Path) -> None:
    geometry = _geometry_file(tmp_path)
    result = EPAPI().simulate(
        {
            "subject_id": "S1",
            "anatomy_ref": {
                "artifact_id": "geometry",
                "kind": "ep_geometry",
                "uri": geometry.as_uri(),
            },
            "backend": NATIVE_BACKEND_NAME,
            "parameters": {
                "values": {
                    "fibre_speed": 0.1,
                    "sheet_speed": 0.05,
                    "normal_speed": 0.025,
                    "apd_min_ms": 250.0,
                    "apd_max_ms": 300.0,
                    "apd_gradient_tm": 1.0,
                },
                "source": "fixed",
            },
            "settings": {
                "output_dir": str(tmp_path / "outputs"),
                "ecg_sample_rate_hz": 250.0,
            },
        }
    )
    assert result["validation_status"] == "software_checked"
    assert result["backend"] == NATIVE_BACKEND_NAME
    assert len(result["provenance"]["run_sha256"]) == 64
    assert len(result["provenance"]["implementation_sha256"]) == 64
    assert len(result["provenance"]["runtime_fingerprint_sha256"]) == 64
    assert result["provenance"]["runtime"]["cardiep_version"] == "0.3.0"
    kinds = {item["kind"] for item in result["outputs"]}
    assert {"activation_map", "repolarization_map", "pseudo_ecg", "ep_summary"} <= kinds
    for artifact in result["outputs"]:
        assert Path(urlparse(artifact["uri"]).path).is_file()
        assert len(artifact["sha256"]) == 64


def test_native_calibration_fits_activation_map(tmp_path: Path) -> None:
    geometry = _geometry_file(tmp_path)
    observed = tmp_path / "activation.json"
    observed.write_text(
        json.dumps({"values_ms": [0.0, 10.0, 20.0, 40.0]}) + "\n",
        encoding="utf-8",
    )
    result = EPAPI().calibrate(
        {
            "subject_id": "S1",
            "anatomy_ref": {
                "artifact_id": "geometry",
                "kind": "ep_geometry",
                "uri": geometry.as_uri(),
            },
            "observations": [
                {
                    "observation_id": "lat",
                    "kind": "activation_map",
                    "artifact": {
                        "artifact_id": "lat-artifact",
                        "kind": "activation_map",
                        "uri": observed.as_uri(),
                    },
                    "coordinate_frame": "ep_mesh_node_order",
                    "units": "ms",
                }
            ],
            "backend": NATIVE_BACKEND_NAME,
            "parameter_bounds": {"fibre_speed": [0.05, 0.15]},
            "initial_parameters": {
                "values": {
                    "sheet_speed": 0.05,
                    "normal_speed": 0.025,
                    "apd_ms": 280.0,
                },
                "source": "fixed",
            },
            "settings": {
                "with_ecg": False,
                "max_iterations": 4,
                "output_dir": str(tmp_path / "calibrated"),
            },
        }
    )
    assert result["objective"] == pytest.approx(0.0, abs=1e-10)
    assert result["parameters"]["values"]["fibre_speed"] == pytest.approx(0.1)
    assert result["simulated"]["validation_status"] == "software_checked"
