from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardiep import (
    ArtifactRef,
    CardiEPService,
    EPCalibrationRequest,
    EPObservation,
    EPSimulationRequest,
    EPParameterSet,
    SURFACE_BACKEND_NAME,
)
from cardiep.provenance import file_sha256


def _artifact(path: Path, *, artifact_id: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.resolve().as_uri(),
        sha256=file_sha256(path),
    )


def _surface(tmp_path: Path) -> ArtifactRef:
    path = tmp_path / "surface.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "cardiep-surface-v1",
                "coordinate_unit": "cm",
                "vertices": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "triangles": [[0, 1, 2], [0, 2, 3]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return _artifact(path, artifact_id="surface", kind="surface_mesh")


def _eam(tmp_path: Path) -> ArtifactRef:
    speed = 0.1
    offset = 5.0
    distances = np.asarray([0.0, 1.0, np.sqrt(2.0), 1.0])
    path = tmp_path / "eam.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "cardiep-eam-activation-v1",
                "vertex_indices": [1, 2, 3],
                "activation_ms": (offset + distances[[1, 2, 3]] / speed).tolist(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return _artifact(path, artifact_id="eam", kind="eam_activation")


def test_surface_backend_calibrates_known_root_speed_and_offset(tmp_path: Path) -> None:
    service = CardiEPService()
    assert SURFACE_BACKEND_NAME in service.backends()
    anatomy = _surface(tmp_path)
    observation = EPObservation(
        observation_id="eam-1",
        kind="eam_activation",
        artifact=_eam(tmp_path),
        units="ms",
    )
    result = service.calibrate(
        EPCalibrationRequest(
            subject_id="S1",
            anatomy_ref=anatomy,
            observations=[observation],
            backend=SURFACE_BACKEND_NAME,
            parameter_bounds={"isotropic_speed_cm_per_ms": (0.05, 0.20)},
            settings={
                "root_candidates": [0, 1, 2, 3],
                "output_dir": str(tmp_path / "out"),
            },
        )
    )
    assert result.converged is True
    assert result.objective == pytest.approx(0.0, abs=1e-10)
    assert result.diagnostics["root_node"] == 0
    assert result.parameters.values["isotropic_speed_cm_per_ms"] == pytest.approx(0.1)
    assert result.parameters.values["activation_offset_ms"] == pytest.approx(5.0)
    assert result.simulated is not None
    assert result.simulated.validation_status == "software_checked"


def test_surface_backend_simulation_writes_full_activation_map(tmp_path: Path) -> None:
    result = CardiEPService().simulate(
        EPSimulationRequest(
            subject_id="S1",
            anatomy_ref=_surface(tmp_path),
            backend=SURFACE_BACKEND_NAME,
            parameters=EPParameterSet(
                values={
                    "isotropic_speed_cm_per_ms": 0.1,
                    "activation_offset_ms": 5.0,
                },
                source="fixed",
            ),
            settings={"root_node": 0, "output_dir": str(tmp_path / "out")},
        )
    )
    activation_ref = next(item for item in result.outputs if item.kind == "activation_map")
    payload = json.loads(Path(activation_ref.uri.removeprefix("file://")).read_text())
    assert payload["root_node"] == 0
    assert payload["activation_ms"][0] == pytest.approx(5.0)
    assert payload["activation_ms"][1] == pytest.approx(15.0)
    assert payload["activation_ms"][2] == pytest.approx(5.0 + 10.0 * np.sqrt(2.0))


def test_surface_backend_rejects_missing_speed_bound(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isotropic_speed"):
        CardiEPService().calibrate(
            EPCalibrationRequest(
                subject_id="S1",
                anatomy_ref=_surface(tmp_path),
                observations=[
                    EPObservation(
                        observation_id="eam-1",
                        kind="eam_activation",
                        artifact=_eam(tmp_path),
                    )
                ],
                backend=SURFACE_BACKEND_NAME,
                parameter_bounds={"other": (0.1, 1.0)},
                settings={"root_candidates": [0, 1]},
            )
        )
