import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from cardiep import (
    ArtifactRef,
    EPCalibrationRequest,
    EPGeometry,
    EPObservation,
    EPParameterSet,
    RootSchedule,
    anisotropic_eikonal,
    apd_map,
    evaluate_observations,
    load_ep_geometry,
    pseudo_ecg,
    synthetic_tetra_geometry,
)
from cardiep.discrepancy import correlation_distance, field_discrepancy
from cardiep.native_backend import NativeEikonalBackend


def _write_geometry(tmp_path: Path) -> Path:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    path = tmp_path / "geometry.json"
    path.write_text(
        json.dumps(
            {
                "units": "cm",
                "node_xyz": geometry.node_xyz_cm.tolist(),
                "tetrahedra": geometry.tetrahedra.tolist(),
                "fibre": geometry.fibre.tolist(),
                "sheet": geometry.sheet.tolist(),
                "normal": geometry.normal.tolist(),
                "root_nodes": [0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_antipodal_fibre_axes_do_not_cancel() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    fibre = geometry.fibre.copy()
    fibre[1] *= -1.0
    flipped = EPGeometry(**{**geometry.__dict__, "fibre": fibre})
    result = anisotropic_eikonal(
        flipped,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    assert result.activation_ms[1] == pytest.approx(10.0)


def test_fibre_only_geometry_is_transversely_isotropic() -> None:
    base = synthetic_tetra_geometry(with_fibres=True)
    geometry = EPGeometry(
        **{
            **base.__dict__,
            "sheet": None,
            "normal": None,
        }
    )
    result = anisotropic_eikonal(
        geometry,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    assert result.activation_ms[2] == pytest.approx(result.activation_ms[3])
    assert any("transverse isotropy" in warning for warning in result.warnings)


def test_geometry_rejects_non_integer_and_degenerate_tetrahedra() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    with pytest.raises(ValueError, match="integer"):
        EPGeometry(node_xyz_cm=xyz, tetrahedra=np.asarray([[0, 1, 2, 2.5]]))
    with pytest.raises(ValueError, match="distinct"):
        EPGeometry(node_xyz_cm=xyz, tetrahedra=np.asarray([[0, 1, 2, 2]]))

    coplanar = xyz.copy()
    coplanar[3] = [1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="degenerate"):
        EPGeometry(node_xyz_cm=coplanar, tetrahedra=np.asarray([[0, 1, 2, 3]]))


def test_geometry_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = _write_geometry(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_ep_geometry(
            ArtifactRef(
                artifact_id="g",
                kind="ep_geometry",
                uri=path.as_uri(),
                sha256="0" * 64,
            )
        )


def test_pseudo_ecg_preserves_relative_lead_amplitudes() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    propagation = anisotropic_eikonal(
        geometry,
        RootSchedule(np.asarray([0]), np.asarray([0.0]), "test"),
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    repolarization = apd_map(geometry, propagation.activation_ms, {"apd_ms": 280.0})
    ecg = pseudo_ecg(
        geometry,
        propagation.activation_ms,
        repolarization.repolarization_ms,
        sample_rate_hz=250.0,
    )
    lead_peaks = np.max(np.abs(ecg.values), axis=1)
    assert np.max(lead_peaks) == pytest.approx(1.0)
    nonzero = lead_peaks[lead_peaks > 1e-8]
    assert len(nonzero) >= 2
    assert np.min(nonzero) < 0.95


def test_pseudo_ecg_rejects_nonfinite_or_reversed_repolarization() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    activation = np.asarray([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        pseudo_ecg(geometry, activation, np.asarray([10.0, 11.0, np.nan, 13.0]))
    with pytest.raises(ValueError, match="precede"):
        pseudo_ecg(geometry, activation, np.asarray([10.0, 11.0, 1.0, 13.0]))


def test_discrepancy_rejects_unsupported_observation_instead_of_zero_objective(
    tmp_path: Path,
) -> None:
    path = tmp_path / "other.json"
    path.write_text("{}\n", encoding="utf-8")
    observation = EPObservation(
        observation_id="other",
        kind="other",
        artifact=ArtifactRef(artifact_id="o", kind="other", uri=path.as_uri()),
    )
    with pytest.raises(ValueError, match="No supported discrepancy"):
        evaluate_observations(
            [observation],
            activation_ms=np.asarray([0.0, 1.0]),
            repolarization_ms=np.asarray([10.0, 11.0]),
            ecg=None,
        )


def test_discrepancy_supports_mae_and_rejects_constant_correlation() -> None:
    assert field_discrepancy(
        np.asarray([0.0, 2.0]),
        np.asarray([1.0, 4.0]),
        "mae",
    ) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="constant observed"):
        correlation_distance(np.ones(8), np.arange(8, dtype=float))


def test_observation_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "activation.json"
    path.write_text(json.dumps({"values_ms": [0.0, 1.0, 2.0, 3.0]}), encoding="utf-8")
    observation = EPObservation(
        observation_id="lat",
        kind="activation_map",
        artifact=ArtifactRef(
            artifact_id="lat",
            kind="activation_map",
            uri=path.as_uri(),
            sha256="f" * 64,
        ),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_observations(
            [observation],
            activation_ms=np.asarray([0.0, 1.0, 2.0, 3.0]),
            repolarization_ms=np.asarray([10.0, 11.0, 12.0, 13.0]),
            ecg=None,
        )


def test_calibration_rejects_initial_value_outside_bounds(tmp_path: Path) -> None:
    geometry = _write_geometry(tmp_path)
    observed = tmp_path / "activation.json"
    observed.write_text(
        json.dumps({"values_ms": [0.0, 10.0, 20.0, 40.0]}) + "\n",
        encoding="utf-8",
    )
    request = EPCalibrationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(artifact_id="g", kind="ep_geometry", uri=geometry.as_uri()),
        observations=[
            EPObservation(
                observation_id="lat",
                kind="activation_map",
                artifact=ArtifactRef(
                    artifact_id="lat",
                    kind="activation_map",
                    uri=observed.as_uri(),
                ),
            )
        ],
        backend="numpy-eikonal-v1",
        parameter_bounds={"fibre_speed": (0.05, 0.15)},
        initial_parameters=EPParameterSet(
            values={
                "fibre_speed": 0.5,
                "sheet_speed": 0.05,
                "normal_speed": 0.025,
                "apd_ms": 280.0,
            }
        ),
        settings={"with_ecg": False},
    )
    with pytest.raises(ValueError, match="outside"):
        NativeEikonalBackend().calibrate(request)


def test_contracts_reject_duplicate_observations_and_nonfinite_parameters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "obs.json"
    path.write_text("{}\n", encoding="utf-8")
    obs = {
        "observation_id": "same",
        "kind": "other",
        "artifact": {"artifact_id": "o", "kind": "other", "uri": path.as_uri()},
    }
    with pytest.raises(ValidationError, match="unique"):
        EPCalibrationRequest.model_validate(
            {
                "subject_id": "S1",
                "anatomy_ref": {
                    "artifact_id": "g",
                    "kind": "mesh",
                    "uri": path.as_uri(),
                },
                "observations": [obs, obs],
                "backend": "numpy-eikonal-v1",
                "parameter_bounds": {"speed": [0.1, 1.0]},
            }
        )
    with pytest.raises(ValidationError, match="finite"):
        EPParameterSet(values={"speed": float("inf")})


def test_valid_geometry_hash_is_accepted(tmp_path: Path) -> None:
    path = _write_geometry(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    geometry = load_ep_geometry(
        ArtifactRef(
            artifact_id="g",
            kind="ep_geometry",
            uri=path.as_uri(),
            sha256=digest,
        )
    )
    assert geometry.n_nodes == 4
