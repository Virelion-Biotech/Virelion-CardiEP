import numpy as np
import pytest

from cardiep import (
    ActivationProfile,
    AgreementThresholds,
    ConvergenceLevel,
    EPGeometry,
    RootSchedule,
    anisotropic_eikonal,
    mesh_convergence_report,
)
from cardiep.cellular import membrane_voltage
from cardiep.repolarization import apd_map
from cardiep.scientific_validation import compare_activation_profiles
from cardiep.surface_backend import SurfaceGeometry


def profile(values, **kwargs):
    return ActivationProfile(
        "audit",
        "solver",
        ("P1", "P2", "P3"),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
        np.array(values, dtype=float),
        **kwargs,
    )


def tetra(scale=1.0):
    return EPGeometry(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * scale,
        np.array([[0, 1, 2, 3]]),
    )


def test_distinct_equation_classes_cannot_claim_agreement():
    with pytest.raises(ValueError, match="equation"):
        compare_activation_profiles(
            profile([0, 1, 2], equation="monodomain"), profile([0, 1, 2], equation="eikonal")
        )


def test_distinct_ionic_models_cannot_claim_agreement():
    with pytest.raises(ValueError, match="ionic_model"):
        compare_activation_profiles(
            profile([0, 1, 2], ionic_model="TP06"), profile([0, 1, 2], ionic_model="ToRORd")
        )


def test_identical_constant_profiles_have_undefined_correlation():
    p = profile([1, 1, 1])
    report = compare_activation_profiles(p, p, thresholds=AgreementThresholds(1, 1, 0.9, 1))
    assert report["metrics"]["correlation"] is None
    assert report["status"] == "fail"


def test_nonuniform_self_refinement_has_no_uniform_order_estimate():
    levels = [ConvergenceLevel(h, profile(np.array([0, 1, 2]) + h * h)) for h in (0.5, 0.2, 0.1)]
    report = mesh_convergence_report(levels)
    assert not report["approximately_uniform_refinement"]
    assert all(order is None for order in report["observed_orders"])
    assert report["gci_fine_ms"] is None


def test_divergent_finest_pair_cannot_reuse_earlier_positive_order_for_gci():
    exact = profile([0, 1, 2])
    levels = [
        ConvergenceLevel(h, profile(np.array([0, 1, 2]) + error))
        for h, error in [(0.4, 1), (0.2, 0.1), (0.1, 0.2)]
    ]
    assert mesh_convergence_report(levels, exact=exact)["gci_fine_ms"] is None


@pytest.mark.parametrize("scale", [1e-5, 1.0, 1e5])
def test_valid_tetrahedral_cells_are_scale_invariant(scale):
    geometry = tetra(scale)
    assert geometry.n_nodes == 4


@pytest.mark.parametrize("nodes", [[-1], [4]])
def test_direct_solver_rejects_out_of_bounds_roots(nodes):
    with pytest.raises(ValueError, match="root|Root"):
        anisotropic_eikonal(
            tetra(), RootSchedule(np.array(nodes), np.array([0.0]), "test"), {"isotropic_speed": 1}
        )


@pytest.mark.parametrize("activation", [np.array([0, 1, 2, np.nan]), np.array([0, 1, 2, np.inf])])
def test_apd_rejects_nonfinite_activation(activation):
    with pytest.raises(ValueError, match="finite"):
        apd_map(tetra(), activation, {"apd_ms": 300})


def test_surface_fractional_connectivity_is_not_truncated():
    with pytest.raises(ValueError, match="integer"):
        SurfaceGeometry(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1.9, 2]]))


def test_collinear_surface_cells_are_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        SurfaceGeometry(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([[0, 1, 2]]))


def test_cellular_waveforms_reject_invalid_time_constants():
    with pytest.raises(ValueError, match="finite"):
        membrane_voltage(
            np.array([0.0]), np.array([300.0]), np.array([0.0, 1.0]), upstroke_tau_ms=np.nan
        )


def test_profile_arrays_do_not_alias_callers():
    p = profile([0, 1, 2])
    with pytest.raises(ValueError):
        p.activation_ms[0] = 900


def test_meshio_keeps_every_linear_tetrahedral_block(tmp_path, monkeypatch):
    meshio = pytest.importorskip("meshio")
    from cardiep.geometry import _meshio_geometry

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    mesh = meshio.Mesh(points, [("tetra", [[0, 1, 2, 3]]), ("tetra", [[1, 2, 3, 4]])])
    monkeypatch.setattr(meshio, "read", lambda path: mesh)
    result = _meshio_geometry(tmp_path / "mesh.vtu", unit="cm")
    assert len(result.tetrahedra) == 2


def test_service_rejects_backend_anatomy_identity_conflicts():
    from cardiep import (
        ArtifactRef,
        CardiEPService,
        EPParameterSet,
        EPSimulationRequest,
        EPSimulationResult,
    )
    from cardiep.service import ReadinessError

    class Backend:
        name = "fake"

        def available(self):
            return True

        def simulate(self, request):
            return EPSimulationResult(
                subject_id=request.subject_id,
                backend=self.name,
                parameters=request.parameters,
                provenance={"anatomy_artifact_id": "wrong"},
            )

    request = EPSimulationRequest(
        subject_id="s1",
        anatomy_ref=ArtifactRef(artifact_id="correct", kind="mesh", uri="mesh.json"),
        backend="fake",
        parameters=EPParameterSet(),
    )
    with pytest.raises(ReadinessError, match="anatomy"):
        CardiEPService([Backend()], register_defaults=False).simulate(request)


def test_equal_distance_surface_points_cannot_identify_speed_and_offset():
    from cardiep.surface_backend import _bounded_inverse_speed_fit

    with pytest.raises(ValueError, match="unidentifiable"):
        _bounded_inverse_speed_fit(
            np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0]), (0.05, 0.2)
        )


def test_surface_offset_bound_is_respected():
    from cardiep.surface_backend import _bounded_inverse_speed_fit

    inverse, offset = _bounded_inverse_speed_fit(
        np.array([0.0, 1.0, 2.0]), np.array([100.0, 110.0, 120.0]), (0.05, 0.2), (-1, 1)
    )
    assert -1 <= offset <= 1
    assert 5 <= inverse <= 20


def test_surface_rejects_ignored_parameters():
    from cardiep.surface_backend import _surface_configuration

    with pytest.raises(ValueError, match="Unsupported"):
        _surface_configuration({}, {"fibre_speed": 0.1}, calibration=True)


def test_external_wrapper_artifacts_survive_temporary_exchange_directory(tmp_path):
    import json
    import sys

    from cardiep import ArtifactRef, EPParameterSet, EPSimulationRequest
    from cardiep.external import SubprocessEPBackend
    from cardiep.provenance import uri_to_path, verify_file_sha256

    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text("""import json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
artifact = out.parent/'activation.json'
artifact.write_text('{"activation_ms":[0,10]}')
result = dict(subject_id=request['subject_id'], backend='wrapper', parameters=request['parameters'],
outputs=[dict(artifact_id='activation', kind='activation_map', uri=artifact.as_uri())])
out.write_text(json.dumps(result))
""")
    backend = SubprocessEPBackend(
        name="wrapper", command=[sys.executable, str(wrapper), "{request}", "{output}"]
    )
    request = EPSimulationRequest(
        subject_id="s1",
        anatomy_ref=ArtifactRef(artifact_id="mesh", kind="mesh", uri="mesh.json"),
        backend="wrapper",
        parameters=EPParameterSet(),
        settings={"output_dir": str(tmp_path / "saved")},
    )
    result = backend.simulate(request)
    path = uri_to_path(result.outputs[0].uri)
    assert json.loads(path.read_text())["activation_ms"] == [0, 10]
    verify_file_sha256(path, result.outputs[0].sha256)


def test_artifact_writer_rejects_path_traversal_and_preserves_old_data(tmp_path):
    from cardiep.artifacts import write_json_artifact

    with pytest.raises(ValueError, match="filename"):
        write_json_artifact(tmp_path, artifact_id="../escape", kind="test", payload={})
    write_json_artifact(tmp_path, artifact_id="same", kind="test", payload={"value": 1})
    original = (tmp_path / "same.json").read_bytes()
    with pytest.raises(ValueError):
        write_json_artifact(tmp_path, artifact_id="same", kind="test", payload={"value": np.nan})
    assert (tmp_path / "same.json").read_bytes() == original


def test_declared_parameter_units_cannot_be_silently_reinterpreted():
    from cardiep import EPParameterSet
    from cardiep.configuration import validate_parameter_units

    with pytest.raises(ValueError, match="units"):
        validate_parameter_units(
            EPParameterSet(values={"isotropic_speed": 0.1}, units={"isotropic_speed": "mm/ms"})
        )


def test_remote_file_uri_is_not_silently_treated_as_a_local_path():
    from cardiep.provenance import uri_to_path

    with pytest.raises(ValueError, match="authorities"):
        uri_to_path("file://remote-server/path/mesh.json")
