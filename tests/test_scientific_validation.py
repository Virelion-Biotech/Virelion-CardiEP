import json
from pathlib import Path

import numpy as np
import pytest

from cardiep import (
    ActivationProfile,
    AgreementThresholds,
    ConvergenceLevel,
    compare_activation_profiles,
    fenicsx_beat_niederer_reference,
    mesh_convergence_report,
    niederer_2011_spec,
)


def _profile(
    values,
    *,
    solver="solver",
    ids=("P1", "P2", "P3"),
    points=None,
    benchmark="test-benchmark",
):
    if points is None:
        points = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
    return ActivationProfile(
        benchmark_id=benchmark,
        solver_name=solver,
        sample_ids=tuple(ids),
        points_cm=np.asarray(points, dtype=float),
        activation_ms=np.asarray(values, dtype=float),
    )


def test_activation_profile_normalizes_units_and_sample_order() -> None:
    profile = ActivationProfile.from_dict(
        {
            "schema_version": "cardiep-activation-profile-v1",
            "benchmark_id": "b1",
            "solver": {"name": "external"},
            "coordinate_unit": "mm",
            "time_unit": "s",
            "sample_points": [
                {"id": "B", "xyz": [10.0, 0.0, 0.0]},
                {"id": "A", "xyz": [0.0, 0.0, 0.0]},
            ],
            "activation": {"A": 0.001, "B": 0.011},
        }
    )
    assert profile.sample_ids == ("B", "A")
    assert profile.points_cm.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert profile.activation_ms.tolist() == pytest.approx([11.0, 1.0])


def test_activation_profile_requires_exact_activation_keys() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        ActivationProfile.from_dict(
            {
                "schema_version": "cardiep-activation-profile-v1",
                "benchmark_id": "b1",
                "solver": {"name": "external"},
                "coordinate_unit": "cm",
                "time_unit": "ms",
                "sample_points": [{"id": "A", "xyz": [0.0, 0.0, 0.0]}],
                "activation": {"B": 1.0},
            }
        )


def test_cross_solver_comparison_is_order_independent_and_gated() -> None:
    reference = _profile([0.0, 10.0, 20.0], solver="reference")
    candidate = _profile(
        [20.5, 0.5, 10.5],
        solver="candidate",
        ids=("P3", "P1", "P2"),
        points=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )
    report = compare_activation_profiles(
        reference,
        candidate,
        thresholds=AgreementThresholds(
            rmse_ms_max=1.0,
            max_abs_ms_max=1.0,
            correlation_min=0.999,
            abs_bias_ms_max=1.0,
        ),
    )
    assert report["status"] == "pass"
    assert report["metrics"]["rmse_ms"] == pytest.approx(0.5)
    assert report["metrics"]["bias_ms"] == pytest.approx(0.5)


def test_cross_solver_gate_fails_material_disagreement() -> None:
    report = compare_activation_profiles(
        _profile([0.0, 10.0, 20.0], solver="reference"),
        _profile([0.0, 13.0, 27.0], solver="candidate"),
        thresholds=AgreementThresholds(
            rmse_ms_max=1.0,
            max_abs_ms_max=2.0,
            correlation_min=0.99,
            abs_bias_ms_max=1.0,
        ),
    )
    assert report["status"] == "fail"
    assert not report["checks"]["rmse_ms"]
    assert not report["checks"]["max_abs_ms"]


def test_p1_relative_alignment_removes_only_global_latency_offset() -> None:
    reference = _profile([5.0, 15.0, 25.0], solver="reference")
    candidate = _profile([12.0, 22.0, 32.0], solver="candidate")
    absolute = compare_activation_profiles(reference, candidate)
    relative = compare_activation_profiles(
        reference,
        candidate,
        alignment="p1-relative",
    )
    assert absolute["metrics"]["bias_ms"] == pytest.approx(7.0)
    assert relative["metrics"]["rmse_ms"] == pytest.approx(0.0, abs=1e-12)


def test_coordinate_mismatch_fails_closed() -> None:
    reference = _profile([0.0, 10.0, 20.0])
    candidate = _profile(
        [0.0, 10.0, 20.0],
        points=[[0.0, 0.0, 0.0], [0.51, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="coordinates differ"):
        compare_activation_profiles(reference, candidate, point_tolerance_cm=1e-3)


def test_mesh_convergence_recovers_second_order_manufactured_error() -> None:
    exact = _profile([0.0, 10.0, 20.0], solver="exact")
    levels = []
    for h in (0.4, 0.2, 0.1):
        error = 10.0 * h**2
        levels.append(
            ConvergenceLevel(
                h=h,
                profile=_profile(
                    np.asarray([0.0, 10.0, 20.0]) + error,
                    solver=f"mesh-{h}",
                ),
            )
        )
    report = mesh_convergence_report(levels, exact=exact)
    assert report["monotone_self_convergence"] is True
    assert report["observed_orders"] == pytest.approx([2.0, 2.0])
    assert report["gci_fine_ms"] is not None
    assert report["errors_to_exact_ms"] == pytest.approx([1.6, 0.4, 0.1])


def test_mesh_self_convergence_without_exact_truth_recovers_order() -> None:
    levels = []
    for h in (0.4, 0.2, 0.1, 0.05):
        error = 8.0 * h**2
        levels.append(
            ConvergenceLevel(
                h=h,
                profile=_profile(
                    np.asarray([1.0, 11.0, 21.0]) + error,
                    solver=f"mesh-{h}",
                ),
            )
        )
    report = mesh_convergence_report(levels)
    assert report["monotone_self_convergence"] is True
    assert report["observed_orders"] == pytest.approx([2.0, 2.0])


def test_niederer_spec_matches_standard_geometry_and_points() -> None:
    spec = niederer_2011_spec()
    assert spec["geometry"]["dimensions_cm"] == [2.0, 0.7, 0.3]
    assert spec["stimulus"]["box_dimensions_cm"] == [0.15, 0.15, 0.15]
    assert spec["sample_points_cm"]["P8"] == [2.0, 0.7, 0.3]
    assert spec["sample_points_cm"]["P9"] == [1.0, 0.35, 0.15]
    assert spec["activation_threshold_mV"] == 0.0


def test_pinned_fenicsx_niederer_fixture_is_canonical_roundtrip(tmp_path: Path) -> None:
    fixture = fenicsx_beat_niederer_reference()
    assert fixture.sample_ids == tuple(f"P{i}" for i in range(1, 10))
    assert fixture.activation_ms[7] == pytest.approx(37.93)
    assert fixture.metadata["dx_mm"] == pytest.approx(0.1)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture.to_dict()) + "\n", encoding="utf-8")
    restored = ActivationProfile.from_json(path)
    assert np.allclose(restored.points_cm, fixture.points_cm)
    assert np.allclose(restored.activation_ms, fixture.activation_ms)


def test_mesh_convergence_requires_common_benchmark() -> None:
    levels = [
        ConvergenceLevel(h=0.4, profile=_profile([0, 1, 2], benchmark="a")),
        ConvergenceLevel(h=0.2, profile=_profile([0, 1, 2], benchmark="a")),
        ConvergenceLevel(h=0.1, profile=_profile([0, 1, 2], benchmark="b")),
    ]
    with pytest.raises(ValueError, match="same benchmark"):
        mesh_convergence_report(levels)
