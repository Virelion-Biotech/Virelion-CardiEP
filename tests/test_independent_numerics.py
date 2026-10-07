"""Independent optimization and graph references; no PDE equivalence claim."""

import numpy as np
import pytest

optimize = pytest.importorskip("scipy.optimize")
csgraph = pytest.importorskip("scipy.sparse.csgraph")

from cardiep.propagation import _simplex_candidate
from cardiep.surface_backend import (
    SurfaceGeometry,
    _adjacency,
    _bounded_inverse_speed_fit,
    _distances,
)


@pytest.mark.parametrize("seed", range(12))
def test_tetrahedral_local_update_matches_independent_convex_optimization(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(3, 3))
    target = rng.normal(size=3)
    matrix = rng.normal(size=(3, 3))
    metric = matrix.T @ matrix + np.eye(3)
    times = rng.uniform(0, 2, size=3)

    def objective(weights):
        delta = target - weights @ points
        return float(weights @ times + np.sqrt(delta @ metric @ delta))

    reference = min(
        optimize.minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0, 1)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            options={"ftol": 1e-12, "maxiter": 500},
        ).fun
        for start in [np.full(3, 1 / 3), *np.eye(3)]
    )
    assert _simplex_candidate(target, points, times, metric) == pytest.approx(reference, abs=2e-7)


@pytest.mark.parametrize("seed", range(10))
def test_constrained_surface_fit_matches_scipy_lsq_linear(seed):
    rng = np.random.default_rng(seed)
    distances = rng.uniform(0, 3, size=20)
    observed = rng.normal(size=20) + 12 * distances + 20
    speed_bounds, offset_bounds = (0.05, 0.2), (-2.0, 2.0)
    inv_speed, offset = _bounded_inverse_speed_fit(distances, observed, speed_bounds, offset_bounds)
    reference = optimize.lsq_linear(
        np.column_stack([distances, np.ones(20)]), observed, bounds=([5, -2], [20, 2]), tol=1e-12
    )
    assert [inv_speed, offset] == pytest.approx(reference.x, abs=1e-7)
    assert offset_bounds[0] <= offset <= offset_bounds[1]


def test_surface_shortest_paths_match_scipy():
    geometry = SurfaceGeometry(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
    )
    adjacency = _adjacency(geometry)
    dense = np.zeros((4, 4))
    for node, neighbors in enumerate(adjacency):
        for neighbor, distance in neighbors:
            dense[node, neighbor] = distance
    assert _distances(adjacency, 0) == pytest.approx(csgraph.dijkstra(dense, indices=0))
