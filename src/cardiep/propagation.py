from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations

import numpy as np

from .conduction import RootSchedule
from .geometry import EPGeometry


@dataclass(frozen=True)
class PropagationResult:
    activation_ms: np.ndarray
    edge_count: int
    parameters: dict[str, float]
    root_schedule: RootSchedule
    warnings: tuple[str, ...] = ()


def _speed(parameters: dict[str, float], name: str, default: float) -> float:
    aliases = [name, f"{name}_cm_per_ms"]
    if name == "fibre_speed":
        aliases.extend(["fiber_speed", "fiber_speed_cm_per_ms"])
    value = next((parameters[key] for key in aliases if key in parameters), default)
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _axial_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average sign-ambiguous material axes without cancelling antipodal vectors."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    mean = a + b
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        return a / max(float(np.linalg.norm(a)), 1e-12)
    return mean / norm


def _axial_mean_many(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0:
        raise ValueError("Material-axis averaging requires an N x 3 array")
    reference = vectors[0] / max(float(np.linalg.norm(vectors[0])), 1e-12)
    aligned = []
    for vector in vectors:
        item = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if float(np.dot(reference, item)) < 0.0:
            item = -item
        aligned.append(item)
    mean = np.sum(np.asarray(aligned), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        return reference
    return mean / norm


def _orthonormal_basis(
    fibre: np.ndarray | None,
    sheet: np.ndarray | None,
    normal: np.ndarray | None,
    edge_direction: np.ndarray,
) -> np.ndarray:
    if fibre is None:
        f = edge_direction / max(float(np.linalg.norm(edge_direction)), 1e-12)
    else:
        f = fibre / max(float(np.linalg.norm(fibre)), 1e-12)

    if sheet is not None:
        s = sheet - np.dot(sheet, f) * f
        if float(np.linalg.norm(s)) > 1e-10:
            s = s / np.linalg.norm(s)
            n = np.cross(f, s)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            if normal is not None and float(np.dot(n, normal)) < 0.0:
                n = -n
            return np.column_stack((f, s, n))

    if normal is not None:
        n = normal - np.dot(normal, f) * f
        if float(np.linalg.norm(n)) > 1e-10:
            n = n / np.linalg.norm(n)
            s = np.cross(n, f)
            s = s / max(float(np.linalg.norm(s)), 1e-12)
            return np.column_stack((f, s, n))

    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, f))) > 0.85:
        axis = np.array([0.0, 1.0, 0.0])
    s = axis - np.dot(axis, f) * f
    s = s / max(float(np.linalg.norm(s)), 1e-12)
    n = np.cross(f, s)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    return np.column_stack((f, s, n))


def _tissue_scale(
    labels: np.ndarray | None,
    nodes: np.ndarray,
    params: dict[str, float],
) -> float:
    if labels is None:
        return 1.0
    border = float(params.get("border_zone_speed_scale", 0.55))
    scar = float(params.get("scar_speed_scale", 0.08))
    if not 0 < scar <= border <= 1:
        raise ValueError("Expected 0 < scar_speed_scale <= border_zone_speed_scale <= 1")
    scale = np.ones(len(nodes), dtype=float)
    selected = labels[np.asarray(nodes, dtype=int)]
    scale[selected == 1] = border
    scale[selected == 2] = scar
    return float(np.mean(scale))


def _tetra_metric(
    geometry: EPGeometry,
    tetrahedron: np.ndarray,
    *,
    fibre_speed: float,
    sheet_speed: float,
    normal_speed: float,
    isotropic_speed: float,
    transverse_speed: float,
    parameters: dict[str, float],
) -> np.ndarray:
    nodes = np.asarray(tetrahedron, dtype=int)
    tissue = _tissue_scale(geometry.scar_labels, nodes, parameters)
    if geometry.fibre is None:
        return np.eye(3, dtype=float) / (isotropic_speed * tissue) ** 2

    fibre = _axial_mean_many(geometry.fibre[nodes])
    if geometry.sheet is None and geometry.normal is None:
        inverse_transverse_sq = 1.0 / transverse_speed**2
        metric = (
            inverse_transverse_sq * np.eye(3, dtype=float)
            + (1.0 / fibre_speed**2 - inverse_transverse_sq)
            * np.outer(fibre, fibre)
        )
        return metric / tissue**2

    sheet = (
        None
        if geometry.sheet is None
        else _axial_mean_many(geometry.sheet[nodes])
    )
    normal = (
        None
        if geometry.normal is None
        else _axial_mean_many(geometry.normal[nodes])
    )
    basis = _orthonormal_basis(
        fibre,
        sheet,
        normal,
        np.asarray([1.0, 0.0, 0.0]),
    )
    inverse_speed_sq = np.diag(
        [
            1.0 / fibre_speed**2,
            1.0 / sheet_speed**2,
            1.0 / normal_speed**2,
        ]
    )
    return (basis @ inverse_speed_sq @ basis.T) / tissue**2


def _simplex_candidate(
    target_xyz: np.ndarray,
    known_xyz: np.ndarray,
    known_times: np.ndarray,
    metric: np.ndarray,
) -> float:
    """Hopf-Lax update over vertices/edges/faces of one tetrahedron.

    For every non-empty subset of the known opposite vertices, minimize the
    affine arrival-time interpolant plus anisotropic metric distance from the
    target vertex to that subset's convex hull. Interior stationary points are
    solved analytically; invalid stationary points fall back to lower-dimensional
    subsets. With at most three opposite vertices there are only seven subsets.
    """
    target = np.asarray(target_xyz, dtype=float)
    points = np.asarray(known_xyz, dtype=float)
    times = np.asarray(known_times, dtype=float).reshape(-1)
    if points.shape != (len(times), 3):
        raise ValueError("Known simplex coordinates/times have incompatible shapes")
    if len(times) == 0 or len(times) > 3:
        raise ValueError("A tetrahedral update requires one to three known vertices")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(times)):
        raise ValueError("Known simplex values must be finite")

    best = np.inf
    indices = range(len(times))
    for subset_size in range(1, len(times) + 1):
        for subset in combinations(indices, subset_size):
            selected = np.asarray(subset, dtype=int)
            subset_points = points[selected]
            subset_times = times[selected]

            if subset_size == 1:
                delta = target - subset_points[0]
                distance = float(np.sqrt(max(float(delta @ metric @ delta), 0.0)))
                best = min(best, float(subset_times[0] + distance))
                continue

            base = subset_points[-1]
            r0 = target - base
            basis = (subset_points[:-1] - base).T
            hessian = basis.T @ metric @ basis
            time_gradient = subset_times[:-1] - subset_times[-1]
            projected = basis.T @ metric @ r0

            try:
                h_inv_projected = np.linalg.solve(hessian, projected)
                h_inv_gradient = np.linalg.solve(hessian, time_gradient)
            except np.linalg.LinAlgError:
                continue

            causal = 1.0 - float(time_gradient @ h_inv_gradient)
            if causal <= 1e-12:
                continue
            orthogonal = r0 - basis @ h_inv_projected
            orthogonal_sq = float(orthogonal @ metric @ orthogonal)
            if orthogonal_sq < -1e-10:
                continue
            metric_distance = float(
                np.sqrt(max(orthogonal_sq, 0.0) / causal)
            )
            coordinates = h_inv_projected - metric_distance * h_inv_gradient
            weights = np.concatenate(
                [
                    coordinates,
                    np.asarray([1.0 - float(np.sum(coordinates))]),
                ]
            )
            if float(np.min(weights)) < -1e-9:
                continue
            weights = np.maximum(weights, 0.0)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                continue
            weights /= weight_sum

            point = weights @ subset_points
            delta = target - point
            distance = float(np.sqrt(max(float(delta @ metric @ delta), 0.0)))
            candidate = float(subset_times @ weights + distance)
            if np.isfinite(candidate):
                best = min(best, candidate)

    return float(best)


def anisotropic_eikonal(
    geometry: EPGeometry,
    roots: RootSchedule,
    parameters: dict[str, float],
) -> PropagationResult:
    fibre_speed = _speed(parameters, "fibre_speed", 0.065)
    sheet_speed = _speed(parameters, "sheet_speed", 0.051)
    normal_speed = _speed(parameters, "normal_speed", 0.048)
    isotropic_speed = _speed(parameters, "isotropic_speed", normal_speed)
    transverse_speed = _speed(
        parameters,
        "transverse_speed",
        0.5 * (sheet_speed + normal_speed),
    )

    warnings: list[str] = []
    if geometry.fibre is None:
        warnings.append("No fibre field supplied; propagation uses isotropic_speed.")
    elif geometry.sheet is None and geometry.normal is None:
        warnings.append(
            "No sheet/normal fields supplied; propagation uses transverse isotropy."
        )

    tetrahedra = np.asarray(geometry.tetrahedra, dtype=int)
    incident: list[list[int]] = [[] for _ in range(geometry.n_nodes)]
    metrics: list[np.ndarray] = []
    for tetra_index, tetrahedron in enumerate(tetrahedra):
        for node in tetrahedron:
            incident[int(node)].append(tetra_index)
        metric = _tetra_metric(
            geometry,
            tetrahedron,
            fibre_speed=fibre_speed,
            sheet_speed=sheet_speed,
            normal_speed=normal_speed,
            isotropic_speed=isotropic_speed,
            transverse_speed=transverse_speed,
            parameters=parameters,
        )
        if (
            metric.shape != (3, 3)
            or not np.all(np.isfinite(metric))
            or np.min(np.linalg.eigvalsh(metric)) <= 0
        ):
            raise ValueError("Non-positive-definite tetrahedral conduction metric")
        metrics.append(metric)

    distance = np.full(geometry.n_nodes, np.inf, dtype=float)
    heap: list[tuple[float, int]] = []
    for node, time in zip(roots.nodes.tolist(), roots.activation_ms.tolist(), strict=True):
        if time < distance[node]:
            distance[node] = float(time)
            heappush(heap, (float(time), int(node)))

    update_count = 0
    max_updates = max(10000, geometry.n_nodes * 128)
    while heap:
        current, changed_node = heappop(heap)
        if current > distance[changed_node] + 1e-10:
            continue

        for tetra_index in incident[changed_node]:
            tetrahedron = tetrahedra[tetra_index]
            metric = metrics[tetra_index]
            for target_raw in tetrahedron:
                target = int(target_raw)
                known = np.asarray(
                    [
                        int(node)
                        for node in tetrahedron
                        if int(node) != target and np.isfinite(distance[int(node)])
                    ],
                    dtype=int,
                )
                if known.size == 0:
                    continue
                candidate = _simplex_candidate(
                    geometry.node_xyz_cm[target],
                    geometry.node_xyz_cm[known],
                    distance[known],
                    metric,
                )
                if candidate + 1e-10 < distance[target]:
                    distance[target] = candidate
                    heappush(heap, (candidate, target))
                    update_count += 1
                    if update_count > max_updates:
                        raise RuntimeError(
                            "Tetrahedral Eikonal relaxation did not converge within "
                            f"{max_updates} updates"
                        )

    if not np.isfinite(distance).all():
        unreachable = int(np.sum(~np.isfinite(distance)))
        raise ValueError(f"EP mesh has {unreachable} nodes unreachable from the selected roots")
    distance = distance - float(np.min(distance))
    used = {
        "fibre_speed": fibre_speed,
        "sheet_speed": sheet_speed,
        "normal_speed": normal_speed,
        "isotropic_speed": isotropic_speed,
        "transverse_speed": transverse_speed,
        "border_zone_speed_scale": float(parameters.get("border_zone_speed_scale", 0.55)),
        "scar_speed_scale": float(parameters.get("scar_speed_scale", 0.08)),
    }
    return PropagationResult(
        activation_ms=distance,
        edge_count=len(geometry.edges),
        parameters=used,
        root_schedule=roots,
        warnings=tuple(warnings),
    )
