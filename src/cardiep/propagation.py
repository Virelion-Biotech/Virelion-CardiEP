from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

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
    aliases = (name, f"{name}_cm_per_ms")
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


def _edge_tissue_scale(labels: np.ndarray | None, u: int, v: int, params: dict[str, float]) -> float:
    if labels is None:
        return 1.0
    border = float(params.get("border_zone_speed_scale", 0.55))
    scar = float(params.get("scar_speed_scale", 0.08))
    if not 0 < scar <= border <= 1:
        raise ValueError("Expected 0 < scar_speed_scale <= border_zone_speed_scale <= 1")
    scale = np.ones(2, dtype=float)
    for index, node in enumerate((u, v)):
        if labels[node] == 1:
            scale[index] = border
        elif labels[node] == 2:
            scale[index] = scar
    return float(np.mean(scale))


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

    edges = geometry.edges
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(geometry.n_nodes)]
    warnings: list[str] = []
    if geometry.fibre is None:
        warnings.append("No fibre field supplied; propagation uses isotropic_speed.")
    elif geometry.sheet is None and geometry.normal is None:
        warnings.append(
            "No sheet/normal fields supplied; propagation uses transverse isotropy."
        )

    for u_raw, v_raw in edges:
        u, v = int(u_raw), int(v_raw)
        delta = geometry.node_xyz_cm[v] - geometry.node_xyz_cm[u]
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        tissue = _edge_tissue_scale(geometry.scar_labels, u, v, parameters)
        if geometry.fibre is None:
            cost = length / (isotropic_speed * tissue)
        else:
            fibre = _axial_mean(geometry.fibre[u], geometry.fibre[v])
            if geometry.sheet is None and geometry.normal is None:
                longitudinal = float(np.dot(delta, fibre))
                transverse_sq = max(length**2 - longitudinal**2, 0.0)
                cost = float(
                    np.sqrt(
                        (longitudinal / fibre_speed) ** 2
                        + transverse_sq / transverse_speed**2
                    )
                ) / tissue
            else:
                sheet = (
                    None
                    if geometry.sheet is None
                    else _axial_mean(geometry.sheet[u], geometry.sheet[v])
                )
                normal = (
                    None
                    if geometry.normal is None
                    else _axial_mean(geometry.normal[u], geometry.normal[v])
                )
                basis = _orthonormal_basis(fibre, sheet, normal, delta)
                local = basis.T @ delta
                cost = float(
                    np.sqrt(
                        (local[0] / fibre_speed) ** 2
                        + (local[1] / sheet_speed) ** 2
                        + (local[2] / normal_speed) ** 2
                    )
                ) / tissue
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError("Non-finite propagation cost encountered")
        adjacency[u].append((v, cost))
        adjacency[v].append((u, cost))

    distance = np.full(geometry.n_nodes, np.inf, dtype=float)
    heap: list[tuple[float, int]] = []
    for node, time in zip(roots.nodes.tolist(), roots.activation_ms.tolist(), strict=True):
        if time < distance[node]:
            distance[node] = float(time)
            heappush(heap, (float(time), int(node)))

    while heap:
        current, u = heappop(heap)
        if current > distance[u] + 1e-12:
            continue
        for v, cost in adjacency[u]:
            candidate = current + cost
            if candidate + 1e-12 < distance[v]:
                distance[v] = candidate
                heappush(heap, (candidate, v))

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
        edge_count=len(edges),
        parameters=used,
        root_schedule=roots,
        warnings=tuple(warnings),
    )
