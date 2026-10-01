from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import EPGeometry


@dataclass(frozen=True)
class RootSchedule:
    nodes: np.ndarray
    activation_ms: np.ndarray
    method: str

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=int).reshape(-1)
        activation = np.asarray(self.activation_ms, dtype=float).reshape(-1)
        if len(nodes) == 0 or len(nodes) != len(activation):
            raise ValueError("Root schedule requires equally sized non-empty node/time arrays")
        if len(np.unique(nodes)) != len(nodes):
            raise ValueError("Root schedule contains duplicate nodes")
        if not np.isfinite(activation).all():
            raise ValueError("Root activation times must be finite")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "activation_ms", activation)


def _farthest_point_roots(geometry: EPGeometry, count: int) -> np.ndarray:
    candidates = geometry.endocardial_nodes
    if candidates is None or len(candidates) == 0:
        raise ValueError(
            "Automatic root selection requires geometry.endocardial_nodes; "
            "otherwise provide settings.root_nodes explicitly"
        )
    count = max(1, min(int(count), len(candidates)))
    xyz = geometry.node_xyz_cm[candidates]
    center = xyz.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(xyz - center, axis=1)))]
    distance = np.linalg.norm(xyz - xyz[selected[0]], axis=1)
    while len(selected) < count:
        nxt = int(np.argmax(distance))
        selected.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(xyz - xyz[nxt], axis=1))
    return candidates[np.asarray(selected, dtype=int)]


def resolve_root_schedule(
    geometry: EPGeometry,
    settings: dict[str, Any],
    parameters: dict[str, float],
) -> RootSchedule:
    raw_nodes = settings.get("root_nodes")
    method = "explicit"
    if raw_nodes is None:
        if geometry.root_nodes:
            raw_nodes = list(geometry.root_nodes)
            method = "geometry"
        elif settings.get("auto_root_count"):
            raw_nodes = _farthest_point_roots(geometry, int(settings["auto_root_count"])).tolist()
            method = "endocardial-farthest-point-heuristic"
        else:
            raise ValueError(
                "No ventricular activation roots supplied. Provide settings.root_nodes, "
                "embed root_nodes in the EP geometry, or use auto_root_count with an endocardial mask."
            )
    nodes = np.asarray(raw_nodes, dtype=int).reshape(-1)
    if np.any(nodes < 0) or np.any(nodes >= geometry.n_nodes):
        raise ValueError("root_nodes contain out-of-range node indices")

    raw_times = settings.get("root_activation_ms")
    if raw_times is None:
        times = np.zeros(len(nodes), dtype=float)
    elif isinstance(raw_times, dict):
        times = np.asarray([float(raw_times.get(str(int(node)), raw_times.get(int(node), 0.0))) for node in nodes])
    else:
        times = np.asarray(raw_times, dtype=float).reshape(-1)
        if len(times) != len(nodes):
            raise ValueError("root_activation_ms must match root_nodes")

    distances = settings.get("purkinje_root_distance_cm")
    if distances is not None:
        distance = np.asarray(distances, dtype=float).reshape(-1)
        if len(distance) != len(nodes) or np.any(distance < 0) or not np.isfinite(distance).all():
            raise ValueError("purkinje_root_distance_cm must contain one finite non-negative value per root")
        speed = float(parameters.get("purkinje_speed", parameters.get("purkinje_speed_cm_per_ms", 0.30)))
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("purkinje_speed must be positive")
        times = times + distance / speed
        method += "+purkinje-distance"

    times = times - float(np.min(times))
    return RootSchedule(nodes=nodes, activation_ms=times, method=method)
