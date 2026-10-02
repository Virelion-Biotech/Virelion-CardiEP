from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import EPGeometry


def _integer_nodes(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.int64, copy=False).reshape(-1)
    numeric = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer node indices")
    return numeric.astype(np.int64)


def _positive_integer(value: Any, name: str) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be one positive integer")
    numeric = float(raw)
    if not np.isfinite(numeric) or numeric <= 0 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be one positive integer")
    return int(numeric)


@dataclass(frozen=True)
class RootSchedule:
    nodes: np.ndarray
    activation_ms: np.ndarray
    method: str

    def __post_init__(self) -> None:
        nodes = _integer_nodes(self.nodes, "Root schedule nodes")
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
    count = min(_positive_integer(count, "auto_root_count"), len(candidates))
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
        elif "auto_root_count" in settings:
            raw_nodes = _farthest_point_roots(
                geometry,
                _positive_integer(settings["auto_root_count"], "auto_root_count"),
            ).tolist()
            method = "endocardial-farthest-point-heuristic"
        else:
            raise ValueError(
                "No ventricular activation roots supplied. Provide settings.root_nodes, "
                "embed root_nodes in the EP geometry, or use auto_root_count with an endocardial mask."
            )

    nodes = _integer_nodes(raw_nodes, "root_nodes")
    if len(nodes) == 0:
        raise ValueError("root_nodes must contain at least one node")
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("root_nodes must be unique")
    if np.any(nodes < 0) or np.any(nodes >= geometry.n_nodes):
        raise ValueError("root_nodes contain out-of-range node indices")

    raw_times = settings.get("root_activation_ms")
    if raw_times is None:
        times = np.zeros(len(nodes), dtype=float)
    elif isinstance(raw_times, dict):
        times_list: list[float] = []
        missing: list[int] = []
        for node in nodes:
            int_node = int(node)
            if str(int_node) in raw_times:
                value = raw_times[str(int_node)]
            elif int_node in raw_times:
                value = raw_times[int_node]
            else:
                missing.append(int_node)
                continue
            times_list.append(float(value))
        if missing:
            raise ValueError(
                f"root_activation_ms mapping is missing root nodes: {missing}"
            )
        times = np.asarray(times_list, dtype=float)
    else:
        times = np.asarray(raw_times, dtype=float).reshape(-1)
        if len(times) != len(nodes):
            raise ValueError("root_activation_ms must match root_nodes")
    if not np.isfinite(times).all():
        raise ValueError("root_activation_ms must contain only finite values")

    distances = settings.get("purkinje_root_distance_cm")
    if distances is not None:
        distance = np.asarray(distances, dtype=float).reshape(-1)
        if len(distance) != len(nodes) or np.any(distance < 0) or not np.isfinite(distance).all():
            raise ValueError(
                "purkinje_root_distance_cm must contain one finite non-negative value per root"
            )
        speed = float(
            parameters.get(
                "purkinje_speed",
                parameters.get("purkinje_speed_cm_per_ms", 0.30),
            )
        )
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("purkinje_speed must be positive and finite")
        times = times + distance / speed
        method += "+purkinje-distance"

    times = times - float(np.min(times))
    return RootSchedule(nodes=nodes, activation_ms=times, method=method)
