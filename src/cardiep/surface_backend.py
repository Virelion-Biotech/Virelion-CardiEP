from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_artifact
from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPParameterSet,
    EPSimulationRequest,
    EPSimulationResult,
)
from .provenance import sha256_json, uri_to_path, verify_file_sha256

SURFACE_BACKEND_NAME = "surface-eikonal-v1"


@dataclass(frozen=True)
class SurfaceGeometry:
    vertices_cm: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_cm, dtype=float)
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("Surface geometry vertices must have shape (N, 3)")
        if not np.isfinite(vertices).all():
            raise ValueError("Surface geometry vertices must be finite")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
            raise ValueError("Surface geometry triangles must have shape (M, 3)")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("Surface geometry contains out-of-range triangle indices")
        if np.any(np.sort(triangles, axis=1)[:, 1:] == np.sort(triangles, axis=1)[:, :-1]):
            raise ValueError("Surface geometry contains degenerate triangles")
        object.__setattr__(self, "vertices_cm", vertices)
        object.__setattr__(self, "triangles", triangles)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices_cm)


def _load_json_artifact(ref) -> dict[str, Any]:
    path = uri_to_path(ref.uri)
    if not path.is_file():
        raise FileNotFoundError(path)
    verify_file_sha256(path, ref.sha256)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Surface-Eikonal artifacts must contain JSON objects")
    return payload


def load_surface_geometry(ref) -> SurfaceGeometry:
    payload = _load_json_artifact(ref)
    if payload.get("schema_version") != "cardiep-surface-v1":
        raise ValueError("Surface anatomy must use schema_version cardiep-surface-v1")
    unit = str(payload.get("coordinate_unit", "")).lower()
    scale = {"cm": 1.0, "mm": 0.1, "m": 100.0}.get(unit)
    if scale is None:
        raise ValueError("Surface coordinate_unit must be one of cm, mm, or m")
    vertices = np.asarray(payload.get("vertices"), dtype=float) * scale
    triangles = np.asarray(payload.get("triangles"), dtype=np.int64)
    return SurfaceGeometry(vertices_cm=vertices, triangles=triangles)


def _adjacency(geometry: SurfaceGeometry) -> list[list[tuple[int, float]]]:
    neighbors: list[dict[int, float]] = [{} for _ in range(geometry.n_vertices)]
    for triangle in geometry.triangles:
        for left, right in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            distance = float(
                np.linalg.norm(geometry.vertices_cm[left] - geometry.vertices_cm[right])
            )
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError("Surface mesh edges must have positive finite length")
            previous = neighbors[left].get(right)
            if previous is None or distance < previous:
                neighbors[left][right] = distance
                neighbors[right][left] = distance
    return [sorted(items.items()) for items in neighbors]


def _distances(
    adjacency: list[list[tuple[int, float]]],
    root: int,
) -> np.ndarray:
    n = len(adjacency)
    if root < 0 or root >= n:
        raise ValueError("root_node is outside the surface mesh")
    values = np.full(n, np.inf, dtype=float)
    values[root] = 0.0
    queue: list[tuple[float, int]] = [(0.0, root)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != values[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < values[neighbor]:
                values[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return values


def _output_dir(subject_id: str, request_sha: str, settings: dict[str, Any]) -> Path:
    configured = settings.get("output_dir")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return (Path.cwd() / "cardiep_runs" / subject_id / request_sha[:16]).resolve()


def _speed(parameters: dict[str, float]) -> float:
    supplied = [
        key
        for key in ("isotropic_speed", "isotropic_speed_cm_per_ms")
        if key in parameters
    ]
    if len(supplied) != 1:
        raise ValueError(
            "surface-eikonal-v1 requires exactly one isotropic_speed alias"
        )
    value = float(parameters[supplied[0]])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Surface isotropic speed must be positive and finite")
    return value


def _activation_observations(
    request: EPCalibrationRequest,
    n_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[int] = []
    activation: list[float] = []
    for observation in request.observations:
        if observation.kind not in {"eam_activation", "activation_map"}:
            continue
        payload = _load_json_artifact(observation.artifact)
        if payload.get("schema_version") != "cardiep-eam-activation-v1":
            raise ValueError(
                "Surface calibration observations must use "
                "schema_version cardiep-eam-activation-v1"
            )
        raw_vertices = np.asarray(payload.get("vertex_indices"), dtype=float).reshape(-1)
        raw_activation = np.asarray(payload.get("activation_ms"), dtype=float).reshape(-1)
        if len(raw_vertices) != len(raw_activation) or len(raw_vertices) == 0:
            raise ValueError("EAM activation vertex/time arrays must be equally sized")
        if (
            not np.isfinite(raw_vertices).all()
            or not np.equal(raw_vertices, np.floor(raw_vertices)).all()
        ):
            raise ValueError("EAM activation vertex indices must be finite integers")
        indices = raw_vertices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= n_vertices):
            raise ValueError("EAM activation contains out-of-range surface vertices")
        if not np.isfinite(raw_activation).all():
            raise ValueError("EAM activation times must be finite")
        vertices.extend(indices.tolist())
        activation.extend(raw_activation.tolist())
    if len(vertices) < 2:
        raise ValueError(
            "surface-eikonal-v1 calibration requires at least two EAM activation points"
        )
    return np.asarray(vertices, dtype=np.int64), np.asarray(activation, dtype=float)


class SurfaceEikonalBackend:
    """Isotropic graph-Eikonal backend for electroanatomical surface meshes.

    This backend is intended for observable-level EAM validation when volumetric
    fibre-resolved anatomy is unavailable. It is not a volumetric EP solver.
    """

    name = SURFACE_BACKEND_NAME

    def available(self) -> bool:
        return True

    def _activation(
        self,
        *,
        anatomy_ref,
        parameters: dict[str, float],
        settings: dict[str, Any],
    ) -> tuple[SurfaceGeometry, np.ndarray, int, float, float]:
        geometry = load_surface_geometry(anatomy_ref)
        adjacency = _adjacency(geometry)
        if "root_node" not in settings:
            raise ValueError("surface-eikonal-v1 requires settings.root_node")
        root = int(settings["root_node"])
        speed = _speed(parameters)
        offset = float(parameters.get("activation_offset_ms", 0.0))
        if not math.isfinite(offset):
            raise ValueError("activation_offset_ms must be finite")
        distance = _distances(adjacency, root)
        if not np.isfinite(distance).all():
            raise ValueError("Surface mesh is disconnected from the selected root")
        activation = offset + distance / speed
        return geometry, activation, root, speed, offset

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        geometry, activation, root, speed, offset = self._activation(
            anatomy_ref=request.anatomy_ref,
            parameters=dict(request.parameters.values),
            settings=dict(request.settings),
        )
        request_json = request.model_dump(mode="json")
        request_sha = sha256_json(request_json)
        output_dir = _output_dir(request.subject_id, request_sha, request.settings)
        activation_ref = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-surface-activation-{request_sha[:12]}",
            kind="activation_map",
            payload={
                "schema_version": "cardiep-surface-activation-v1",
                "subject_id": request.subject_id,
                "units": "ms",
                "activation_ms": activation.tolist(),
                "root_node": root,
                "isotropic_speed_cm_per_ms": speed,
                "activation_offset_ms": offset,
            },
            metadata={
                "model": self.name,
                "n_vertices": geometry.n_vertices,
                "scientific_scope": "surface EAM observable model",
            },
        )
        summary_ref = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-surface-summary-{request_sha[:12]}",
            kind="ep_summary",
            payload={
                "schema_version": "cardiep-surface-summary-v1",
                "subject_id": request.subject_id,
                "backend": self.name,
                "root_node": root,
                "n_vertices": geometry.n_vertices,
                "n_triangles": len(geometry.triangles),
                "activation_min_ms": float(np.min(activation)),
                "activation_max_ms": float(np.max(activation)),
                "activation_span_ms": float(np.ptp(activation)),
            },
            metadata={"model": self.name},
        )
        return EPSimulationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=request.parameters,
            outputs=[activation_ref, summary_ref],
            validation_status="software_checked",
            warnings=[
                (
                    "surface-eikonal-v1 is isotropic and surface-only; it does not "
                    "represent fibre-resolved transmural ventricular propagation."
                )
            ],
            provenance={
                "engine": "Virelion-CardiEP",
                "backend": self.name,
                "request_sha256": request_sha,
                "anatomy_artifact_id": request.anatomy_ref.artifact_id,
                "root_node": root,
                "scientific_status": (
                    "research observable model for EAM surfaces; not clinically validated"
                ),
            },
        )

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        geometry = load_surface_geometry(request.anatomy_ref)
        adjacency = _adjacency(geometry)
        vertices, observed = _activation_observations(request, geometry.n_vertices)

        speed_key = None
        for candidate in ("isotropic_speed_cm_per_ms", "isotropic_speed"):
            if candidate in request.parameter_bounds:
                if speed_key is not None:
                    raise ValueError("Provide only one isotropic speed parameter bound")
                speed_key = candidate
        if speed_key is None:
            raise ValueError(
                "Surface calibration requires isotropic_speed_cm_per_ms parameter bounds"
            )
        low, high = request.parameter_bounds[speed_key]
        low_speed, high_speed = float(low), float(high)
        if not (0 < low_speed < high_speed and math.isfinite(high_speed)):
            raise ValueError("Surface speed bounds must satisfy 0 < lower < upper")

        raw_candidates = request.settings.get("root_candidates")
        if raw_candidates is None:
            candidates = sorted({int(item) for item in vertices})
        else:
            candidates = [int(item) for item in raw_candidates]
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("root_candidates must contain unique surface vertex indices")
        if any(item < 0 or item >= geometry.n_vertices for item in candidates):
            raise ValueError("root_candidates contain out-of-range surface vertices")

        best: dict[str, Any] | None = None
        for root in candidates:
            distance = _distances(adjacency, root)
            sampled = distance[vertices]
            if not np.isfinite(sampled).all():
                continue
            centered_distance = sampled - float(np.mean(sampled))
            centered_observed = observed - float(np.mean(observed))
            denominator = float(np.dot(centered_distance, centered_distance))
            if denominator <= 1e-18:
                inv_speed = 1.0 / (0.5 * (low_speed + high_speed))
            else:
                inv_speed = float(
                    np.dot(centered_distance, centered_observed) / denominator
                )
            inv_speed = float(
                np.clip(inv_speed, 1.0 / high_speed, 1.0 / low_speed)
            )
            speed = 1.0 / inv_speed
            offset = float(np.mean(observed - sampled * inv_speed))
            predicted = offset + sampled * inv_speed
            residual = predicted - observed
            rmse = float(np.sqrt(np.mean(residual**2)))
            candidate = {
                "root_node": root,
                "speed": speed,
                "offset": offset,
                "rmse": rmse,
                "mae": float(np.mean(np.abs(residual))),
                "max_abs": float(np.max(np.abs(residual))),
            }
            if best is None or (candidate["rmse"], root) < (best["rmse"], best["root_node"]):
                best = candidate
        if best is None:
            raise ValueError("No root candidate can reach all calibration vertices")

        parameters = EPParameterSet(
            values={
                "isotropic_speed_cm_per_ms": float(best["speed"]),
                "activation_offset_ms": float(best["offset"]),
            },
            units={
                "isotropic_speed_cm_per_ms": "cm/ms",
                "activation_offset_ms": "ms",
            },
            source="calibrated",
        )
        simulation_settings = dict(request.settings)
        simulation_settings.pop("root_candidates", None)
        simulation_settings["root_node"] = int(best["root_node"])
        simulated = self.simulate(
            EPSimulationRequest(
                subject_id=request.subject_id,
                anatomy_ref=request.anatomy_ref,
                backend=self.name,
                parameters=parameters,
                observations=request.observations,
                settings=simulation_settings,
            )
        )
        return EPCalibrationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=parameters,
            objective=float(best["rmse"]),
            converged=True,
            simulated=simulated,
            diagnostics={
                "method": "discrete-root-plus-bounded-linear-inverse-speed-fit",
                "root_node": int(best["root_node"]),
                "n_root_candidates": len(candidates),
                "n_calibration_points": len(vertices),
                "rmse_ms": float(best["rmse"]),
                "mae_ms": float(best["mae"]),
                "max_abs_ms": float(best["max_abs"]),
            },
            provenance={
                "engine": "Virelion-CardiEP",
                "backend": self.name,
                "scientific_status": (
                    "surface EAM calibration baseline; not volumetric or clinical validation"
                ),
            },
        )
