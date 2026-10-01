from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import ArtifactRef
from .provenance import uri_to_path

_UNIT_TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0}


def _unit_scale(unit: str) -> float:
    key = str(unit).strip().lower()
    if key not in _UNIT_TO_CM:
        raise ValueError("Geometry units must be explicitly one of: mm, cm, m")
    return _UNIT_TO_CM[key]


def _normalize_vectors(values: np.ndarray | None, name: str, n: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n, 3):
        raise ValueError(f"{name} must have shape ({n}, 3)")
    norm = np.linalg.norm(arr, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError(f"{name} contains non-finite or zero vectors")
    return arr / norm[:, None]


def _load_array(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".csv", ".txt"}:
        return np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    raise ValueError(f"Unsupported auxiliary anatomy artifact format: {path.suffix}")


@dataclass(frozen=True)
class EPGeometry:
    node_xyz_cm: np.ndarray
    tetrahedra: np.ndarray
    fibre: np.ndarray | None = None
    sheet: np.ndarray | None = None
    normal: np.ndarray | None = None
    scar_labels: np.ndarray | None = None
    ventricular_coordinates: dict[str, np.ndarray] = field(default_factory=dict)
    electrodes_cm: dict[str, np.ndarray] = field(default_factory=dict)
    root_nodes: tuple[int, ...] = ()
    endocardial_nodes: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xyz = np.asarray(self.node_xyz_cm, dtype=float)
        tetra = np.asarray(self.tetrahedra, dtype=int)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4:
            raise ValueError("node_xyz_cm must have shape (N, 3) with N >= 4")
        if not np.isfinite(xyz).all():
            raise ValueError("node coordinates must be finite")
        if tetra.ndim != 2 or tetra.shape[1] != 4 or len(tetra) < 1:
            raise ValueError("tetrahedra must have shape (M, 4) with M >= 1")
        if tetra.min() < 0 or tetra.max() >= len(xyz):
            raise ValueError("tetrahedra contain out-of-range node indices")
        object.__setattr__(self, "node_xyz_cm", xyz)
        object.__setattr__(self, "tetrahedra", tetra)
        object.__setattr__(self, "fibre", _normalize_vectors(self.fibre, "fibre", len(xyz)))
        object.__setattr__(self, "sheet", _normalize_vectors(self.sheet, "sheet", len(xyz)))
        object.__setattr__(self, "normal", _normalize_vectors(self.normal, "normal", len(xyz)))
        if self.scar_labels is not None:
            labels = np.asarray(self.scar_labels, dtype=int).reshape(-1)
            if len(labels) != len(xyz) or np.any(~np.isin(labels, [0, 1, 2])):
                raise ValueError("scar_labels must be node-wise labels in {0,1,2}")
            object.__setattr__(self, "scar_labels", labels)
        vc: dict[str, np.ndarray] = {}
        for name, values in self.ventricular_coordinates.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if len(arr) != len(xyz) or not np.isfinite(arr).all():
                raise ValueError(f"Ventricular coordinate {name!r} must contain N finite values")
            vc[str(name)] = arr
        object.__setattr__(self, "ventricular_coordinates", vc)
        electrodes = {}
        for name, point in self.electrodes_cm.items():
            arr = np.asarray(point, dtype=float).reshape(-1)
            if arr.shape != (3,) or not np.isfinite(arr).all():
                raise ValueError(f"Electrode {name!r} must be a finite xyz coordinate")
            electrodes[str(name)] = arr
        object.__setattr__(self, "electrodes_cm", electrodes)
        roots = tuple(int(x) for x in self.root_nodes)
        if any(x < 0 or x >= len(xyz) for x in roots):
            raise ValueError("root_nodes contain out-of-range indices")
        object.__setattr__(self, "root_nodes", roots)
        if self.endocardial_nodes is not None:
            nodes = np.unique(np.asarray(self.endocardial_nodes, dtype=int).reshape(-1))
            if nodes.size and (nodes.min() < 0 or nodes.max() >= len(xyz)):
                raise ValueError("endocardial_nodes contain out-of-range indices")
            object.__setattr__(self, "endocardial_nodes", nodes)

    @property
    def n_nodes(self) -> int:
        return len(self.node_xyz_cm)

    @property
    def edges(self) -> np.ndarray:
        t = self.tetrahedra
        pairs = np.vstack(
            [
                t[:, [0, 1]], t[:, [0, 2]], t[:, [0, 3]],
                t[:, [1, 2]], t[:, [1, 3]], t[:, [2, 3]],
            ]
        )
        return np.unique(np.sort(pairs, axis=1), axis=0)

    def summary(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_tetrahedra": len(self.tetrahedra),
            "n_edges": len(self.edges),
            "has_fibre": self.fibre is not None,
            "has_sheet": self.sheet is not None,
            "has_normal": self.normal is not None,
            "has_scar": self.scar_labels is not None,
            "ventricular_coordinates": sorted(self.ventricular_coordinates),
            "electrodes": sorted(self.electrodes_cm),
            "root_nodes": list(self.root_nodes),
            "n_endocardial_nodes": 0 if self.endocardial_nodes is None else len(self.endocardial_nodes),
        }


def _payload_to_geometry(payload: dict[str, Any], *, fallback_unit: str | None = None) -> EPGeometry:
    raw = payload.get("geometry", payload)
    unit = raw.get("units") or raw.get("coordinate_unit") or fallback_unit
    if not unit:
        raise ValueError(
            "Native EP geometry requires explicit coordinate units (mm, cm, or m); "
            "set geometry.units or EPSimulationRequest.settings.geometry_unit"
        )
    scale = _unit_scale(str(unit))
    electrodes = raw.get("electrodes") or raw.get("electrode_xyz") or {}
    if isinstance(electrodes, list):
        names = raw.get("electrode_names")
        if not isinstance(names, list) or len(names) != len(electrodes):
            raise ValueError("electrode_names must accompany list-form electrodes")
        electrodes = dict(zip(names, electrodes, strict=True))
    return EPGeometry(
        node_xyz_cm=np.asarray(raw["node_xyz"], dtype=float) * scale,
        tetrahedra=np.asarray(raw["tetrahedra"], dtype=int),
        fibre=None if raw.get("fibre", raw.get("fiber")) is None else np.asarray(raw.get("fibre", raw.get("fiber")), dtype=float),
        sheet=None if raw.get("sheet") is None else np.asarray(raw["sheet"], dtype=float),
        normal=None if raw.get("normal", raw.get("sheet_normal")) is None else np.asarray(raw.get("normal", raw.get("sheet_normal")), dtype=float),
        scar_labels=None if raw.get("scar_labels") is None else np.asarray(raw["scar_labels"], dtype=int),
        ventricular_coordinates={
            str(key): np.asarray(value, dtype=float)
            for key, value in dict(raw.get("ventricular_coordinates") or {}).items()
        },
        electrodes_cm={str(key): np.asarray(value, dtype=float) * scale for key, value in dict(electrodes).items()},
        root_nodes=tuple(int(x) for x in raw.get("root_nodes", ())),
        endocardial_nodes=None if raw.get("endocardial_nodes") is None else np.asarray(raw["endocardial_nodes"], dtype=int),
        metadata={"source_coordinate_unit": str(unit), **dict(raw.get("metadata") or {})},
    )


def _meshio_geometry(path: Path, *, unit: str) -> EPGeometry:
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "VTK/VTU/Gmsh anatomy loading requires the optional meshio extra: "
            "pip install 'virelion-cardiep[io]'"
        ) from exc
    mesh = meshio.read(path)
    tetra = None
    for block in mesh.cells:
        if block.type in {"tetra", "tetra10"}:
            tetra = np.asarray(block.data[:, :4], dtype=int)
            break
    if tetra is None:
        raise ValueError(f"No tetrahedral cells found in {path}")
    point_data = {str(k).lower(): np.asarray(v) for k, v in mesh.point_data.items()}

    def pick(*names: str):
        for name in names:
            if name.lower() in point_data:
                return point_data[name.lower()]
        return None

    vc = {}
    for name in ("ab", "tm", "rt", "tv", "aprt", "rvlv"):
        if name in point_data:
            vc[name] = point_data[name].reshape(-1)
    return EPGeometry(
        node_xyz_cm=np.asarray(mesh.points[:, :3], dtype=float) * _unit_scale(unit),
        tetrahedra=tetra,
        fibre=pick("fibre", "fiber", "f0"),
        sheet=pick("sheet", "s0"),
        normal=pick("normal", "sheet_normal", "n0"),
        scar_labels=pick("scar", "scar_labels"),
        ventricular_coordinates=vc,
        metadata={"source_coordinate_unit": unit, "mesh_path": str(path)},
    )


def _resolve_bundle_artifact(bundle: dict[str, Any], kind: str) -> dict[str, Any] | None:
    candidates = [item for item in bundle.get("artifacts", []) if item.get("kind") == kind]
    return candidates[-1] if candidates else None


def _load_bundle(path: Path, bundle: dict[str, Any], *, fallback_unit: str | None) -> EPGeometry:
    volume = _resolve_bundle_artifact(bundle, "volume_mesh")
    if volume is None:
        raise ValueError("CardiAnatomy bundle does not contain a volume_mesh artifact")
    frame_unit = None
    frame_id = volume.get("frame_id")
    if frame_id:
        for frame in bundle.get("frames", []):
            if frame.get("frame_id") == frame_id:
                frame_unit = frame.get("units")
                break
    unit = volume.get("metadata", {}).get("units") or frame_unit or fallback_unit
    if not unit:
        raise ValueError("Could not determine volume-mesh coordinate units from the AnatomyBundle")
    volume_path = uri_to_path(str(volume["uri"]), relative_to=path.parent)
    suffix = volume_path.suffix.lower()
    if suffix in {".vtk", ".vtu", ".msh"}:
        base = _meshio_geometry(volume_path, unit=str(unit))
    elif suffix == ".npz":
        data = _load_array(volume_path)
        base = _payload_to_geometry({**data, "units": unit}, fallback_unit=str(unit))
    else:
        raw = _load_array(volume_path)
        if not isinstance(raw, dict):
            raise ValueError("Volume mesh artifact must contain a mapping")
        base = _payload_to_geometry(raw, fallback_unit=str(unit))

    kwargs: dict[str, Any] = {
        "node_xyz_cm": base.node_xyz_cm,
        "tetrahedra": base.tetrahedra,
        "fibre": base.fibre,
        "sheet": base.sheet,
        "normal": base.normal,
        "scar_labels": base.scar_labels,
        "ventricular_coordinates": dict(base.ventricular_coordinates),
        "electrodes_cm": dict(base.electrodes_cm),
        "root_nodes": base.root_nodes,
        "endocardial_nodes": base.endocardial_nodes,
        "metadata": {**base.metadata, "anatomy_bundle": str(path)},
    }

    fibre_art = _resolve_bundle_artifact(bundle, "fiber_field")
    if fibre_art is not None:
        aux = _load_array(uri_to_path(str(fibre_art["uri"]), relative_to=path.parent))
        if isinstance(aux, dict):
            kwargs["fibre"] = aux.get("fibre", aux.get("fiber", kwargs["fibre"]))
            kwargs["sheet"] = aux.get("sheet", kwargs["sheet"])
            kwargs["normal"] = aux.get("normal", aux.get("sheet_normal", kwargs["normal"]))
        else:
            kwargs["fibre"] = aux

    coord_art = _resolve_bundle_artifact(bundle, "coordinate_field")
    if coord_art is not None:
        aux = _load_array(uri_to_path(str(coord_art["uri"]), relative_to=path.parent))
        if isinstance(aux, dict):
            kwargs["ventricular_coordinates"].update(
                {str(key): np.asarray(value) for key, value in aux.items()}
            )

    scar_art = _resolve_bundle_artifact(bundle, "scar_map")
    if scar_art is not None:
        aux = _load_array(uri_to_path(str(scar_art["uri"]), relative_to=path.parent))
        if isinstance(aux, dict):
            aux = aux.get("scar_labels", aux.get("labels"))
        kwargs["scar_labels"] = aux

    return EPGeometry(**kwargs)


def load_ep_geometry(ref: ArtifactRef, settings: dict[str, Any] | None = None) -> EPGeometry:
    settings = dict(settings or {})
    path = uri_to_path(ref.uri)
    fallback_unit = settings.get("geometry_unit") or ref.metadata.get("geometry_unit")
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".vtk", ".vtu", ".msh"}:
        if not fallback_unit:
            raise ValueError("VTK/VTU/Gmsh anatomy requires settings.geometry_unit")
        geometry = _meshio_geometry(path, unit=str(fallback_unit))
    elif suffix == ".npz":
        data = _load_array(path)
        geometry = _payload_to_geometry(dict(data), fallback_unit=fallback_unit)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("contract_version") == "2.0.0" and "artifacts" in payload:
            geometry = _load_bundle(path, payload, fallback_unit=fallback_unit)
        elif isinstance(payload, dict):
            geometry = _payload_to_geometry(payload, fallback_unit=fallback_unit)
        else:
            raise ValueError("Geometry JSON must contain an object")
    else:
        raise ValueError(f"Unsupported anatomy artifact format: {path.suffix}")

    metadata = dict(ref.metadata)
    if metadata.get("root_nodes") and not geometry.root_nodes:
        geometry = EPGeometry(
            **{
                **geometry.__dict__,
                "root_nodes": tuple(int(x) for x in metadata["root_nodes"]),
            }
        )
    return geometry
