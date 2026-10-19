"""Deterministic geometry staging for the Camps QRS research workflow."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .camps_workspace import CampsWorkspaceManifest
from .provenance import file_sha256, uri_to_path, verify_file_sha256

_COORDINATE_SCALE_TO_CM = {"mm": 0.1, "cm": 1.0, "m": 100.0}
_COORDINATE_FIELDS = {"node_xyz", "electrode_xyz"}


class CampsStagingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    workspace_manifest_fingerprint: str
    staging_root: str
    data_path: str
    results_path: str
    path_mapping_json: str
    upstream_relative_paths: dict[str, str]
    staged_sha256: dict[str, str]
    source_sha256: dict[str, str]
    coordinate_conversion: dict[str, Any] = Field(default_factory=dict)


def _load_coordinate_matrix(path: Path, *, field_name: str) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{field_name} must be an N x 3 CSV matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} contains non-finite coordinates")
    return values


def materialize_camps_geometry_workspace(
    manifest: CampsWorkspaceManifest | dict[str, Any],
    *,
    staging_root: str | Path,
) -> CampsStagingResult:
    """Materialize hash-verified geometry in the exact audited upstream layout.

    Only geometry is staged here. Clinical ECG parsing/resampling remains a
    separate measurement-preprocessing responsibility.
    """
    selected = (
        manifest
        if isinstance(manifest, CampsWorkspaceManifest)
        else CampsWorkspaceManifest.model_validate(manifest)
    )
    root = Path(staging_root).expanduser().resolve()
    data_root = (root / "data").resolve()
    results_root = (root / "results").resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    relative_paths = selected.upstream_relative_paths()
    source_sha256: dict[str, str] = {}
    staged_sha256: dict[str, str] = {}
    scale = _COORDINATE_SCALE_TO_CM[selected.coordinate_unit]

    for field_name, relative_path in relative_paths.items():
        artifact = getattr(selected, field_name)
        source = uri_to_path(artifact.uri)
        if not source.is_file():
            raise FileNotFoundError(f"Camps workspace source is missing: {source}")
        verify_file_sha256(source, artifact.sha256)
        source_sha256[field_name] = file_sha256(source)

        destination = (data_root / relative_path).resolve()
        if data_root not in destination.parents:
            raise ValueError("Camps workspace path escaped staging data root")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if field_name in _COORDINATE_FIELDS:
            values = _load_coordinate_matrix(source, field_name=field_name)
            np.savetxt(destination, values * scale, delimiter=",", fmt="%.17g")
        else:
            shutil.copyfile(source, destination)
        staged_sha256[field_name] = file_sha256(destination)

    # Upstream concatenates strings rather than using pathlib; keep trailing
    # separators in the generated config to preserve the audited behavior.
    data_path = str(data_root) + "/"
    results_path = str(results_root) + "/"
    path_mapping = {"data_path": data_path, "results_path": results_path}
    path_mapping_json = json.dumps(path_mapping, sort_keys=True, separators=(",", ":"))

    return CampsStagingResult(
        workspace_manifest_fingerprint=selected.fingerprint,
        staging_root=str(root),
        data_path=data_path,
        results_path=results_path,
        path_mapping_json=path_mapping_json,
        upstream_relative_paths=relative_paths,
        staged_sha256=staged_sha256,
        source_sha256=source_sha256,
        coordinate_conversion={
            "source_unit": selected.coordinate_unit,
            "target_unit": "cm",
            "scale": scale,
            "converted_fields": sorted(_COORDINATE_FIELDS),
        },
    )
