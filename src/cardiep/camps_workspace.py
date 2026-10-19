"""Typed staging contract for the Camps Cardiac-Digital-Twin QRS workflow.

The upstream research scripts load a directory of separate CSV fields rather
than one general-purpose mesh object.  This module makes that implicit file
layout explicit, hash-pinned, and independent of filenames so an external
wrapper can materialize the exact upstream workspace reproducibly.
"""
from __future__ import annotations

import string
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import ArtifactRef
from .provenance import sha256_json

_REQUIRED_KINDS = {
    "node_xyz": "node_xyz",
    "tetra": "tetra_connectivity",
    "lvendo_nodes": "lvendo_nodes",
    "rvendo_nodes": "rvendo_nodes",
    "fibre": "fibre_field",
    "sheet": "sheet_field",
    "normal": "normal_field",
    "material": "tetra_material",
    "vc_ab_cut": "ventricular_coordinate_ab_cut",
    "vc_rt": "ventricular_coordinate_rt",
    "electrode_xyz": "electrode_xyz",
}


def _is_sha256(value: str | None) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


class CampsWorkspaceManifest(BaseModel):
    """All geometry fields required by upstream ``personalise_to_QRS.py`` v1."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    subject_id: str
    source_bundle_fingerprint: str = Field(min_length=64, max_length=64)
    coordinate_frame: str
    coordinate_unit: Literal["mm", "cm", "m"]
    resolution: Literal["coarse"] = "coarse"
    index_base: Literal[0, 1]
    node_xyz: ArtifactRef
    tetra: ArtifactRef
    lvendo_nodes: ArtifactRef
    rvendo_nodes: ArtifactRef
    fibre: ArtifactRef
    sheet: ArtifactRef
    normal: ArtifactRef
    material: ArtifactRef
    vc_ab_cut: ArtifactRef
    vc_rt: ArtifactRef
    electrode_xyz: ArtifactRef
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_workspace(self) -> CampsWorkspaceManifest:
        if not _is_sha256(self.source_bundle_fingerprint):
            raise ValueError("source_bundle_fingerprint must be SHA-256 hex")

        artifact_ids: list[str] = []
        for field_name, expected_kind in _REQUIRED_KINDS.items():
            artifact = getattr(self, field_name)
            artifact_ids.append(artifact.artifact_id)
            if artifact.kind != expected_kind:
                raise ValueError(
                    f"workspace field {field_name!r} must use kind {expected_kind!r}; "
                    f"got {artifact.kind!r}"
                )
            if not _is_sha256(artifact.sha256):
                raise ValueError(f"workspace artifact {field_name!r} must be SHA-256 pinned")

        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("workspace artifacts must have unique artifact_id values")

        for field_name in ("node_xyz", "fibre", "sheet", "normal", "vc_ab_cut", "vc_rt", "electrode_xyz"):
            artifact = getattr(self, field_name)
            if artifact.coordinate_frame not in {None, self.coordinate_frame}:
                raise ValueError(
                    f"workspace artifact {field_name!r} coordinate frame disagrees with manifest"
                )
        return self

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    def upstream_relative_paths(self) -> dict[str, str]:
        """Return the exact filenames expected by the pinned upstream QRS loader."""
        subject = self.subject_id
        prefix = f"{subject}_{self.resolution}"
        geometry_root = f"geometric_data/{subject}/{prefix}"
        return {
            "node_xyz": f"{geometry_root}/{prefix}_xyz.csv",
            "tetra": f"{geometry_root}/{prefix}_tetra.csv",
            "lvendo_nodes": f"{geometry_root}/{prefix}_boundarynodefield_lvendo.csv",
            "rvendo_nodes": f"{geometry_root}/{prefix}_boundarynodefield_rvendo.csv",
            "fibre": f"{geometry_root}/{prefix}_nodefield_fibre.csv",
            "sheet": f"{geometry_root}/{prefix}_nodefield_sheet.csv",
            "normal": f"{geometry_root}/{prefix}_nodefield_normal.csv",
            "material": f"{geometry_root}/{prefix}_material_tetra.csv",
            "vc_ab_cut": f"{geometry_root}/{prefix}_nodefield_ab_cut.csv",
            "vc_rt": f"{geometry_root}/{prefix}_nodefield_rt.csv",
            "electrode_xyz": f"geometric_data/{subject}/{subject}_electrode_xyz.csv",
        }
