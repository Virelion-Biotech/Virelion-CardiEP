from __future__ import annotations

import pytest
from pydantic import ValidationError

from cardiep.camps_workspace import CampsWorkspaceManifest
from cardiep.models import ArtifactRef


def _artifact(name: str, kind: str, *, frame: str | None = "model-mm", sha: str = "a" * 64) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=name,
        kind=kind,
        uri=f"file:///{name}.csv",
        sha256=sha,
        coordinate_frame=frame,
    )


def _manifest(**updates) -> CampsWorkspaceManifest:
    payload = {
        "subject_id": "P1",
        "source_bundle_fingerprint": "b" * 64,
        "coordinate_frame": "model-mm",
        "coordinate_unit": "mm",
        "index_base": 0,
        "node_xyz": _artifact("xyz", "node_xyz"),
        "tetra": _artifact("tetra", "tetra_connectivity", frame=None),
        "lvendo_nodes": _artifact("lvendo", "lvendo_nodes", frame=None),
        "rvendo_nodes": _artifact("rvendo", "rvendo_nodes", frame=None),
        "fibre": _artifact("fibre", "fibre_field"),
        "sheet": _artifact("sheet", "sheet_field"),
        "normal": _artifact("normal", "normal_field"),
        "material": _artifact("material", "tetra_material", frame=None),
        "vc_ab_cut": _artifact("ab-cut", "ventricular_coordinate_ab_cut"),
        "vc_rt": _artifact("rt", "ventricular_coordinate_rt"),
        "electrode_xyz": _artifact("electrodes", "electrode_xyz"),
    }
    payload.update(updates)
    return CampsWorkspaceManifest(**payload)


def test_workspace_produces_exact_pinned_upstream_qrs_paths() -> None:
    manifest = _manifest()
    paths = manifest.upstream_relative_paths()
    assert paths["node_xyz"] == "geometric_data/P1/P1_coarse/P1_coarse_xyz.csv"
    assert paths["tetra"] == "geometric_data/P1/P1_coarse/P1_coarse_tetra.csv"
    assert paths["lvendo_nodes"].endswith("P1_coarse_boundarynodefield_lvendo.csv")
    assert paths["rvendo_nodes"].endswith("P1_coarse_boundarynodefield_rvendo.csv")
    assert paths["fibre"].endswith("P1_coarse_nodefield_fibre.csv")
    assert paths["sheet"].endswith("P1_coarse_nodefield_sheet.csv")
    assert paths["normal"].endswith("P1_coarse_nodefield_normal.csv")
    assert paths["material"].endswith("P1_coarse_material_tetra.csv")
    assert paths["vc_ab_cut"].endswith("P1_coarse_nodefield_ab_cut.csv")
    assert paths["vc_rt"].endswith("P1_coarse_nodefield_rt.csv")
    assert paths["electrode_xyz"] == "geometric_data/P1/P1_electrode_xyz.csv"
    assert len(manifest.fingerprint) == 64


def test_workspace_rejects_wrong_semantic_kind() -> None:
    with pytest.raises(ValidationError, match="must use kind 'fibre_field'"):
        _manifest(fibre=_artifact("fibre", "sheet_field"))


def test_workspace_rejects_unpinned_artifact() -> None:
    bad = _artifact("rt", "ventricular_coordinate_rt")
    bad = bad.model_copy(update={"sha256": None})
    with pytest.raises(ValidationError, match="must be SHA-256 pinned"):
        _manifest(vc_rt=bad)


def test_workspace_rejects_cross_frame_vector_field() -> None:
    with pytest.raises(ValidationError, match="coordinate frame disagrees"):
        _manifest(fibre=_artifact("fibre", "fibre_field", frame="other-frame"))


def test_workspace_accepts_one_based_upstream_indices_explicitly() -> None:
    manifest = _manifest(index_base=1)
    assert manifest.index_base == 1
