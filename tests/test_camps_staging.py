from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardiep.camps_staging import materialize_camps_geometry_workspace
from cardiep.camps_workspace import CampsWorkspaceManifest
from cardiep.models import ArtifactRef
from cardiep.provenance import file_sha256, uri_to_path


def _write(path: Path, values) -> ArtifactRef:
    array = np.asarray(values)
    np.savetxt(path, array, delimiter=",", fmt="%.17g")
    return ArtifactRef(
        artifact_id=path.stem,
        kind="placeholder",
        uri=path.resolve().as_uri(),
        sha256=file_sha256(path),
        coordinate_frame="model-mm",
    )


def _manifest(tmp_path: Path) -> CampsWorkspaceManifest:
    def artifact(filename: str, kind: str, values, *, frame: str | None = "model-mm") -> ArtifactRef:
        item = _write(tmp_path / filename, values)
        return item.model_copy(update={"kind": kind, "coordinate_frame": frame})

    return CampsWorkspaceManifest(
        subject_id="P1",
        source_bundle_fingerprint="b" * 64,
        coordinate_frame="model-mm",
        coordinate_unit="mm",
        index_base=0,
        node_xyz=artifact("xyz.csv", "node_xyz", [[10.0, 20.0, 30.0], [0.0, 5.0, 10.0]]),
        tetra=artifact("tetra.csv", "tetra_connectivity", [[0, 1, 1, 0]], frame=None),
        lvendo_nodes=artifact("lv.csv", "lvendo_nodes", [0], frame=None),
        rvendo_nodes=artifact("rv.csv", "rvendo_nodes", [1], frame=None),
        fibre=artifact("fibre.csv", "fibre_field", [[1, 0, 0], [1, 0, 0]]),
        sheet=artifact("sheet.csv", "sheet_field", [[0, 1, 0], [0, 1, 0]]),
        normal=artifact("normal.csv", "normal_field", [[0, 0, 1], [0, 0, 1]]),
        material=artifact("material.csv", "tetra_material", [1], frame=None),
        vc_ab_cut=artifact("ab.csv", "ventricular_coordinate_ab_cut", [0.2, 0.8]),
        vc_rt=artifact("rt.csv", "ventricular_coordinate_rt", [0.1, 0.9]),
        electrode_xyz=artifact("electrodes.csv", "electrode_xyz", [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]),
    )


def test_staging_materializes_exact_layout_and_converts_mm_to_cm(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    result = materialize_camps_geometry_workspace(
        manifest,
        staging_root=tmp_path / "stage",
    )
    xyz_path = Path(result.data_path) / result.upstream_relative_paths["node_xyz"]
    electrode_path = Path(result.data_path) / result.upstream_relative_paths["electrode_xyz"]
    tetra_path = Path(result.data_path) / result.upstream_relative_paths["tetra"]

    assert np.allclose(
        np.loadtxt(xyz_path, delimiter=","),
        np.array([[1.0, 2.0, 3.0], [0.0, 0.5, 1.0]]),
    )
    assert np.allclose(
        np.loadtxt(electrode_path, delimiter=","),
        np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
    )
    assert np.array_equal(
        np.loadtxt(tetra_path, delimiter=","),
        np.array([0.0, 1.0, 1.0, 0.0]),
    )
    assert result.coordinate_conversion == {
        "source_unit": "mm",
        "target_unit": "cm",
        "scale": 0.1,
        "converted_fields": ["electrode_xyz", "node_xyz"],
    }
    mapping = json.loads(result.path_mapping_json)
    assert mapping["data_path"].endswith("/data/")
    assert mapping["results_path"].endswith("/results/")


def test_staging_detects_source_hash_change(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source = uri_to_path(manifest.node_xyz.uri)
    source.write_text("999,999,999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        materialize_camps_geometry_workspace(manifest, staging_root=tmp_path / "stage")


def test_staging_rejects_non_xyz_coordinate_matrix(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    bad_path = tmp_path / "bad-xyz.csv"
    bad = _write(bad_path, [[1.0, 2.0], [3.0, 4.0]])
    bad = bad.model_copy(update={"kind": "node_xyz"})
    manifest = manifest.model_copy(update={"node_xyz": bad})
    with pytest.raises(ValueError, match="N x 3"):
        materialize_camps_geometry_workspace(manifest, staging_root=tmp_path / "stage")
