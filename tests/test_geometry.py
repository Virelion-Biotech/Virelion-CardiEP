import json
from pathlib import Path

import pytest

from cardiep import ArtifactRef, load_ep_geometry


def test_geometry_requires_explicit_coordinate_units(tmp_path: Path) -> None:
    path = tmp_path / "geometry.json"
    path.write_text(
        json.dumps(
            {
                "node_xyz": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "tetrahedra": [[0, 1, 2, 3]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit coordinate units"):
        load_ep_geometry(ArtifactRef(artifact_id="g", kind="mesh", uri=path.as_uri()))


def test_geometry_converts_mm_to_cm(tmp_path: Path) -> None:
    path = tmp_path / "geometry.json"
    path.write_text(
        json.dumps(
            {
                "units": "mm",
                "node_xyz": [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
                "tetrahedra": [[0, 1, 2, 3]],
            }
        ),
        encoding="utf-8",
    )
    geometry = load_ep_geometry(
        ArtifactRef(artifact_id="g", kind="mesh", uri=path.as_uri())
    )
    assert geometry.node_xyz_cm[1, 0] == pytest.approx(1.0)
