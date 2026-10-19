from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from mesh_patient import (
    OUT,
    ROOT,
    SUBJECT_ID,
    connected_tets,
    download,
    parse_tri_bytes,
    read_tetgen_ele,
    read_tetgen_node,
    sha256_bytes,
    sha256_file,
    signed_volume6,
    tet_boundary,
    undirected_edge_counts,
    write_off,
)

MATCH_TOLERANCE_MM = 1e-5


def nearest_with_distance(point: list[float], nodes: list[list[float]]) -> tuple[int, float]:
    best_index = -1
    best_sq = float("inf")
    for index, candidate in enumerate(nodes):
        distance_sq = sum(
            (float(point[axis]) - float(candidate[axis])) ** 2
            for axis in range(3)
        )
        if distance_sq < best_sq:
            best_sq = distance_sq
            best_index = index
    return best_index, math.sqrt(best_sq)


def exact_matches(
    source_points: list[list[float]],
    target_points: list[list[float]],
    *,
    tolerance_mm: float = MATCH_TOLERANCE_MM,
) -> tuple[list[int], int, float]:
    """Return target indices for source points that truly coincide geometrically.

    Closed cavity surfaces contain closure/cap vertices in addition to shared
    myocardial endocardial vertices. Unmatched closure vertices are deliberately
    excluded rather than snapped to the myocardium.
    """
    matched: set[int] = set()
    unmatched = 0
    maximum_matched_distance = 0.0
    for point in source_points:
        index, distance = nearest_with_distance(point, target_points)
        if distance <= tolerance_mm:
            matched.add(index)
            maximum_matched_distance = max(maximum_matched_distance, distance)
        else:
            unmatched += 1
    return sorted(matched), unmatched, maximum_matched_distance


def exact_output_index(point: list[float], nodes: list[list[float]]) -> int:
    index, distance = nearest_with_distance(point, nodes)
    if distance > MATCH_TOLERANCE_MM:
        raise ValueError(
            "Published ventricular boundary vertex was not preserved by TetGen "
            f"(nearest {distance:.6g} mm)"
        )
    return index


def main() -> None:
    required = [
        ROOT / "ventricular_surface.json",
        ROOT / "electrodes_120.json",
        ROOT / "manifest.json",
        ROOT / "split_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run preflight.py first; missing: " + ", ".join(missing))

    OUT.mkdir(parents=True, exist_ok=True)
    ventricular = json.loads(
        (ROOT / "ventricular_surface.json").read_text(encoding="utf-8")
    )
    vertices = ventricular["vertices"]
    triangles = ventricular["triangles"]
    if ventricular.get("coordinate_unit") != "mm":
        raise ValueError("Case 3 ventricular surface must remain in source millimetres")

    edge_counts = undirected_edge_counts(triangles)
    if not edge_counts or any(value != 2 for value in edge_counts.values()):
        bad = sum(value != 2 for value in edge_counts.values())
        raise ValueError(
            f"Published ventricular surface is not watertight/manifold: {bad} bad edges"
        )
    euler_characteristic = len(vertices) - len(edge_counts) + len(triangles)
    if euler_characteristic != 0:
        raise ValueError(
            "Expected the documented genus-1 ventricular surface "
            f"(Euler characteristic 0), got {euler_characteristic}"
        )

    # Identify only the exact intersection between the closed cavity surfaces
    # and the myocardial boundary. Cavity closure/cap vertices are not snapped.
    endocardial_sources: dict[str, dict] = {}
    endocardial_original_indices: set[int] = set()
    for role, filename in (
        ("left_endocardium", "case3_lendo.tri"),
        ("right_endocardium", "case3_rendo.tri"),
    ):
        payload, url = download(filename)
        cavity_vertices, cavity_triangles = parse_tri_bytes(payload)
        matched, unmatched, max_distance = exact_matches(cavity_vertices, vertices)
        if len(matched) < 20:
            raise ValueError(
                f"{role} shares only {len(matched)} exact vertices with the myocardial boundary; "
                "refusing an implausible endocardial mapping"
            )
        endocardial_original_indices.update(matched)
        endocardial_sources[role] = {
            "filename": filename,
            "retrieved_from": url,
            "sha256": sha256_bytes(payload),
            "closed_cavity_vertices": len(cavity_vertices),
            "closed_cavity_triangles": len(cavity_triangles),
            "matched_myocardial_boundary_vertices": len(matched),
            "unmatched_closure_or_nonmyocardial_vertices": unmatched,
            "max_exact_match_distance_mm": max_distance,
            "matching_tolerance_mm": MATCH_TOLERANCE_MM,
        }

    if len(endocardial_original_indices) < 50:
        raise ValueError(
            "Combined LV/RV endocardial intersection is implausibly small: "
            f"{len(endocardial_original_indices)} vertices"
        )

    off_path = OUT / "case3_ventricular_surface.off"
    write_off(off_path, vertices, triangles)
    process = subprocess.run(
        ["tetgen", "-pq1.5YQ", off_path.name],
        cwd=OUT,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            "TetGen constrained quality meshing failed:\n"
            + (process.stdout or "")
            + "\n"
            + (process.stderr or "")
        )

    node_path = OUT / "case3_ventricular_surface.1.node"
    ele_path = OUT / "case3_ventricular_surface.1.ele"
    if not node_path.is_file() or not ele_path.is_file():
        raise RuntimeError("TetGen did not emit expected .node/.ele output")
    nodes = read_tetgen_node(node_path)
    tets = read_tetgen_ele(ele_path, len(nodes))
    if not tets:
        raise ValueError("TetGen emitted no tetrahedra")

    original_to_output = [exact_output_index(point, nodes) for point in vertices]
    if len(set(original_to_output)) != len(vertices):
        raise ValueError(
            "Published ventricular vertices do not map one-to-one into TetGen output"
        )
    expected_boundary = {
        tuple(sorted(original_to_output[index] for index in tri))
        for tri in triangles
    }
    observed_boundary = tet_boundary(tets)
    if observed_boundary != expected_boundary:
        raise ValueError(
            "TetGen tetrahedral boundary differs from the published ventricular surface: "
            f"expected={len(expected_boundary)} observed={len(observed_boundary)} "
            f"missing={len(expected_boundary - observed_boundary)} "
            f"extra={len(observed_boundary - expected_boundary)}"
        )
    if not connected_tets(tets):
        raise ValueError("TetGen ventricular volume mesh is disconnected")

    volumes6 = [signed_volume6(nodes, tet) for tet in tets]
    absolute_volumes6 = [abs(value) for value in volumes6]
    if min(absolute_volumes6) <= 1e-9:
        raise ValueError("TetGen emitted zero/near-zero volume tetrahedra")
    oriented_tets: list[list[int]] = []
    for tet, volume6 in zip(tets, volumes6, strict=True):
        oriented_tets.append(
            tet if volume6 > 0 else [tet[0], tet[2], tet[1], tet[3]]
        )

    endocardial_nodes = sorted(
        {original_to_output[index] for index in endocardial_original_indices}
    )
    if len(endocardial_nodes) != len(endocardial_original_indices):
        raise ValueError("Endocardial boundary mapping lost unique source vertices")

    electrodes = json.loads(
        (ROOT / "electrodes_120.json").read_text(encoding="utf-8")
    )
    if electrodes.get("coordinate_frame") != "physionet-case3-native":
        raise ValueError("Electrode coordinate frame differs from ventricular surface")
    if len(electrodes.get("electrodes") or {}) != 120:
        raise ValueError("Expected all 120 measured electrode coordinates")

    ep_geometry = {
        "schema_version": "cardiep-physionet-case3-volume-v1",
        "subject_id": SUBJECT_ID,
        "coordinate_frame": "physionet-case3-native",
        "units": "mm",
        "node_xyz": nodes,
        "tetrahedra": oriented_tets,
        "endocardial_nodes": endocardial_nodes,
        "electrodes": electrodes["electrodes"],
        "metadata": {
            "volume_mesher": "TetGen",
            "tetgen_command": "tetgen -pq1.5YQ case3_ventricular_surface.off",
            "input_surface_sha256": sha256_file(ROOT / "ventricular_surface.json"),
            "fibre_field": "unavailable; fast calibration must use isotropic propagation",
            "endocardial_source": (
                "exact coordinate intersection of published closed LV/RV cavity surfaces "
                "with the published ventricular myocardial boundary"
            ),
            "endocardial_match_tolerance_mm": MATCH_TOLERANCE_MM,
        },
    }
    geometry_path = OUT / "native_ep_geometry.json"
    geometry_path.write_text(
        json.dumps(ep_geometry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    sorted_volumes = sorted(value / 6.0 for value in absolute_volumes6)
    mesh_manifest = {
        "schema_version": "cardiep-physionet-case3-mesh-manifest-v2",
        "subject_id": SUBJECT_ID,
        "source_coordinate_unit": "mm",
        "topology": {
            "input_vertices": len(vertices),
            "input_triangles": len(triangles),
            "input_edges": len(edge_counts),
            "input_euler_characteristic": euler_characteristic,
            "watertight_manifold": True,
            "output_nodes": len(nodes),
            "output_tetrahedra": len(oriented_tets),
            "output_boundary_faces": len(observed_boundary),
            "boundary_matches_input_exactly": True,
            "tetrahedral_domain_connected": True,
            "endocardial_nodes": len(endocardial_nodes),
        },
        "tetrahedron_volume_mm3": {
            "min": sorted_volumes[0],
            "median": sorted_volumes[len(sorted_volumes) // 2],
            "max": sorted_volumes[-1],
            "total": sum(sorted_volumes),
        },
        "tetgen": {
            "command": "tetgen -pq1.5YQ case3_ventricular_surface.off",
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        },
        "endocardial_sources": endocardial_sources,
        "artifacts": {
            geometry_path.name: sha256_file(geometry_path),
            off_path.name: sha256_file(off_path),
            node_path.name: sha256_file(node_path),
            ele_path.name: sha256_file(ele_path),
        },
        "scientific_scope": (
            "Constrained tetrahedralization of the published patient-specific closed "
            "ventricular myocardial surface. Endocardial candidates use only exact shared "
            "vertices between the myocardial boundary and published closed LV/RV cavity "
            "surfaces; cavity closure vertices are excluded. No fibre orientation is "
            "inferred from unavailable data; downstream fast EP must remain isotropic."
        ),
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(mesh_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "nodes": len(nodes),
                "tetrahedra": len(oriented_tets),
                "boundary_faces": len(observed_boundary),
                "endocardial_nodes": len(endocardial_nodes),
                "endocardial_sources": endocardial_sources,
                "total_volume_mm3": mesh_manifest["tetrahedron_volume_mm3"]["total"],
                "geometry_sha256": sha256_file(geometry_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
