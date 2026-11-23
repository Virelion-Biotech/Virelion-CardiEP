from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path

SUBJECT_ID = "physionet-cinc2007-case0003"
ROOT = Path("case3_preflight")
OUT = Path("case3_patient_mesh")
CURRENT_BASE = "https://physionet.org/files/challenge-2007/1.0.0/data/case0003-tri/"
ARCHIVE_BASE = "https://archive.physionet.org/challenge/2007/data/case0003-tri/"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def download(name: str) -> tuple[bytes, str]:
    failures: list[str] = []
    for base in (CURRENT_BASE, ARCHIVE_BASE):
        url = base + name
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Virelion-CardiEP-PhysioNet-Case3-Mesh/1.0"},
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    payload = response.read()
                if not payload:
                    raise RuntimeError("empty response")
                return payload, url
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                failures.append(f"{url} attempt {attempt + 1}: {exc}")
                time.sleep(1 + attempt)
    raise RuntimeError("Could not download public geometry:\n" + "\n".join(failures))


def parse_tri_bytes(data: bytes) -> tuple[list[list[float]], list[list[int]]]:
    lines = [line.strip() for line in data.decode("utf-8", errors="strict").splitlines() if line.strip()]
    cursor = 0
    n_vertices = int(lines[cursor].split()[0])
    cursor += 1
    vertices: list[list[float]] = []
    for label in range(1, n_vertices + 1):
        fields = lines[cursor].split()
        cursor += 1
        if int(fields[0]) != label or len(fields) < 4:
            raise ValueError("TRI vertex labels are not sequential")
        xyz = [float(value) for value in fields[1:4]]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("TRI contains non-finite coordinates")
        vertices.append(xyz)
    n_triangles = int(lines[cursor].split()[0])
    cursor += 1
    triangles: list[list[int]] = []
    for label in range(1, n_triangles + 1):
        fields = lines[cursor].split()
        cursor += 1
        if int(fields[0]) != label or len(fields) < 4:
            raise ValueError("TRI triangle labels are not sequential")
        tri = [int(value) - 1 for value in fields[1:4]]
        if min(tri) < 0 or max(tri) >= n_vertices or len(set(tri)) != 3:
            raise ValueError("TRI contains invalid triangle indices")
        triangles.append(tri)
    return vertices, triangles


def undirected_edge_counts(triangles: list[list[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in triangles:
        for left, right in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted((left, right)))] += 1
    return counts


def write_off(path: Path, vertices: list[list[float]], triangles: list[list[int]]) -> None:
    lines = ["OFF", f"{len(vertices)} {len(triangles)} 0"]
    lines.extend(" ".join(f"{value:.12g}" for value in point) for point in vertices)
    lines.extend(f"3 {tri[0]} {tri[1]} {tri[2]}" for tri in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_tetgen_node(path: Path) -> list[list[float]]:
    raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    header = raw[0].split()
    count, dimension = int(header[0]), int(header[1])
    if dimension != 3:
        raise ValueError("TetGen node file is not 3D")
    points: dict[int, list[float]] = {}
    for line in raw[1:]:
        fields = line.split()
        if len(fields) < 4:
            continue
        label = int(fields[0])
        point = [float(value) for value in fields[1:4]]
        if not all(math.isfinite(value) for value in point):
            raise ValueError("TetGen emitted non-finite node coordinates")
        points[label] = point
    if len(points) != count:
        raise ValueError(f"TetGen node header says {count}, parsed {len(points)}")
    labels = sorted(points)
    if labels not in (list(range(count)), list(range(1, count + 1))):
        raise ValueError("TetGen node labels are not contiguous")
    return [points[label] for label in labels]


def read_tetgen_ele(path: Path, node_count: int) -> list[list[int]]:
    raw = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    header = raw[0].split()
    count, nodes_per_tet = int(header[0]), int(header[1])
    if nodes_per_tet != 4:
        raise ValueError("TetGen did not emit linear tetrahedra")
    rows: dict[int, list[int]] = {}
    labels_seen: list[int] = []
    for line in raw[1:]:
        fields = line.split()
        if len(fields) < 5:
            continue
        label = int(fields[0])
        labels_seen.extend(int(value) for value in fields[1:5])
        rows[label] = [int(value) for value in fields[1:5]]
    if len(rows) != count:
        raise ValueError(f"TetGen element header says {count}, parsed {len(rows)}")
    base = 0 if min(labels_seen) == 0 else 1
    tets = [[value - base for value in rows[label]] for label in sorted(rows)]
    for tet in tets:
        if len(set(tet)) != 4 or min(tet) < 0 or max(tet) >= node_count:
            raise ValueError("TetGen emitted invalid tetrahedron indices")
    return tets


def signed_volume6(points: list[list[float]], tet: list[int]) -> float:
    a, b, c, d = ([float(x) for x in points[index]] for index in tet)
    ab = [b[i] - a[i] for i in range(3)]
    ac = [c[i] - a[i] for i in range(3)]
    ad = [d[i] - a[i] for i in range(3)]
    cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
    return sum(cross[i] * ad[i] for i in range(3))


def tet_boundary(tets: list[list[int]]) -> set[tuple[int, int, int]]:
    counts: Counter[tuple[int, int, int]] = Counter()
    for a, b, c, d in tets:
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            counts[tuple(sorted(face))] += 1
    if any(value not in (1, 2) for value in counts.values()):
        raise ValueError("Non-manifold tetrahedral face incidence")
    return {face for face, value in counts.items() if value == 1}


def connected_tets(tets: list[list[int]]) -> bool:
    face_to_tets: dict[tuple[int, int, int], list[int]] = {}
    for index, tet in enumerate(tets):
        a, b, c, d = tet
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_to_tets.setdefault(tuple(sorted(face)), []).append(index)
    adjacency = [[] for _ in tets]
    for incident in face_to_tets.values():
        if len(incident) == 2:
            left, right = incident
            adjacency[left].append(right)
            adjacency[right].append(left)
    seen = {0}
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for nxt in adjacency[current]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(tets)


def nearest_node(point: list[float], nodes: list[list[float]], *, tolerance_mm: float = 1e-5) -> int:
    best_index = -1
    best_sq = float("inf")
    for index, candidate in enumerate(nodes):
        distance_sq = sum((float(point[axis]) - float(candidate[axis])) ** 2 for axis in range(3))
        if distance_sq < best_sq:
            best_sq = distance_sq
            best_index = index
    if math.sqrt(best_sq) > tolerance_mm:
        raise ValueError(f"Published endocardial node is not preserved by TetGen (nearest {math.sqrt(best_sq):.3g} mm)")
    return best_index


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
    ventricular = json.loads((ROOT / "ventricular_surface.json").read_text(encoding="utf-8"))
    vertices = ventricular["vertices"]
    triangles = ventricular["triangles"]
    if ventricular.get("coordinate_unit") != "mm":
        raise ValueError("Case 3 ventricular surface must remain in source millimetres")

    edge_counts = undirected_edge_counts(triangles)
    if not edge_counts or any(value != 2 for value in edge_counts.values()):
        bad = sum(value != 2 for value in edge_counts.values())
        raise ValueError(f"Published ventricular surface is not watertight/manifold: {bad} bad edges")
    euler_characteristic = len(vertices) - len(edge_counts) + len(triangles)
    if euler_characteristic != 0:
        raise ValueError(
            f"Expected the documented genus-1 ventricular surface (Euler characteristic 0), got {euler_characteristic}"
        )

    endocardial_sources: dict[str, dict] = {}
    endocardial_points: list[list[float]] = []
    for role, filename in (("left_endocardium", "case3_lendo.tri"), ("right_endocardium", "case3_rendo.tri")):
        payload, url = download(filename)
        surface_vertices, surface_triangles = parse_tri_bytes(payload)
        endocardial_sources[role] = {
            "filename": filename,
            "retrieved_from": url,
            "sha256": sha256_bytes(payload),
            "vertices": len(surface_vertices),
            "triangles": len(surface_triangles),
        }
        endocardial_points.extend(surface_vertices)

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

    # -Y must preserve each published boundary vertex and triangle exactly.
    original_to_output = [nearest_node(point, nodes) for point in vertices]
    if len(set(original_to_output)) != len(vertices):
        raise ValueError("Published ventricular vertices do not map one-to-one into TetGen output")
    expected_boundary = {
        tuple(sorted(original_to_output[index] for index in tri))
        for tri in triangles
    }
    observed_boundary = tet_boundary(tets)
    if observed_boundary != expected_boundary:
        raise ValueError(
            "TetGen tetrahedral boundary differs from the published ventricular surface: "
            f"expected={len(expected_boundary)} observed={len(observed_boundary)} "
            f"missing={len(expected_boundary - observed_boundary)} extra={len(observed_boundary - expected_boundary)}"
        )
    if not connected_tets(tets):
        raise ValueError("TetGen ventricular volume mesh is disconnected")

    volumes6 = [signed_volume6(nodes, tet) for tet in tets]
    absolute_volumes6 = [abs(value) for value in volumes6]
    if min(absolute_volumes6) <= 1e-9:
        raise ValueError("TetGen emitted zero/near-zero volume tetrahedra")
    # Orient tetrahedra consistently positive for downstream consumers.
    oriented_tets: list[list[int]] = []
    for tet, volume6 in zip(tets, volumes6, strict=True):
        if volume6 < 0:
            tet = [tet[0], tet[2], tet[1], tet[3]]
        oriented_tets.append(tet)

    endocardial_nodes = sorted({nearest_node(point, nodes) for point in endocardial_points})
    if len(endocardial_nodes) < 20:
        raise ValueError("Endocardial mapping produced implausibly few unique nodes")

    electrodes = json.loads((ROOT / "electrodes_120.json").read_text(encoding="utf-8"))
    if electrodes.get("coordinate_frame") != "physionet-case3-native":
        raise ValueError("Electrode coordinate frame differs from ventricular surface")
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
            "endocardial_source": "published Case 3 left/right cavity surfaces",
        },
    }
    geometry_path = OUT / "native_ep_geometry.json"
    geometry_path.write_text(
        json.dumps(ep_geometry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    sorted_volumes = sorted(value / 6.0 for value in absolute_volumes6)
    mesh_manifest = {
        "schema_version": "cardiep-physionet-case3-mesh-manifest-v1",
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
            "Constrained tetrahedralization of the published patient-specific closed ventricular myocardial surface. "
            "No fibre orientation is inferred from unavailable data; downstream fast EP must remain isotropic."
        ),
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(mesh_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "nodes": len(nodes),
        "tetrahedra": len(oriented_tets),
        "boundary_faces": len(observed_boundary),
        "endocardial_nodes": len(endocardial_nodes),
        "total_volume_mm3": mesh_manifest["tetrahedron_volume_mm3"]["total"],
        "geometry_sha256": sha256_file(geometry_path),
        "manifest_sha256": sha256_file(manifest_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
