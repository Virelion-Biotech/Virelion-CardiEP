from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

SUBJECT_ID = "physionet-cinc2007-case0003"
OUTPUT = Path("case3_preflight")
LICENSE = "Open Data Commons Attribution License v1.0"
CURRENT_BASE = "https://physionet.org/files/challenge-2007/1.0.0/data/"
ARCHIVE_BASE = "https://archive.physionet.org/challenge/2007/data/"
SOURCE_PATHS = {
    "readme": "readme.txt",
    "glasgow": "case0003.dat",
    "electrodes_120": "case0003_b120.pts",
    "electrodes_352": "case0003_b352.pts",
    "ventricles": "case0003-tri/case3_ventr.tri",
    "torso": "case0003-tri/case3_torso.tri",
    "measured_node_indices": "case0003-tri/elnodes_120_352.lst",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(relative: str) -> tuple[bytes, str]:
    errors: list[str] = []
    for base in (CURRENT_BASE, ARCHIVE_BASE):
        url = base + relative
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Virelion-CardiEP-PhysioNet-Case3/1.0"},
                )
                with urllib.request.urlopen(req, timeout=90) as response:
                    payload = response.read()
                if not payload:
                    raise RuntimeError("empty response")
                return payload, url
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                errors.append(f"{url} attempt {attempt + 1}: {exc}")
                time.sleep(1.0 + attempt)
    raise RuntimeError("Could not download public source file:\n" + "\n".join(errors))


def _write_json(name: str, payload: object) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _parse_pts(data: bytes, expected_count: int) -> list[list[float]]:
    points: list[list[float]] = []
    for raw in data.decode("utf-8", errors="strict").splitlines():
        fields = raw.split()
        if len(fields) != 3:
            continue
        try:
            point = [float(value) for value in fields]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in point):
            raise ValueError("PTS file contains non-finite coordinates")
        points.append(point)
    if len(points) != expected_count:
        raise ValueError(f"Expected {expected_count} PTS rows, found {len(points)}")
    return points


def _parse_index_list(data: bytes, expected_count: int) -> list[int]:
    lines = [line.strip() for line in data.decode("utf-8", errors="strict").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty index list")
    shape = [int(value) for value in lines[0].split()]
    if shape[:1] != [expected_count]:
        raise ValueError(f"Index-list header does not declare {expected_count} entries: {shape}")
    values = [int(line) for line in lines[1:]]
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} index rows, found {len(values)}")
    if len(set(values)) != len(values) or min(values) < 1 or max(values) > 352:
        raise ValueError("Measured-node list is not a unique 1-based subset of 1..352")
    return values


def _parse_tri(data: bytes) -> tuple[list[list[float]], list[list[int]]]:
    lines = [line.strip() for line in data.decode("utf-8", errors="strict").splitlines() if line.strip()]
    cursor = 0
    n_vertices = int(lines[cursor].split()[0])
    cursor += 1
    vertices: list[list[float]] = []
    for expected_label in range(1, n_vertices + 1):
        fields = lines[cursor].split()
        cursor += 1
        if len(fields) < 4 or int(fields[0]) != expected_label:
            raise ValueError("TRI vertex labels are not sequential")
        xyz = [float(value) for value in fields[1:4]]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("TRI file contains non-finite vertex coordinates")
        vertices.append(xyz)

    n_triangles = int(lines[cursor].split()[0])
    cursor += 1
    triangles: list[list[int]] = []
    for expected_label in range(1, n_triangles + 1):
        fields = lines[cursor].split()
        cursor += 1
        if len(fields) < 4 or int(fields[0]) != expected_label:
            raise ValueError("TRI triangle labels are not sequential")
        tri = [int(value) - 1 for value in fields[1:4]]
        if len(set(tri)) != 3 or min(tri) < 0 or max(tri) >= n_vertices:
            raise ValueError("TRI file contains an invalid triangle")
        triangles.append(tri)
    return vertices, triangles


def _parse_glasgow(data: bytes) -> dict:
    lines = data.decode("utf-8", errors="strict").splitlines()
    if len(lines) < 4:
        raise ValueError("Glasgow file is too short")
    header = [float(value) for value in lines[0].split()]
    if len(header) != 7:
        raise ValueError("Unexpected Glasgow timing header")
    sample_interval_ms, p_onset, p_offset, qrs_onset, qrs_offset, t_offset, n_raw = header
    n_samples = int(round(n_raw))
    if not math.isclose(n_raw, n_samples, abs_tol=1e-9):
        raise ValueError("Glasgow sample count is not integral")

    rows: list[dict] = []
    for line in lines[3:]:
        fields = line.split()
        # sample, time, four-character flags, RA, LA, LL, 352 nodes
        if len(fields) != 358:
            continue
        try:
            sample_number = int(fields[0])
            time_ms = float(fields[1])
            flags = fields[2]
            limbs = [float(value) for value in fields[3:6]]
            nodes = [float(value) for value in fields[6:]]
        except ValueError:
            continue
        if len(flags) != 4 or any(char not in "01" for char in flags):
            raise ValueError("Unexpected Glasgow interval flags")
        if len(nodes) != 352:
            raise ValueError("Glasgow row does not contain 352 torso nodes")
        values = [time_ms, *limbs, *nodes]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Glasgow data contain non-finite values")
        rows.append(
            {
                "sample_number": sample_number,
                "time_ms": time_ms,
                "flags": flags,
                "ra": limbs[0],
                "la": limbs[1],
                "ll": limbs[2],
                "nodes": nodes,
            }
        )
    if len(rows) != n_samples:
        raise ValueError(f"Glasgow header declares {n_samples} samples, parsed {len(rows)}")
    if [row["sample_number"] for row in rows] != list(range(1, n_samples + 1)):
        raise ValueError("Glasgow sample numbers are not contiguous")
    for left, right in zip(rows, rows[1:]):
        if not math.isclose(
            right["time_ms"] - left["time_ms"], sample_interval_ms, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("Glasgow time grid does not match declared sampling interval")
    return {
        "sample_interval_ms": sample_interval_ms,
        "p_onset_ms": p_onset,
        "p_offset_ms": p_offset,
        "qrs_onset_ms": qrs_onset,
        "qrs_offset_ms": qrs_offset,
        "t_offset_ms": t_offset,
        "n_samples": n_samples,
        "rows": rows,
    }


def _sq_distance(left: Iterable[float], right: Iterable[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _geometry_only_kcenter(points: list[list[float]], count: int) -> list[int]:
    if count < 1 or count > len(points):
        raise ValueError("Invalid calibration electrode count")
    center = [sum(point[axis] for point in points) / len(points) for axis in range(3)]
    first = max(range(len(points)), key=lambda index: (_sq_distance(points[index], center), -index))
    selected = [first]
    min_distance = [_sq_distance(point, points[first]) for point in points]
    while len(selected) < count:
        candidate = max(
            (index for index in range(len(points)) if index not in selected),
            key=lambda index: (min_distance[index], -index),
        )
        selected.append(candidate)
        for index, point in enumerate(points):
            min_distance[index] = min(min_distance[index], _sq_distance(point, points[candidate]))
    return selected


def _lead_name(index_zero_based: int) -> str:
    return f"BSPM{index_zero_based + 1:03d}"


def _baseline_correct(signal: list[float], time_ms: list[float], qrs_onset_ms: float) -> list[float]:
    baseline = [
        value
        for value, time_value in zip(signal, time_ms)
        if qrs_onset_ms - 120.0 <= time_value <= qrs_onset_ms - 30.0
    ]
    if not baseline:
        baseline = signal[: max(1, min(len(signal), 100))]
    offset = sum(baseline) / len(baseline)
    return [float(value - offset) for value in signal]


def _multilead_reference(
    signals: dict[str, list[float]],
    time_ms: list[float],
    qrs_onset_ms: float,
    qrs_offset_ms: float,
) -> float:
    best_index = None
    best_energy = -1.0
    for index, time_value in enumerate(time_ms):
        if not qrs_onset_ms <= time_value <= qrs_offset_ms:
            continue
        energy = math.sqrt(sum(values[index] ** 2 for values in signals.values()) / len(signals))
        if energy > best_energy:
            best_energy = energy
            best_index = index
    if best_index is None:
        raise ValueError("No samples fall within declared QRS interval")
    return float(time_ms[best_index])


def _windowed_ecg(
    signals: dict[str, list[float]],
    time_ms: list[float],
    *,
    reference_ms: float,
    qrs_onset_ms: float,
    qrs_offset_ms: float,
) -> dict:
    start = qrs_onset_ms - 50.0
    stop = qrs_offset_ms + 50.0
    indices = [index for index, value in enumerate(time_ms) if start <= value <= stop]
    if len(indices) < 20:
        raise ValueError("QRS comparison window is unexpectedly short")
    names = list(signals)
    return {
        "schema_version": "cardiep-ecg-observation-v1",
        "subject_id": SUBJECT_ID,
        "lead_names": names,
        "relative_time_s": [(time_ms[index] - reference_ms) / 1000.0 for index in indices],
        "beat_template": {
            name: [signals[name][index] for index in indices]
            for name in names
        },
        "units": "uV",
        "reference": {
            "method": "max_multilead_rms_within_source_declared_qrs",
            "source_time_ms": reference_ms,
            "qrs_onset_ms": qrs_onset_ms,
            "qrs_offset_ms": qrs_offset_ms,
            "window_padding_ms": 50.0,
        },
    }


def _standard_12(rows: list[dict], time_ms: list[float], qrs_onset_ms: float) -> dict[str, list[float]]:
    signals = {name: [] for name in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")}
    for row in rows:
        ra = row["ra"]
        la = row["la"]
        ll = row["ll"]
        lead_i = la - ra
        lead_ii = ll - ra
        lead_iii = ll - la
        nodes = row["nodes"]
        values = {
            "I": lead_i,
            "II": lead_ii,
            "III": lead_iii,
            "aVR": -(lead_i + lead_ii) / 2.0,
            "aVL": (lead_i - lead_iii) / 2.0,
            "aVF": (lead_ii + lead_iii) / 2.0,
            "V1": nodes[168],
            "V2": nodes[170],
            "V3": (nodes[191] + nodes[192]) / 2.0,
            "V4": nodes[215],
            "V5": (nodes[216] + 2.0 * nodes[217]) / 3.0,
            "V6": nodes[218],
        }
        for name, value in values.items():
            signals[name].append(float(value))
    return {
        name: _baseline_correct(values, time_ms, qrs_onset_ms)
        for name, values in signals.items()
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}
    blobs: dict[str, bytes] = {}
    for name, relative in SOURCE_PATHS.items():
        payload, url = _download(relative)
        blobs[name] = payload
        sources[name] = {
            "relative_path": relative,
            "retrieved_from": url,
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }

    glasgow = _parse_glasgow(blobs["glasgow"])
    rows = glasgow["rows"]
    time_ms = [float(row["time_ms"]) for row in rows]
    qrs_onset_ms = float(glasgow["qrs_onset_ms"])
    qrs_offset_ms = float(glasgow["qrs_offset_ms"])

    measured_nodes_1based = _parse_index_list(blobs["measured_node_indices"], 120)
    electrode_xyz_mm = _parse_pts(blobs["electrodes_120"], 120)
    _parse_pts(blobs["electrodes_352"], 352)

    measured_signals: dict[str, list[float]] = {}
    for electrode_index, node_index_1based in enumerate(measured_nodes_1based):
        raw = [float(row["nodes"][node_index_1based - 1]) for row in rows]
        measured_signals[_lead_name(electrode_index)] = _baseline_correct(raw, time_ms, qrs_onset_ms)

    selected = _geometry_only_kcenter(electrode_xyz_mm, 12)
    selected_set = set(selected)
    holdout = [index for index in range(120) if index not in selected_set]
    calibration_names = [_lead_name(index) for index in selected]
    holdout_names = [_lead_name(index) for index in holdout]
    calibration_signals = {name: measured_signals[name] for name in calibration_names}
    holdout_signals = {name: measured_signals[name] for name in holdout_names}
    reference_ms = _multilead_reference(
        calibration_signals,
        time_ms,
        qrs_onset_ms,
        qrs_offset_ms,
    )

    calibration_payload = _windowed_ecg(
        calibration_signals,
        time_ms,
        reference_ms=reference_ms,
        qrs_onset_ms=qrs_onset_ms,
        qrs_offset_ms=qrs_offset_ms,
    )
    calibration_payload["provenance"] = {
        "source": "PhysioNet/CinC Challenge 2007 Case 3",
        "source_file": "case0003.dat",
        "channel_type": "directly measured BSPM electrodes only",
        "selection_uses_signal_values": False,
        "selection_method": "deterministic Euclidean farthest-point k-center on case0003_b120.pts",
    }
    holdout_payload = _windowed_ecg(
        holdout_signals,
        time_ms,
        reference_ms=reference_ms,
        qrs_onset_ms=qrs_onset_ms,
        qrs_offset_ms=qrs_offset_ms,
    )
    holdout_payload["provenance"] = {
        "source": "PhysioNet/CinC Challenge 2007 Case 3",
        "source_file": "case0003.dat",
        "channel_type": "directly measured BSPM electrodes only",
        "sealed_against_fast_calibration": True,
        "alignment_reference_source": "calibration electrodes only",
    }

    standard_signals = _standard_12(rows, time_ms, qrs_onset_ms)
    standard_reference = _windowed_ecg(
        standard_signals,
        time_ms,
        reference_ms=reference_ms,
        qrs_onset_ms=qrs_onset_ms,
        qrs_offset_ms=qrs_offset_ms,
    )
    standard_reference["provenance"] = {
        "source": "PhysioNet/CinC Challenge 2007 Case 3",
        "derivation": "Challenge readme.txt formulas from limb channels and 352-node BSP map",
        "holdout_status": "reference_only_not_independent",
        "reason": (
            "The 352-node BSP map includes interpolation from the measured torso array; "
            "therefore this conventional 12-lead reconstruction is not used to define the sealed holdout."
        ),
    }

    ventr_vertices, ventr_triangles = _parse_tri(blobs["ventricles"])
    torso_vertices, torso_triangles = _parse_tri(blobs["torso"])
    _write_json(
        "ventricular_surface.json",
        {
            "schema_version": "cardiep-surface-v1",
            "subject_id": SUBJECT_ID,
            "coordinate_unit": "mm",
            "coordinate_frame": "physionet-case3-native",
            "vertices": ventr_vertices,
            "triangles": ventr_triangles,
            "source_sha256": sources["ventricles"]["sha256"],
            "surface_role": "closed_surface_bounding_ventricular_myocardial_mass",
        },
    )
    _write_json(
        "torso_surface.json",
        {
            "schema_version": "cardiep-torso-surface-v1",
            "subject_id": SUBJECT_ID,
            "coordinate_unit": "mm",
            "coordinate_frame": "physionet-case3-native",
            "vertices": torso_vertices,
            "triangles": torso_triangles,
            "source_sha256": sources["torso"]["sha256"],
        },
    )
    _write_json(
        "electrodes_120.json",
        {
            "schema_version": "cardiep-electrode-array-v1",
            "subject_id": SUBJECT_ID,
            "coordinate_unit": "mm",
            "coordinate_frame": "physionet-case3-native",
            "electrodes": {
                _lead_name(index): point for index, point in enumerate(electrode_xyz_mm)
            },
            "source_sha256": sources["electrodes_120"]["sha256"],
        },
    )
    calibration_path = _write_json("calibration_12_measured.json", calibration_payload)
    holdout_path = _write_json("holdout_108_measured.json", holdout_payload)
    standard_path = _write_json("standard_12_reference.json", standard_reference)

    split = {
        "schema_version": "cardiep-spatial-holdout-v1",
        "subject_id": SUBJECT_ID,
        "calibration_count": 12,
        "holdout_count": 108,
        "calibration_electrodes": calibration_names,
        "holdout_electrodes": holdout_names,
        "calibration_indices_zero_based": selected,
        "holdout_indices_zero_based": holdout,
        "measured_torso_node_indices_1based": measured_nodes_1based,
        "selection": {
            "method": "deterministic_geometry_only_farthest_point_kcenter",
            "signal_values_consulted": False,
            "coordinate_source": "case0003_b120.pts",
        },
        "validation_scope": (
            "same-acquisition spatially held-out body-surface ECG channels; "
            "not an independent visit or repeat acquisition"
        ),
        "leakage_control": (
            "The conventional 12-lead reconstruction is not used for calibration or holdout definition, "
            "because it is partly derived from the 352-node interpolated BSP map."
        ),
        "qrs_reference_ms_from_calibration_only": reference_ms,
        "artifact_sha256": {
            calibration_path.name: _sha256(calibration_path.read_bytes()),
            holdout_path.name: _sha256(holdout_path.read_bytes()),
            standard_path.name: _sha256(standard_path.read_bytes()),
        },
    }
    split_path = _write_json("split_manifest.json", split)

    manifest = {
        "schema_version": "cardiep-physionet-case3-preflight-v1",
        "subject_id": SUBJECT_ID,
        "dataset": "PhysioNet/Computing in Cardiology Challenge 2007 v1.0.0",
        "license": LICENSE,
        "source_files": sources,
        "source_timing": {
            "sample_interval_ms": glasgow["sample_interval_ms"],
            "n_samples": glasgow["n_samples"],
            "p_onset_ms": glasgow["p_onset_ms"],
            "p_offset_ms": glasgow["p_offset_ms"],
            "qrs_onset_ms": qrs_onset_ms,
            "qrs_offset_ms": qrs_offset_ms,
            "t_offset_ms": glasgow["t_offset_ms"],
        },
        "privacy": {
            "public_dataset_code_only": True,
            "patient_demographics_propagated": False,
        },
        "scientific_scope": {
            "anatomy": "patient-specific public Case 3 ventricular and torso surfaces",
            "calibration": "12 geometry-selected directly measured BSPM channels",
            "validation": "108 directly measured BSPM channels sealed from calibration",
            "standard_12": "reference-only reconstruction; not treated as independent holdout",
        },
        "split_manifest_sha256": _sha256(split_path.read_bytes()),
    }
    manifest_path = _write_json("manifest.json", manifest)

    print(json.dumps({
        "subject_id": SUBJECT_ID,
        "source_count": len(sources),
        "ventricular_vertices": len(ventr_vertices),
        "ventricular_triangles": len(ventr_triangles),
        "torso_vertices": len(torso_vertices),
        "torso_triangles": len(torso_triangles),
        "calibration_electrodes": calibration_names,
        "holdout_count": len(holdout_names),
        "qrs_reference_ms": reference_ms,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
