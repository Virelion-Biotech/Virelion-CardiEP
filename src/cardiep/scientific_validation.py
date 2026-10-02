from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np


_COORD_TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0}
_TIME_TO_MS = {
    "ms": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "us": 0.001,
    "µs": 0.001,
    "μs": 0.001,
    "microsecond": 0.001,
    "microseconds": 0.001,
}


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ActivationProfile:
    benchmark_id: str
    solver_name: str
    sample_ids: tuple[str, ...]
    points_cm: np.ndarray
    activation_ms: np.ndarray
    solver_version: str | None = None
    solver_commit: str | None = None
    equation: str | None = None
    ionic_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(str(item) for item in self.sample_ids)
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        if not self.solver_name.strip():
            raise ValueError("solver_name must be non-empty")
        if len(ids) == 0 or len(ids) != len(set(ids)):
            raise ValueError("sample_ids must be unique and non-empty")
        points = _finite_array(self.points_cm, name="points_cm", ndim=2)
        activation = _finite_array(self.activation_ms, name="activation_ms", ndim=1)
        if points.shape != (len(ids), 3):
            raise ValueError("points_cm must have shape (N, 3) matching sample_ids")
        if activation.shape != (len(ids),):
            raise ValueError("activation_ms must contain one value per sample_id")
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "points_cm", points)
        object.__setattr__(self, "activation_ms", activation)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ActivationProfile:
        if raw.get("schema_version") != "cardiep-activation-profile-v1":
            raise ValueError(
                "Activation profile schema_version must be 'cardiep-activation-profile-v1'"
            )
        benchmark_id = str(raw.get("benchmark_id") or "").strip()
        solver = raw.get("solver")
        if not isinstance(solver, dict):
            raise TypeError("Activation profile solver must be an object")
        points_raw = raw.get("sample_points")
        if not isinstance(points_raw, list) or not points_raw:
            raise TypeError("sample_points must be a non-empty list")

        coord_unit = str(raw.get("coordinate_unit") or "").strip().lower()
        if coord_unit not in _COORD_TO_CM:
            raise ValueError("coordinate_unit must be one of mm, cm, m")
        time_unit = str(raw.get("time_unit") or "").strip().lower()
        if time_unit not in _TIME_TO_MS:
            raise ValueError("time_unit must be a recognized time unit")

        ids: list[str] = []
        points: list[list[float]] = []
        for item in points_raw:
            if not isinstance(item, dict):
                raise TypeError("Each sample_points entry must be an object")
            sample_id = str(item.get("id") or "").strip()
            xyz = _finite_array(item.get("xyz"), name=f"sample point {sample_id!r}")
            if xyz.shape != (3,):
                raise ValueError(f"sample point {sample_id!r} xyz must contain 3 values")
            ids.append(sample_id)
            points.append(xyz.tolist())

        activation_raw = raw.get("activation")
        if not isinstance(activation_raw, dict):
            raise TypeError("activation must be an object keyed by sample id")
        if set(activation_raw) != set(ids):
            missing = sorted(set(ids) - set(activation_raw))
            extra = sorted(set(activation_raw) - set(ids))
            raise ValueError(
                f"activation keys must exactly match sample_points; missing={missing}, extra={extra}"
            )

        return cls(
            benchmark_id=benchmark_id,
            solver_name=str(solver.get("name") or "").strip(),
            solver_version=(
                None if solver.get("version") is None else str(solver.get("version"))
            ),
            solver_commit=(
                None if solver.get("commit") is None else str(solver.get("commit"))
            ),
            equation=None if raw.get("equation") is None else str(raw.get("equation")),
            ionic_model=(
                None if raw.get("ionic_model") is None else str(raw.get("ionic_model"))
            ),
            sample_ids=tuple(ids),
            points_cm=np.asarray(points, dtype=float) * _COORD_TO_CM[coord_unit],
            activation_ms=np.asarray(
                [float(activation_raw[sample_id]) for sample_id in ids],
                dtype=float,
            )
            * _TIME_TO_MS[time_unit],
            metadata=dict(raw.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ActivationProfile:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("Activation profile JSON must contain an object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cardiep-activation-profile-v1",
            "benchmark_id": self.benchmark_id,
            "solver": {
                "name": self.solver_name,
                "version": self.solver_version,
                "commit": self.solver_commit,
            },
            "equation": self.equation,
            "ionic_model": self.ionic_model,
            "coordinate_unit": "cm",
            "time_unit": "ms",
            "sample_points": [
                {"id": sample_id, "xyz": self.points_cm[i].tolist()}
                for i, sample_id in enumerate(self.sample_ids)
            ],
            "activation": {
                sample_id: float(self.activation_ms[i])
                for i, sample_id in enumerate(self.sample_ids)
            },
            "metadata": dict(self.metadata),
        }



def activation_profile_from_csv(
    path: str | Path,
    *,
    benchmark_id: str,
    solver_name: str,
    coordinate_unit: str = "cm",
    time_unit: str = "ms",
    id_column: str = "id",
    x_column: str = "x",
    y_column: str = "y",
    z_column: str = "z",
    activation_column: str = "activation",
    solver_version: str | None = None,
    solver_commit: str | None = None,
    equation: str | None = None,
    ionic_model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ActivationProfile:
    coord_key = str(coordinate_unit).strip().lower()
    time_key = str(time_unit).strip().lower()
    if coord_key not in _COORD_TO_CM:
        raise ValueError("coordinate_unit must be one of mm, cm, m")
    if time_key not in _TIME_TO_MS:
        raise ValueError("time_unit must be a recognized time unit")

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            id_column,
            x_column,
            y_column,
            z_column,
            activation_column,
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Activation CSV is missing columns: {missing}")
        ids: list[str] = []
        points: list[list[float]] = []
        activation: list[float] = []
        for row in reader:
            sample_id = str(row[id_column]).strip()
            if not sample_id:
                raise ValueError("Activation CSV contains an empty sample ID")
            ids.append(sample_id)
            points.append(
                [
                    float(row[x_column]),
                    float(row[y_column]),
                    float(row[z_column]),
                ]
            )
            activation.append(float(row[activation_column]))

    if not ids:
        raise ValueError("Activation CSV contains no rows")
    return ActivationProfile(
        benchmark_id=benchmark_id,
        solver_name=solver_name,
        solver_version=solver_version,
        solver_commit=solver_commit,
        equation=equation,
        ionic_model=ionic_model,
        sample_ids=tuple(ids),
        points_cm=np.asarray(points, dtype=float) * _COORD_TO_CM[coord_key],
        activation_ms=np.asarray(activation, dtype=float) * _TIME_TO_MS[time_key],
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class AgreementThresholds:
    rmse_ms_max: float
    max_abs_ms_max: float
    correlation_min: float
    abs_bias_ms_max: float

    def __post_init__(self) -> None:
        numeric = {
            "rmse_ms_max": self.rmse_ms_max,
            "max_abs_ms_max": self.max_abs_ms_max,
            "correlation_min": self.correlation_min,
            "abs_bias_ms_max": self.abs_bias_ms_max,
        }
        if not all(math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("Agreement thresholds must be finite")
        if self.rmse_ms_max < 0 or self.max_abs_ms_max < 0 or self.abs_bias_ms_max < 0:
            raise ValueError("Error thresholds must be non-negative")
        if not -1.0 <= self.correlation_min <= 1.0:
            raise ValueError("correlation_min must lie in [-1, 1]")


def _aligned_values(
    reference: ActivationProfile,
    candidate: ActivationProfile,
    *,
    point_tolerance_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if reference.benchmark_id != candidate.benchmark_id:
        raise ValueError(
            f"Benchmark mismatch: {reference.benchmark_id!r} != {candidate.benchmark_id!r}"
        )
    if set(reference.sample_ids) != set(candidate.sample_ids):
        raise ValueError("Activation profiles do not contain the same sample IDs")
    candidate_index = {name: i for i, name in enumerate(candidate.sample_ids)}
    candidate_order = np.asarray(
        [candidate_index[name] for name in reference.sample_ids],
        dtype=int,
    )
    candidate_points = candidate.points_cm[candidate_order]
    point_delta = np.linalg.norm(reference.points_cm - candidate_points, axis=1)
    if np.any(point_delta > point_tolerance_cm):
        bad = [
            reference.sample_ids[i]
            for i in np.where(point_delta > point_tolerance_cm)[0].tolist()
        ]
        raise ValueError(
            f"Sample coordinates differ beyond tolerance for IDs: {bad}"
        )
    return reference.activation_ms.copy(), candidate.activation_ms[candidate_order].copy()


def compare_activation_profiles(
    reference: ActivationProfile,
    candidate: ActivationProfile,
    *,
    alignment: Literal["absolute", "p1-relative", "minimum-relative"] = "absolute",
    thresholds: AgreementThresholds | None = None,
    point_tolerance_cm: float = 1e-6,
) -> dict[str, Any]:
    if point_tolerance_cm < 0 or not math.isfinite(float(point_tolerance_cm)):
        raise ValueError("point_tolerance_cm must be finite and non-negative")
    ref, cand = _aligned_values(
        reference,
        candidate,
        point_tolerance_cm=point_tolerance_cm,
    )
    if alignment == "p1-relative":
        if "P1" not in reference.sample_ids:
            raise ValueError("p1-relative alignment requires sample ID 'P1'")
        index = reference.sample_ids.index("P1")
        ref = ref - ref[index]
        cand = cand - cand[index]
    elif alignment == "minimum-relative":
        ref = ref - float(np.min(ref))
        cand = cand - float(np.min(cand))
    elif alignment != "absolute":
        raise ValueError(f"Unknown activation alignment {alignment!r}")

    residual = cand - ref
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    max_abs = float(np.max(np.abs(residual)))
    bias = float(np.mean(residual))
    residual_sd = float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0
    reference_span = float(np.ptp(ref))
    normalized_rmse = rmse / reference_span if reference_span > 1e-12 else None

    if np.allclose(ref, ref[0]) or np.allclose(cand, cand[0]):
        correlation = 1.0 if np.allclose(ref, cand) else 0.0
        slope = None
        intercept = None
    else:
        correlation = float(np.corrcoef(ref, cand)[0, 1])
        slope, intercept = np.polyfit(ref, cand, 1)
        slope = float(slope)
        intercept = float(intercept)

    checks: dict[str, bool] = {}
    status = "not_gated"
    if thresholds is not None:
        checks = {
            "rmse_ms": rmse <= thresholds.rmse_ms_max,
            "max_abs_ms": max_abs <= thresholds.max_abs_ms_max,
            "correlation": correlation >= thresholds.correlation_min,
            "abs_bias_ms": abs(bias) <= thresholds.abs_bias_ms_max,
        }
        status = "pass" if all(checks.values()) else "fail"

    return {
        "schema_version": "cardiep-cross-solver-report-v1",
        "benchmark_id": reference.benchmark_id,
        "reference_solver": reference.solver_name,
        "candidate_solver": candidate.solver_name,
        "alignment": alignment,
        "n_points": len(ref),
        "metrics": {
            "rmse_ms": rmse,
            "mae_ms": mae,
            "max_abs_ms": max_abs,
            "bias_ms": bias,
            "residual_sd_ms": residual_sd,
            "loa95_lower_ms": bias - 1.96 * residual_sd,
            "loa95_upper_ms": bias + 1.96 * residual_sd,
            "correlation": correlation,
            "linear_slope": slope,
            "linear_intercept_ms": intercept,
            "normalized_rmse": normalized_rmse,
        },
        "thresholds": (
            None
            if thresholds is None
            else {
                "rmse_ms_max": thresholds.rmse_ms_max,
                "max_abs_ms_max": thresholds.max_abs_ms_max,
                "correlation_min": thresholds.correlation_min,
                "abs_bias_ms_max": thresholds.abs_bias_ms_max,
            }
        ),
        "checks": checks,
        "status": status,
        "scientific_boundary": (
            "Agreement on this benchmark is a numerical cross-solver check. It does not "
            "establish physiological fidelity, patient-specific validity, or clinical utility."
        ),
    }


@dataclass(frozen=True)
class ConvergenceLevel:
    h: float
    profile: ActivationProfile
    dt_ms: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.h)) or self.h <= 0:
            raise ValueError("Convergence level h must be positive and finite")
        if self.dt_ms is not None and (
            not math.isfinite(float(self.dt_ms)) or self.dt_ms <= 0
        ):
            raise ValueError("Convergence level dt_ms must be positive and finite")


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def mesh_convergence_report(
    levels: list[ConvergenceLevel],
    *,
    exact: ActivationProfile | None = None,
    refinement_ratio_tolerance: float = 0.05,
) -> dict[str, Any]:
    if len(levels) < 3:
        raise ValueError("At least three mesh levels are required")
    ordered = sorted(levels, key=lambda item: item.h, reverse=True)
    h = np.asarray([item.h for item in ordered], dtype=float)
    if not np.all(np.diff(h) < 0):
        raise ValueError("Mesh sizes must be unique")
    benchmark_ids = {item.profile.benchmark_id for item in ordered}
    if len(benchmark_ids) != 1:
        raise ValueError("All convergence profiles must use the same benchmark_id")

    base = ordered[0].profile
    values = []
    for level in ordered:
        ref, aligned = _aligned_values(
            base,
            level.profile,
            point_tolerance_cm=1e-6,
        )
        if not np.allclose(ref, base.activation_ms):
            raise RuntimeError("Unexpected convergence alignment failure")
        values.append(aligned)
    values_array = np.asarray(values, dtype=float)

    ratios = h[:-1] / h[1:]
    approximately_uniform = bool(
        np.max(np.abs(ratios - ratios[-1])) <= refinement_ratio_tolerance * ratios[-1]
    )

    errors_to_exact: list[float] | None = None
    observed_orders: list[float | None] = []
    if exact is not None:
        _, exact_values = _aligned_values(
            base,
            exact,
            point_tolerance_cm=1e-6,
        )
        errors_to_exact = [
            _rmse(values_array[i] - exact_values)
            for i in range(len(ordered))
        ]
        for i in range(len(ordered) - 1):
            coarse_error = errors_to_exact[i]
            fine_error = errors_to_exact[i + 1]
            if coarse_error <= 0 or fine_error <= 0:
                observed_orders.append(None)
            else:
                observed_orders.append(
                    float(math.log(coarse_error / fine_error) / math.log(ratios[i]))
                )
    else:
        differences = [
            _rmse(values_array[i] - values_array[i + 1])
            for i in range(len(ordered) - 1)
        ]
        for i in range(len(differences) - 1):
            d_coarse = differences[i]
            d_fine = differences[i + 1]
            ratio = ratios[i + 1]
            if d_coarse <= 0 or d_fine <= 0:
                observed_orders.append(None)
            else:
                observed_orders.append(
                    float(math.log(d_coarse / d_fine) / math.log(ratio))
                )

    consecutive_rmse = [
        _rmse(values_array[i] - values_array[i + 1])
        for i in range(len(ordered) - 1)
    ]
    monotone = all(
        consecutive_rmse[i + 1] <= consecutive_rmse[i] + 1e-12
        for i in range(len(consecutive_rmse) - 1)
    )

    gci_fine_ms = None
    finite_orders = [item for item in observed_orders if item is not None and item > 0]
    if finite_orders and approximately_uniform and len(consecutive_rmse) >= 2:
        p = float(finite_orders[-1])
        r = float(ratios[-1])
        denominator = r**p - 1.0
        if denominator > 1e-12:
            gci_fine_ms = float(1.25 * consecutive_rmse[-1] / denominator)

    return {
        "schema_version": "cardiep-mesh-convergence-report-v1",
        "benchmark_id": base.benchmark_id,
        "levels": [
            {
                "label": item.label,
                "h": float(item.h),
                "dt_ms": item.dt_ms,
                "solver": item.profile.solver_name,
            }
            for item in ordered
        ],
        "refinement_ratios": ratios.tolist(),
        "approximately_uniform_refinement": approximately_uniform,
        "consecutive_rmse_ms": consecutive_rmse,
        "errors_to_exact_ms": errors_to_exact,
        "observed_orders": observed_orders,
        "gci_fine_ms": gci_fine_ms,
        "monotone_self_convergence": monotone,
        "scientific_boundary": (
            "Mesh self-convergence is necessary for numerical credibility but does not "
            "show that the underlying electrophysiology model is biologically correct."
        ),
    }


def niederer_2011_spec() -> dict[str, Any]:
    return {
        "schema_version": "cardiep-benchmark-spec-v1",
        "benchmark_id": "niederer-2011",
        "name": "Niederer N-version monodomain benchmark",
        "geometry": {
            "shape": "cuboid",
            "dimensions_cm": [2.0, 0.7, 0.3],
            "fibre_direction": [1.0, 0.0, 0.0],
            "boundary_condition": "zero_flux",
        },
        "stimulus": {
            "corner": [0.0, 0.0, 0.0],
            "box_dimensions_cm": [0.15, 0.15, 0.15],
        },
        "equation": "monodomain",
        "ionic_model": "tenTusscher-Panfilov-2006-epicardial",
        "activation_threshold_mV": 0.0,
        "sample_points_cm": {
            "P1": [0.0, 0.0, 0.0],
            "P2": [0.0, 0.7, 0.0],
            "P3": [2.0, 0.0, 0.0],
            "P4": [2.0, 0.7, 0.0],
            "P5": [0.0, 0.0, 0.3],
            "P6": [0.0, 0.7, 0.3],
            "P7": [2.0, 0.0, 0.3],
            "P8": [2.0, 0.7, 0.3],
            "P9": [1.0, 0.35, 0.15],
        },
        "recommended_resolution_grid": {
            "dx_mm": [0.5, 0.2, 0.1],
            "dt_ms": [0.05, 0.01, 0.005],
        },
        "provenance": {
            "primary_reference": "Niederer et al., Phil Trans R Soc A (2011), doi:10.1098/rsta.2011.0139",
            "interoperability_reference": (
                "finsberg/fenicsx-beat demos/niederer_benchmark.py "
                "@ 7ab18453ae57c28d798b744234157f36297113d5"
            ),
        },
    }


def fenicsx_beat_niederer_reference() -> ActivationProfile:
    points = niederer_2011_spec()["sample_points_cm"]
    ids = tuple(points)
    return ActivationProfile(
        benchmark_id="niederer-2011",
        solver_name="fenicsx-beat",
        solver_version="0.7.0",
        solver_commit="7ab18453ae57c28d798b744234157f36297113d5",
        equation="monodomain",
        ionic_model="tenTusscher-Panfilov-2006-epicardial",
        sample_ids=ids,
        points_cm=np.asarray([points[item] for item in ids], dtype=float),
        activation_ms=np.asarray(
            [1.225, 25.5, 31.26, 37.81, 7.99, 26.09, 31.72, 37.93, 17.835],
            dtype=float,
        ),
        metadata={
            "dx_mm": 0.1,
            "dt_ms": 0.005,
            "activation_threshold_mV": 0.0,
            "source": "finsberg/fenicsx-beat demos/niederer_benchmark.py",
            "source_commit": "7ab18453ae57c28d798b744234157f36297113d5",
            "role": "interoperability fixture; rerun upstream solver for publication claims",
        },
    )


def load_convergence_manifest(path: str | Path) -> tuple[list[ConvergenceLevel], ActivationProfile | None]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "cardiep-convergence-manifest-v1":
        raise ValueError(
            "Convergence manifest schema_version must be 'cardiep-convergence-manifest-v1'"
        )
    base = Path(path).resolve().parent
    levels_raw = raw.get("levels")
    if not isinstance(levels_raw, list) or len(levels_raw) < 3:
        raise ValueError("Convergence manifest requires at least three levels")
    levels = []
    for item in levels_raw:
        if not isinstance(item, dict):
            raise TypeError("Each convergence level must be an object")
        profile_path = (base / str(item["profile"])).resolve()
        levels.append(
            ConvergenceLevel(
                h=float(item["h"]),
                dt_ms=None if item.get("dt_ms") is None else float(item["dt_ms"]),
                label=None if item.get("label") is None else str(item["label"]),
                profile=ActivationProfile.from_json(profile_path),
            )
        )
    exact = None
    if raw.get("exact_profile") is not None:
        exact = ActivationProfile.from_json((base / str(raw["exact_profile"])).resolve())
    return levels, exact
