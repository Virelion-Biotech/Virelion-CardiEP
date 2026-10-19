from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from cardiep.discrepancy import correlation_distance, normalized_rmse
from cardiep.models import ArtifactRef, EPCalibrationRequest, EPObservation, EPParameterSet
from cardiep.native_backend import NativeEikonalBackend
from cardiep.provenance import uri_to_path

SUBJECT_ID = "physionet-cinc2007-case0003"
PREFLIGHT = Path("case3_preflight")
MESH = Path("case3_patient_mesh")
OUT = Path("case3_fast_calibration")
ROOT_COUNT_CANDIDATES = (4, 6, 8, 10)
SPEED_BOUNDS_CM_PER_MS = (0.03, 0.12)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def artifact(path: Path, *, artifact_id: str, kind: str, coordinate_frame: str | None = None) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=str(path.resolve()),
        sha256=sha256_file(path),
        coordinate_frame=coordinate_frame,
    )


def per_lead_metrics(raw: dict, simulated) -> dict[str, dict[str, float]]:
    observed = raw.get("beat_template")
    if not isinstance(observed, dict):
        raise ValueError("Expected beat_template ECG observation")
    by_name = {name: simulated.values[index] for index, name in enumerate(simulated.lead_names)}
    relative_time_s = np.asarray(raw["relative_time_s"], dtype=float)
    if simulated.reference_time_ms is None:
        raise ValueError("Simulated ECG has no QRS reference time")
    sim_time = np.asarray(simulated.time_ms, dtype=float) - float(simulated.reference_time_ms)
    target_time = relative_time_s * 1000.0
    metrics: dict[str, dict[str, float]] = {}
    for name, values in observed.items():
        if name not in by_name:
            raise ValueError(f"Simulated ECG is missing lead {name}")
        obs = np.asarray(values, dtype=float)
        pred = np.interp(target_time, sim_time, by_name[name], left=0.0, right=0.0)
        metrics[name] = {
            "correlation": 1.0 - correlation_distance(obs, pred),
            "nrmse": normalized_rmse(obs, pred),
        }
    return metrics


def summarize(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    correlations = np.asarray([item["correlation"] for item in metrics.values()], dtype=float)
    nrmse = np.asarray([item["nrmse"] for item in metrics.values()], dtype=float)
    return {
        "n_leads": int(len(metrics)),
        "correlation_mean": float(np.mean(correlations)),
        "correlation_median": float(np.median(correlations)),
        "correlation_min": float(np.min(correlations)),
        "correlation_max": float(np.max(correlations)),
        "nrmse_mean": float(np.mean(nrmse)),
        "nrmse_median": float(np.median(nrmse)),
        "nrmse_min": float(np.min(nrmse)),
        "nrmse_max": float(np.max(nrmse)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry_path = MESH / "native_ep_geometry.json"
    calibration_path = PREFLIGHT / "calibration_12_measured.json"
    split_path = PREFLIGHT / "split_manifest.json"
    for path in (geometry_path, calibration_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite {path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    calibration_names = list(split["calibration_electrodes"])
    if len(calibration_names) != 12 or int(split["holdout_count"]) != 108:
        raise ValueError("Expected sealed 12/108 split")

    anatomy_ref = artifact(
        geometry_path,
        artifact_id=f"{SUBJECT_ID}-tetgen-volume",
        kind="volume_mesh",
        coordinate_frame="physionet-case3-native",
    )
    calibration_ref = artifact(
        calibration_path,
        artifact_id=f"{SUBJECT_ID}-bspm-calibration-12",
        kind="ecg",
        coordinate_frame="physionet-case3-native",
    )
    calibration_observation = EPObservation(
        observation_id="case3-bspm-calibration-12",
        kind="ecg",
        artifact=calibration_ref,
        coordinate_frame="physionet-case3-native",
        units="uV",
    )

    backend = NativeEikonalBackend()
    candidate_results: list[dict] = []
    best = None
    best_result = None
    for root_count in ROOT_COUNT_CANDIDATES:
        candidate_dir = OUT / f"roots_{root_count}"
        request = EPCalibrationRequest(
            subject_id=SUBJECT_ID,
            anatomy_ref=anatomy_ref,
            observations=[calibration_observation],
            backend=backend.name,
            parameter_bounds={"isotropic_speed": SPEED_BOUNDS_CM_PER_MS},
            initial_parameters=EPParameterSet(
                values={"isotropic_speed": 0.06},
                units={"isotropic_speed": "cm/ms"},
                source="prior",
            ),
            settings={
                "auto_root_count": root_count,
                "with_ecg": True,
                "ecg_sample_rate_hz": 1000.0,
                "ecg_pre_activation_ms": 250.0,
                "duration_ms": 200.0,
                "qrs_sigma_ms": 5.0,
                "t_sigma_ms": 20.0,
                "repolarization_scale": 0.55,
                "ecg_chunk_size": 1024,
                "output_dir": str(candidate_dir.resolve()),
                "step_fraction": 0.25,
                "max_iterations": 12,
                "step_tolerance": 2.5e-4,
                "likelihood_hints": [
                    {
                        "term_id": "calibration-qrs-correlation",
                        "observation_id": calibration_observation.observation_id,
                        "model_output": "ecg",
                        "discrepancy": "correlation",
                        "weight": 1.0,
                        "metadata": {"lead_weighting": "equal"},
                    }
                ],
            },
        )
        result = backend.calibrate(request)
        if result.objective is None or not math.isfinite(float(result.objective)):
            raise ValueError(f"Root-count {root_count} calibration returned no finite objective")
        candidate = {
            "root_count": root_count,
            "objective_correlation_distance": float(result.objective),
            "isotropic_speed_cm_per_ms": float(result.parameters.values["isotropic_speed"]),
            "converged": bool(result.converged),
            "iterations": int(result.diagnostics.get("iterations", -1)),
        }
        candidate_results.append(candidate)
        key = (candidate["objective_correlation_distance"], root_count)
        if best is None or key < best:
            best = key
            best_result = result

    if best_result is None or best is None:
        raise RuntimeError("No calibration candidate completed")
    selected_root_count = int(best[1])
    selected_speed = float(best_result.parameters.values["isotropic_speed"])

    # Re-simulate the frozen selection once, still using only the calibration observation.
    selected_settings = {
        "auto_root_count": selected_root_count,
        "with_ecg": True,
        "ecg_sample_rate_hz": 1000.0,
        "ecg_pre_activation_ms": 250.0,
        "duration_ms": 200.0,
        "qrs_sigma_ms": 5.0,
        "t_sigma_ms": 20.0,
        "repolarization_scale": 0.55,
        "ecg_chunk_size": 1024,
        "output_dir": str((OUT / "selected").resolve()),
    }
    fast = backend._simulate_arrays(
        anatomy_ref=anatomy_ref,
        parameters={"isotropic_speed": selected_speed},
        observations=[calibration_observation],
        settings=selected_settings,
    )
    if fast.ecg is None:
        raise RuntimeError("Selected fast model did not produce ECG")
    calibration_raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_metrics = per_lead_metrics(calibration_raw, fast.ecg)

    # Emit the activation field in the exact schema consumed by the prescribed-activation
    # openCARP profiles. This is the transferred state; scalar speed/root settings are not replayed.
    activation_path = OUT / "calibration_activation.json"
    write_json(
        activation_path,
        {
            "schema_version": "cardiep-field-v1",
            "subject_id": SUBJECT_ID,
            "units": "ms",
            "activation_ms": fast.propagation.activation_ms.tolist(),
            "root_nodes": fast.propagation.root_schedule.nodes.tolist(),
            "root_activation_ms": fast.propagation.root_schedule.activation_ms.tolist(),
            "root_method": fast.propagation.root_schedule.method,
            "source": "sealed_fast_calibration_case3",
        },
    )

    sealed_path = OUT / "sealed_model.json"
    sealed_model = {
        "schema_version": "cardiep-physionet-case3-sealed-fast-model-v1",
        "subject_id": SUBJECT_ID,
        "selection_data": "calibration_12_measured.json only",
        "holdout_opened_before_seal": False,
        "candidate_root_counts": list(ROOT_COUNT_CANDIDATES),
        "candidates": candidate_results,
        "selected": {
            "root_count": selected_root_count,
            "isotropic_speed_cm_per_ms": selected_speed,
            "objective_correlation_distance": float(best_result.objective),
            "root_nodes": fast.propagation.root_schedule.nodes.tolist(),
            "root_method": fast.propagation.root_schedule.method,
            "activation_span_ms": float(np.ptp(fast.propagation.activation_ms)),
        },
        "calibration_summary": summarize(calibration_metrics),
        "calibration_per_lead": calibration_metrics,
        "activation_artifact": {
            "path": activation_path.name,
            "sha256": sha256_file(activation_path),
            "schema_version": "cardiep-field-v1",
        },
        "anatomy_sha256": sha256_file(geometry_path),
        "calibration_ecg_sha256": sha256_file(calibration_path),
        "split_manifest_sha256": sha256_file(split_path),
        "scientific_scope": (
            "Exploratory fast isotropic ventricular Eikonal calibration with geometry-only root placement. "
            "No myocardial fibre field or scar field is inferred. The activation field, not scalar calibration parameters, "
            "is the state transferred to higher-fidelity forward confirmation."
        ),
    }
    write_json(sealed_path, sealed_model)
    sealed_sha = sha256_file(sealed_path)

    # HOLDOUT GATE: no holdout waveform is read above this line.
    holdout_path = PREFLIGHT / "holdout_108_measured.json"
    if not holdout_path.is_file():
        raise FileNotFoundError(holdout_path)
    holdout_raw = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout_metrics = per_lead_metrics(holdout_raw, fast.ecg)
    holdout_summary = summarize(holdout_metrics)
    validation_path = OUT / "fast_model_holdout_evaluation.json"
    write_json(
        validation_path,
        {
            "schema_version": "cardiep-physionet-case3-fast-holdout-v1",
            "subject_id": SUBJECT_ID,
            "sealed_model_sha256": sealed_sha,
            "holdout_sha256": sha256_file(holdout_path),
            "holdout_count": len(holdout_metrics),
            "summary": holdout_summary,
            "per_lead": holdout_metrics,
            "claim_boundary": (
                "Same-acquisition spatial holdout for the exploratory fast pseudo-ECG model. "
                "This is not yet the higher-fidelity openCARP confirmation and is not an independent visit."
            ),
        },
    )

    print(json.dumps({
        "selected_root_count": selected_root_count,
        "selected_isotropic_speed_cm_per_ms": selected_speed,
        "calibration_objective_correlation_distance": float(best_result.objective),
        "activation_span_ms": float(np.ptp(fast.propagation.activation_ms)),
        "calibration": summarize(calibration_metrics),
        "fast_model_holdout": holdout_summary,
        "activation_sha256": sha256_file(activation_path),
        "sealed_model_sha256": sealed_sha,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
