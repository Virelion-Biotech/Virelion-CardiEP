"""Qualified instrumentation driver for the pinned Camps QRS workflow.

The upstream ``personalise_to_QRS.py`` inference script creates the exact
``DiscrepancyEvaluator`` used by SMC-ABC, obtains ``population_theta`` and
``population_parameter``, and then discards the inference object before saving
CSV populations.  That makes post-hoc selection from the CSV files ambiguous:
row order is not a best-particle ranking and no per-particle discrepancy is
saved.

This driver instruments exactly one audited source boundary in an isolated,
pinned git worktree: immediately after
``population_parameter = inference_method.get_parameter_from_theta(...)``.
At that point it recomputes the discrepancy for every posterior particle using
the *same evaluator used by inference*, selects the minimum finite discrepancy,
simulates that exact parameter vector, exports its ECG and activation map, and
writes a complete CardiEP ``EPCalibrationResult``.  The temporary instrumented
script exits immediately after export, before upstream clears the evaluator or
enters plotting code.

The instrumentation is deliberately source-pinned and fail-closed.  If the
audited marker is absent or appears more than once, execution stops rather than
patching a different upstream revision heuristically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_artifact
from .camps_cdt_backend import (
    CAMPS_CDT_BACKEND_NAME,
    UPSTREAM_DOI,
    UPSTREAM_QRS_SAMPLE_RATE_HZ,
    UPSTREAM_REPOSITORY,
    CampsCardiacDigitalTwinExternalBackend,
)
from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPParameterSet,
    EPSimulationRequest,
    EPSimulationResult,
)
from .provenance import sha256_json

DRIVER_VERSION = "cardiep-camps-qualified-driver-v1"
AUDITED_UPSTREAM_COMMIT = "816d51fab0837cfe9e20c7d3a318429e9acf0733"
_INSERTION_MARKER = (
    "        population_parameter = inference_method.get_parameter_from_theta(population_theta)\n"
)
_INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
_STANDARD_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


class CampsScientificDriverError(RuntimeError):
    pass


def select_minimum_finite_discrepancy(discrepancy: Any) -> tuple[int, float]:
    """Return the index/value of the minimum finite posterior discrepancy."""
    values = np.asarray(discrepancy, dtype=float).reshape(-1)
    if values.size == 0:
        raise CampsScientificDriverError("posterior discrepancy population is empty")
    finite = np.isfinite(values)
    if not finite.any():
        raise CampsScientificDriverError("posterior discrepancy population has no finite values")
    indexes = np.flatnonzero(finite)
    local = int(np.argmin(values[finite]))
    index = int(indexes[local])
    return index, float(values[index])


def expand_independent_ecg_to_standard_12(ecg: Any) -> np.ndarray:
    """Expand upstream [I, II, V1..V6] into the standard 12-lead ordering.

    The augmented/derived limb leads are reconstructed algebraically from I and
    II; no information from held-out data is used.
    """
    array = np.asarray(ecg, dtype=float)
    if array.ndim != 2 or array.shape[0] != len(_INDEPENDENT_LEADS):
        raise CampsScientificDriverError(
            "Camps representative ECG must have shape (8, n_samples) in I, II, V1..V6 order"
        )
    if array.shape[1] < 1 or not np.isfinite(array).all():
        raise CampsScientificDriverError("Camps representative ECG is empty or non-finite")
    lead_i, lead_ii = array[0], array[1]
    lead_iii = lead_ii - lead_i
    avr = -(lead_i + lead_ii) / 2.0
    avl = lead_i - lead_ii / 2.0
    avf = lead_ii - lead_i / 2.0
    return np.vstack((lead_i, lead_ii, lead_iii, avr, avl, avf, array[2:]))


def _parameter_units(names: list[str]) -> dict[str, str]:
    units: dict[str, str] = {}
    for name in names:
        lowered = str(name).lower()
        units[str(name)] = "cm/ms" if "speed" in lowered else "1"
    return units


def _instrument_source(source: str, *, plan_path: Path, output_path: Path) -> str:
    count = source.count(_INSERTION_MARKER)
    if count != 1:
        raise CampsScientificDriverError(
            f"audited Camps insertion marker must occur exactly once; found {count}"
        )
    hook = (
        _INSERTION_MARKER
        + "        from cardiep.camps_scientific_driver import export_from_upstream_namespace as _virelion_export\n"
        + f"        _virelion_export(globals(), {str(plan_path)!r}, {str(output_path)!r})\n"
        + "        raise SystemExit(0)\n"
    )
    return source.replace(_INSERTION_MARKER, hook, 1)


def _require_namespace(namespace: dict[str, Any], name: str) -> Any:
    if name not in namespace or namespace[name] is None:
        raise CampsScientificDriverError(f"instrumented upstream namespace is missing {name}")
    return namespace[name]


def _representative_simulation(evaluator: Any, selected_parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adapter = evaluator.adapter
    modules = adapter.distribute_parameter_to_modules(parameter_particle=selected_parameter)
    ecg = np.asarray(
        evaluator.simulator.simulate_particle(parameter_particle_modules_dict=modules),
        dtype=float,
    )
    electrophysiology = evaluator.simulator.electrophysiology_model
    activation, _vm = electrophysiology.simulate_electrophysiology(
        parameter_particle_modules_dict=modules
    )
    activation = np.asarray(activation, dtype=float).reshape(-1)
    if activation.size == 0 or not np.isfinite(activation).all():
        raise CampsScientificDriverError("selected Camps activation map is empty or non-finite")
    return ecg, activation


def _posterior_payload(
    *,
    parameter_names: list[str],
    theta_names: list[str],
    population_parameter: np.ndarray,
    population_theta: np.ndarray,
    discrepancy: np.ndarray,
    selected_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": "cardiep-camps-posterior-v1",
        "selection": {
            "method": "minimum_finite_discrepancy_same_inference_evaluator",
            "selected_index": selected_index,
            "discrepancy_metric": "rmse_pcc_cubic",
        },
        "parameter_names": parameter_names,
        "theta_names": theta_names,
        "parameter_population": population_parameter.tolist(),
        "theta_population": population_theta.tolist(),
        "discrepancy": discrepancy.tolist(),
    }


def export_from_upstream_namespace(
    namespace: dict[str, Any],
    plan_path: str | Path,
    output_path: str | Path,
) -> None:
    """Export a complete calibration result at the audited upstream hook point."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    request = EPCalibrationRequest.model_validate(plan["request"])
    if request.backend != CAMPS_CDT_BACKEND_NAME:
        raise CampsScientificDriverError("driver received a non-Camps calibration request")
    upstream_commit, workspace, source_rates, resampling_method = (
        CampsCardiacDigitalTwinExternalBackend.validate_calibration_request(request)
    )
    if upstream_commit != AUDITED_UPSTREAM_COMMIT:
        raise CampsScientificDriverError(
            "qualified driver only supports the audited upstream Camps commit"
        )
    if str(plan.get("upstream_commit", "")) != upstream_commit:
        raise CampsScientificDriverError("runtime plan upstream commit differs from request")
    if str(plan.get("workspace_manifest_fingerprint", "")).lower() != workspace.fingerprint:
        raise CampsScientificDriverError("runtime plan workspace fingerprint differs from request")

    inference_method = _require_namespace(namespace, "inference_method")
    population_parameter = np.asarray(
        _require_namespace(namespace, "population_parameter"), dtype=float
    )
    population_theta = np.asarray(_require_namespace(namespace, "population_theta"), dtype=float)
    inference_success = bool(_require_namespace(namespace, "inference_success"))
    if population_parameter.ndim != 2 or population_parameter.shape[0] < 1:
        raise CampsScientificDriverError("upstream parameter population has invalid shape")
    if population_theta.ndim != 2 or population_theta.shape[0] != population_parameter.shape[0]:
        raise CampsScientificDriverError("theta and parameter populations have incompatible shapes")
    if not np.isfinite(population_parameter).all() or not np.isfinite(population_theta).all():
        raise CampsScientificDriverError("posterior populations contain non-finite values")

    evaluator = inference_method.evaluator
    discrepancy = np.asarray(
        evaluator.evaluate_parameter_population(parameter_population=population_parameter),
        dtype=float,
    ).reshape(-1)
    if discrepancy.shape[0] != population_parameter.shape[0]:
        raise CampsScientificDriverError("discrepancy population length does not match posterior")
    selected_index, selected_discrepancy = select_minimum_finite_discrepancy(discrepancy)
    selected_parameter = population_parameter[selected_index]
    selected_theta = population_theta[selected_index]

    adapter = evaluator.adapter
    parameter_names = [str(item) for item in adapter.parameter_name_list_in_order]
    theta_names = [str(item) for item in adapter.get_theta_names()]
    if len(parameter_names) != selected_parameter.shape[0]:
        raise CampsScientificDriverError("parameter names do not match selected parameter vector")
    if len(theta_names) != selected_theta.shape[0]:
        raise CampsScientificDriverError("theta names do not match selected theta vector")

    ecg_8, activation = _representative_simulation(evaluator, selected_parameter)
    ecg_12 = expand_independent_ecg_to_standard_12(ecg_8)
    parameter_values = {
        name: float(selected_parameter[index]) for index, name in enumerate(parameter_names)
    }
    units = _parameter_units(parameter_names)
    selected_parameters = EPParameterSet(values=parameter_values, units=units, source="calibrated")

    artifact_root = Path(plan["runtime_root"]).expanduser().resolve() / "scientific-result"
    posterior_ref = write_json_artifact(
        artifact_root,
        artifact_id=f"{request.subject_id}-camps-posterior",
        kind="posterior_population",
        payload=_posterior_payload(
            parameter_names=parameter_names,
            theta_names=theta_names,
            population_parameter=population_parameter,
            population_theta=population_theta,
            discrepancy=discrepancy,
            selected_index=selected_index,
        ),
        metadata={
            "selection_method": "minimum_finite_discrepancy_same_inference_evaluator",
            "selected_index": selected_index,
            "discrepancy_metric": "rmse_pcc_cubic",
        },
    )
    activation_ref = write_json_artifact(
        artifact_root,
        artifact_id=f"{request.subject_id}-camps-activation",
        kind="activation_map",
        payload={
            "schema_version": "cardiep-field-v1",
            "subject_id": request.subject_id,
            "units": "ms",
            "activation_ms": activation.tolist(),
            "selection_index": selected_index,
        },
        metadata={
            "units": "ms",
            "selection_index": selected_index,
            "selection_discrepancy": selected_discrepancy,
        },
    ).model_copy(update={"coordinate_frame": request.anatomy_ref.coordinate_frame})
    ecg_ref = write_json_artifact(
        artifact_root,
        artifact_id=f"{request.subject_id}-camps-calibration-ecg",
        kind="calibration_ecg",
        payload={
            "schema_version": "cardiep-ecg-v1",
            "subject_id": request.subject_id,
            "lead_names": list(_STANDARD_12),
            "sample_rate_hz": UPSTREAM_QRS_SAMPLE_RATE_HZ,
            "values": ecg_12.tolist(),
            "units": "normalized",
            "source_leads": list(_INDEPENDENT_LEADS),
            "selection_index": selected_index,
        },
        metadata={
            "lead_names": list(_STANDARD_12),
            "sample_rate_hz": UPSTREAM_QRS_SAMPLE_RATE_HZ,
            "role": "calibration_forward_model",
            "units": "normalized",
            "selection_index": selected_index,
        },
    )

    simulation_request = EPSimulationRequest(
        subject_id=request.subject_id,
        anatomy_ref=request.anatomy_ref,
        backend=request.backend,
        parameters=selected_parameters,
        observations=[],
        settings=request.settings,
    )
    simulation_provenance = {
        "engine": UPSTREAM_REPOSITORY,
        "upstream_commit": upstream_commit,
        "wrapper_version": DRIVER_VERSION,
        "paper_doi": UPSTREAM_DOI,
        "workspace_manifest_fingerprint": workspace.fingerprint,
        "request_sha256": sha256_json(simulation_request.model_dump(mode="json")),
        "representative_selection": "minimum_finite_discrepancy_same_inference_evaluator",
        "selected_index": selected_index,
        "selected_discrepancy": selected_discrepancy,
    }
    simulated = EPSimulationResult(
        subject_id=request.subject_id,
        backend=request.backend,
        parameters=selected_parameters,
        outputs=[activation_ref, ecg_ref],
        validation_status="software_checked",
        warnings=[
            "Camps QRS forward ECG is normalized for inverse-model calibration and is not a physical-voltage clinical forward solution."
        ],
        provenance=simulation_provenance,
    )
    calibration_provenance = {
        "engine": UPSTREAM_REPOSITORY,
        "upstream_commit": upstream_commit,
        "wrapper_version": DRIVER_VERSION,
        "paper_doi": UPSTREAM_DOI,
        "workspace_manifest_fingerprint": workspace.fingerprint,
        "ecg_preprocessing": {
            "source_sample_rates_hz": [float(item) for item in source_rates],
            "target_sample_rate_hz": UPSTREAM_QRS_SAMPLE_RATE_HZ,
            "method": resampling_method,
        },
        "request_sha256": sha256_json(request.model_dump(mode="json")),
        "representative_selection": "minimum_finite_discrepancy_same_inference_evaluator",
        "selected_index": selected_index,
        "selected_discrepancy": selected_discrepancy,
    }
    result = EPCalibrationResult(
        subject_id=request.subject_id,
        backend=request.backend,
        parameters=selected_parameters,
        objective=selected_discrepancy,
        converged=inference_success,
        posterior_ref=posterior_ref,
        simulated=simulated,
        diagnostics={
            "posterior_population_size": int(population_parameter.shape[0]),
            "inference_method": "smc_abc",
            "discrepancy_metric": "rmse_pcc_cubic",
            "representative_selection": "minimum_finite_discrepancy_same_inference_evaluator",
            "selected_index": selected_index,
            "selected_discrepancy": selected_discrepancy,
            "selected_theta": {
                name: float(selected_theta[index]) for index, name in enumerate(theta_names)
            },
            "inference_success": inference_success,
        },
        provenance=calibration_provenance,
    )
    # Validate the exact result before exposing it to the outer subprocess adapter.
    CampsCardiacDigitalTwinExternalBackend.validate_calibration_result(
        request,
        result,
        upstream_commit=upstream_commit,
        workspace=workspace,
        source_sample_rates=source_rates,
        resampling_method=resampling_method,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_qualified_driver(plan_path: str | Path, output_path: str | Path) -> None:
    plan_file = Path(plan_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise CampsScientificDriverError("Camps runtime plan must be a JSON object")
    if str(plan.get("upstream_commit", "")) != AUDITED_UPSTREAM_COMMIT:
        raise CampsScientificDriverError(
            "qualified driver supports only the audited upstream Camps commit"
        )
    source_path = Path(plan["upstream_entrypoint"]).expanduser().resolve()
    if not source_path.is_file():
        raise CampsScientificDriverError(f"upstream entrypoint is missing: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    instrumented = _instrument_source(source, plan_path=plan_file, output_path=output_file)
    instrumented_path = Path(plan["runtime_root"]).expanduser().resolve() / "instrumented_personalise_to_QRS.py"
    instrumented_path.write_text(instrumented, encoding="utf-8")

    process = subprocess.run(
        [os.environ.get("CARDIEP_CAMPS_PYTHON", os.sys.executable), str(instrumented_path)],
        cwd=str(source_path.parent),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg")},
    )
    if process.returncode:
        raise CampsScientificDriverError(
            process.stderr.strip()
            or process.stdout.strip()
            or f"instrumented Camps workflow exited with {process.returncode}"
        )
    if not output_file.is_file():
        raise CampsScientificDriverError("instrumented Camps workflow produced no result JSON")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualified Camps QRS posterior-selection driver")
    parser.add_argument("plan")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    try:
        run_qualified_driver(args.plan, args.output)
    except (CampsScientificDriverError, OSError, TypeError, ValueError) as exc:
        print(f"cardiep-camps-qualified-driver: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
