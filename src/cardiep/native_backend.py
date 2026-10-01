from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_artifact
from .conduction import resolve_root_schedule
from .discrepancy import DiscrepancyReport, evaluate_observations
from .ecg import ECGResult, pseudo_ecg
from .geometry import EPGeometry, load_ep_geometry
from .models import (
    EPCalibrationRequest,
    EPCalibrationResult,
    EPParameterSet,
    EPSimulationRequest,
    EPSimulationResult,
)
from .propagation import PropagationResult, anisotropic_eikonal
from .provenance import sha256_json
from .repolarization import RepolarizationResult, apd_map

NATIVE_BACKEND_NAME = "numpy-eikonal-v1"


@dataclass(frozen=True)
class NativeSimulation:
    geometry: EPGeometry
    propagation: PropagationResult
    repolarization: RepolarizationResult
    ecg: ECGResult | None
    warnings: tuple[str, ...]


class NativeEikonalBackend:
    """Dependency-light EP backend for twinning/integration and fast inverse loops.

    It implements anisotropic graph Eikonal propagation, scar conduction modifiers,
    root/Purkinje timing, ventricular-coordinate APD gradients and an exploratory
    pseudo-ECG observation model. It is not a monodomain/bidomain replacement.
    """

    name = NATIVE_BACKEND_NAME

    def available(self) -> bool:
        return True

    def _simulate_arrays(
        self,
        *,
        anatomy_ref,
        parameters: dict[str, float],
        observations,
        settings: dict[str, Any],
    ) -> NativeSimulation:
        geometry = load_ep_geometry(anatomy_ref, settings)
        roots = resolve_root_schedule(geometry, settings, parameters)
        propagation = anisotropic_eikonal(geometry, roots, parameters)
        repolarization = apd_map(geometry, propagation.activation_ms, parameters)
        warnings = list(propagation.warnings)

        needs_ecg = any(item.kind == "ecg" for item in observations)
        with_ecg = bool(settings.get("with_ecg", bool(geometry.electrodes_cm) or needs_ecg))
        ecg = None
        if with_ecg:
            ecg = pseudo_ecg(
                geometry,
                propagation.activation_ms,
                repolarization.repolarization_ms,
                sample_rate_hz=float(settings.get("ecg_sample_rate_hz", 500.0)),
                duration_ms=None if settings.get("duration_ms") is None else float(settings["duration_ms"]),
                qrs_sigma_ms=float(settings.get("qrs_sigma_ms", 5.0)),
                t_sigma_ms=float(settings.get("t_sigma_ms", 20.0)),
                repolarization_scale=float(settings.get("repolarization_scale", 0.55)),
                chunk_size=int(settings.get("ecg_chunk_size", 2048)),
            )

        warnings.append(
            "Native numpy-eikonal-v1 is a fast research/integration model; "
            "pseudo-ECG output is not a clinical forward solver."
        )
        if propagation.root_schedule.method.endswith("heuristic"):
            warnings.append("Activation roots were chosen by an endocardial geometric heuristic.")
        return NativeSimulation(
            geometry=geometry,
            propagation=propagation,
            repolarization=repolarization,
            ecg=ecg,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _output_dir(subject_id: str, request_sha: str, settings: dict[str, Any]) -> Path:
        configured = settings.get("output_dir")
        if configured:
            return Path(str(configured)).expanduser().resolve()
        return (Path.cwd() / "cardiep_runs" / subject_id / request_sha[:16]).resolve()

    def _result(
        self,
        request: EPSimulationRequest,
        simulation: NativeSimulation,
    ) -> EPSimulationResult:
        request_json = request.model_dump(mode="json")
        request_sha = sha256_json(request_json)
        output_dir = self._output_dir(request.subject_id, request_sha, request.settings)
        activation = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-activation-{request_sha[:12]}",
            kind="activation_map",
            payload={
                "schema_version": "cardiep-field-v1",
                "subject_id": request.subject_id,
                "units": "ms",
                "activation_ms": simulation.propagation.activation_ms.tolist(),
                "root_nodes": simulation.propagation.root_schedule.nodes.tolist(),
                "root_activation_ms": simulation.propagation.root_schedule.activation_ms.tolist(),
                "root_method": simulation.propagation.root_schedule.method,
            },
            metadata={"model": self.name, "n_nodes": simulation.geometry.n_nodes},
        )
        repolarization = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-repolarization-{request_sha[:12]}",
            kind="repolarization_map",
            payload={
                "schema_version": "cardiep-field-v1",
                "subject_id": request.subject_id,
                "units": "ms",
                "apd_ms": simulation.repolarization.apd_ms.tolist(),
                "repolarization_ms": simulation.repolarization.repolarization_ms.tolist(),
                "apd_method": simulation.repolarization.method,
            },
            metadata={"model": self.name, "n_nodes": simulation.geometry.n_nodes},
        )
        outputs = [activation, repolarization]
        if simulation.ecg is not None:
            outputs.append(
                write_json_artifact(
                    output_dir,
                    artifact_id=f"{request.subject_id}-ecg-{request_sha[:12]}",
                    kind="pseudo_ecg",
                    payload={
                        "schema_version": "cardiep-ecg-v1",
                        "subject_id": request.subject_id,
                        **simulation.ecg.to_dict(),
                    },
                    metadata={
                        "model": simulation.ecg.model,
                        "validation_status": "exploratory",
                    },
                )
            )
        summary = write_json_artifact(
            output_dir,
            artifact_id=f"{request.subject_id}-ep-summary-{request_sha[:12]}",
            kind="ep_summary",
            payload={
                "schema_version": "cardiep-summary-v1",
                "subject_id": request.subject_id,
                "backend": self.name,
                "request_sha256": request_sha,
                "geometry": simulation.geometry.summary(),
                "propagation": {
                    "activation_min_ms": float(np.min(simulation.propagation.activation_ms)),
                    "activation_max_ms": float(np.max(simulation.propagation.activation_ms)),
                    "qrs_activation_span_ms": float(np.ptp(simulation.propagation.activation_ms)),
                    "edge_count": simulation.propagation.edge_count,
                    "parameters": simulation.propagation.parameters,
                },
                "repolarization": {
                    "apd_min_ms": float(np.min(simulation.repolarization.apd_ms)),
                    "apd_max_ms": float(np.max(simulation.repolarization.apd_ms)),
                    "repolarization_min_ms": float(np.min(simulation.repolarization.repolarization_ms)),
                    "repolarization_max_ms": float(np.max(simulation.repolarization.repolarization_ms)),
                    "method": simulation.repolarization.method,
                },
                "warnings": list(simulation.warnings),
            },
            metadata={"model": self.name},
        )
        outputs.append(summary)
        return EPSimulationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=request.parameters,
            outputs=outputs,
            validation_status="software_checked",
            warnings=list(simulation.warnings),
            provenance={
                "engine": "Virelion-CardiEP",
                "backend": self.name,
                "request_sha256": request_sha,
                "anatomy_artifact_id": request.anatomy_ref.artifact_id,
                "root_method": simulation.propagation.root_schedule.method,
                "scientific_status": "integration/reference model; not clinically validated",
            },
        )

    def simulate(self, request: EPSimulationRequest) -> EPSimulationResult:
        simulation = self._simulate_arrays(
            anatomy_ref=request.anatomy_ref,
            parameters=dict(request.parameters.values),
            observations=request.observations,
            settings=dict(request.settings),
        )
        return self._result(request, simulation)

    def _objective(
        self,
        request: EPCalibrationRequest,
        parameters: dict[str, float],
    ) -> tuple[float, DiscrepancyReport]:
        simulation = self._simulate_arrays(
            anatomy_ref=request.anatomy_ref,
            parameters=parameters,
            observations=request.observations,
            settings=dict(request.settings),
        )
        report = evaluate_observations(
            request.observations,
            activation_ms=simulation.propagation.activation_ms,
            repolarization_ms=simulation.repolarization.repolarization_ms,
            ecg=simulation.ecg,
            hints=request.settings.get("likelihood_hints"),
        )
        return report.objective, report

    def calibrate(self, request: EPCalibrationRequest) -> EPCalibrationResult:
        if not request.parameter_bounds:
            raise ValueError("Native calibration requires at least one bounded parameter")
        parameters = dict(request.initial_parameters.values if request.initial_parameters else {})
        for name, (low, high) in request.parameter_bounds.items():
            low_value, high_value = float(low), float(high)
            if name in parameters:
                initial = float(parameters[name])
                if not low_value <= initial <= high_value:
                    raise ValueError(
                        f"Initial parameter {name!r}={initial} lies outside "
                        f"its calibration bounds [{low_value}, {high_value}]"
                    )
            else:
                parameters[name] = 0.5 * (low_value + high_value)

        step_fraction = float(request.settings.get("step_fraction", 0.25))
        max_iterations = int(request.settings.get("max_iterations", 24))
        tolerance = float(request.settings.get("step_tolerance", 1e-4))
        if not 0 < step_fraction <= 1 or max_iterations < 1 or tolerance <= 0:
            raise ValueError("Invalid native calibration controls")

        steps = {
            name: max((float(high) - float(low)) * step_fraction, tolerance)
            for name, (low, high) in request.parameter_bounds.items()
        }
        best, report = self._objective(request, parameters)
        history = [{"iteration": 0, "objective": best, "parameters": dict(parameters)}]
        converged = False

        for iteration in range(1, max_iterations + 1):
            improved = False
            for name, (low, high) in request.parameter_bounds.items():
                current = float(parameters[name])
                for direction in (-1.0, 1.0):
                    proposal = dict(parameters)
                    proposal[name] = float(np.clip(current + direction * steps[name], low, high))
                    if proposal[name] == current:
                        continue
                    objective, candidate_report = self._objective(request, proposal)
                    if objective + 1e-12 < best:
                        parameters = proposal
                        best = objective
                        report = candidate_report
                        improved = True
                        current = proposal[name]
            history.append(
                {"iteration": iteration, "objective": best, "parameters": dict(parameters)}
            )
            if not improved:
                steps = {name: value * 0.5 for name, value in steps.items()}
                if max(steps.values()) <= tolerance:
                    converged = True
                    break

        simulation_request = EPSimulationRequest(
            subject_id=request.subject_id,
            anatomy_ref=request.anatomy_ref,
            backend=self.name,
            parameters=EPParameterSet(
                values={key: float(value) for key, value in parameters.items()},
                units=dict(request.initial_parameters.units if request.initial_parameters else {}),
                source="calibrated",
            ),
            observations=request.observations,
            settings=request.settings,
        )
        simulated = self.simulate(simulation_request)
        return EPCalibrationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=simulation_request.parameters,
            objective=float(best),
            converged=converged,
            simulated=simulated,
            diagnostics={
                "method": "bounded-coordinate-pattern-search",
                "iterations": len(history) - 1,
                "history": history,
                "final_terms": report.to_dict()["terms"],
                "step_tolerance": tolerance,
            },
            provenance={
                "engine": "Virelion-CardiEP",
                "backend": self.name,
                "calibration_method": "bounded-coordinate-pattern-search",
                "scientific_status": "deterministic reference optimizer; not posterior inference",
            },
        )
