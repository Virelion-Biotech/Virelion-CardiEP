"""Reproducible CPU numerical verification and synthetic inverse recovery."""

import argparse
import json
import platform
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from cardiep import (
    ArtifactRef,
    CardiEPService,
    EPCalibrationRequest,
    EPObservation,
    EPParameterSet,
    EPSimulationRequest,
    RootSchedule,
    anisotropic_eikonal,
    run_reference_validation,
)
from cardiep.provenance import file_sha256, uri_to_path
from cardiep.validation import (
    _structured_cube_geometry,
    run_eikonal_refinement_validation,
    synthetic_tetra_geometry,
)


def planar_front():
    base = _structured_cube_geometry(3)
    count = base.n_nodes
    geometry = replace(
        base,
        fibre=np.tile([1, 0, 0], (count, 1)),
        sheet=np.tile([0, 1, 0], (count, 1)),
        normal=np.tile([0, 0, 1], (count, 1)),
    )
    roots = np.flatnonzero(geometry.node_xyz_cm[:, 0] == 0)
    result = anisotropic_eikonal(
        geometry,
        RootSchedule(roots, np.zeros(len(roots)), "analytic-planar-front"),
        {"fibre_speed": 0.1, "sheet_speed": 0.05, "normal_speed": 0.025},
    )
    exact = geometry.node_xyz_cm[:, 0] / 0.1
    error = float(np.max(np.abs(result.activation_ms - exact)))
    return {
        "n_nodes": count,
        "n_tetrahedra": len(geometry.tetrahedra),
        "max_error_ms": error,
        "passed": error < 1e-8,
    }


def synthetic_inverse():
    truth = {"fibre_speed": 0.075, "sheet_speed": 0.04, "normal_speed": 0.03, "apd_ms": 290.0}
    with tempfile.TemporaryDirectory(prefix="cardiep-inverse-") as temporary:
        root = Path(temporary)
        geometry = synthetic_tetra_geometry(with_fibres=True)
        path = root / "geometry.json"
        path.write_text(
            json.dumps(
                {
                    "units": "cm",
                    "node_xyz": geometry.node_xyz_cm.tolist(),
                    "tetrahedra": geometry.tetrahedra.tolist(),
                    "fibre": geometry.fibre.tolist(),
                    "sheet": geometry.sheet.tolist(),
                    "normal": geometry.normal.tolist(),
                    "root_nodes": [0],
                }
            )
        )
        anatomy = ArtifactRef(
            artifact_id="synthetic-geometry",
            kind="ep_geometry",
            uri=path.as_uri(),
            sha256=file_sha256(path),
        )
        service = CardiEPService()
        generated = service.simulate(
            EPSimulationRequest(
                subject_id="synthetic",
                anatomy_ref=anatomy,
                backend="numpy-eikonal-v1",
                parameters=EPParameterSet(values=truth),
                settings={"with_ecg": False, "output_dir": str(root / "truth")},
            )
        )
        observations = [
            EPObservation(observation_id=output.kind, kind=output.kind, artifact=output, units="ms")
            for output in generated.outputs
            if output.kind in {"activation_map", "repolarization_map"}
        ]
        hints = [
            {
                "term_id": obs.kind,
                "observation_id": obs.observation_id,
                "model_output": obs.kind,
                "discrepancy": "rmse",
                "weight": 1.0,
            }
            for obs in observations
        ]
        fitted = service.calibrate(
            EPCalibrationRequest(
                subject_id="synthetic",
                anatomy_ref=anatomy,
                backend="numpy-eikonal-v1",
                observations=observations,
                parameter_bounds={
                    "fibre_speed": (0.05, 0.12),
                    "sheet_speed": (0.02, 0.08),
                    "normal_speed": (0.01, 0.06),
                    "apd_ms": (200, 350),
                },
                initial_parameters=EPParameterSet(
                    values={
                        "fibre_speed": 0.1,
                        "sheet_speed": 0.065,
                        "normal_speed": 0.02,
                        "apd_ms": 240,
                    }
                ),
                settings={
                    "with_ecg": False,
                    "output_dir": str(root / "fit"),
                    "likelihood_hints": hints,
                    "max_iterations": 160,
                    "step_tolerance": 1e-7,
                },
            )
        )
        relative = {
            name: abs(fitted.parameters.values[name] - value) / abs(value)
            for name, value in truth.items()
        }
        output_hashes_valid = all(
            file_sha256(uri_to_path(item.uri)) == item.sha256 for item in fitted.simulated.outputs
        )
        return {
            "truth": truth,
            "recovered": fitted.parameters.values,
            "relative_errors": relative,
            "objective": fitted.objective,
            "converged": fitted.converged,
            "artifact_hashes_valid": output_hashes_valid,
            "passed": max(relative.values()) < 1e-4 and output_hashes_valid and fitted.converged,
            "scope": "noiseless self-model parameter recovery, not independent biological validation",
        }


def run():
    reference = run_reference_validation()
    refinement = run_eikonal_refinement_validation(levels=(4, 8, 12))
    planar = planar_front()
    inverse = synthetic_inverse()
    report = {
        "version": "0.3.0",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "software_reference": reference,
        "isotropic_refinement": refinement,
        "orthotropic_planar_front": planar,
        "synthetic_inverse": inverse,
        "passed": all(item["passed"] for item in (reference, refinement, planar, inverse)),
        "pde_solver_rerun": False,
        "empirical_validation": False,
        "clinical_validation": False,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/cpu/results.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(args.output),
                "inverse": result["synthetic_inverse"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
