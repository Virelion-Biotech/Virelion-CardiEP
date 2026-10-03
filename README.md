# Virelion-CardiEP

[![CI](https://github.com/Virelion-Biotech/Virelion-CardiEP/actions/workflows/ci.yml/badge.svg)](https://github.com/Virelion-Biotech/Virelion-CardiEP/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

**CardiEP is the patient-specific cardiac electrophysiology engine and backend gateway for the Virelion HeartTwin stack.**

It now combines stable cross-service contracts with a real fast EP engine:

- tetrahedral anatomy loading from JSON/NPZ and optional VTK/VTU/Gmsh via `meshio`;
- direct consumption of CardiAnatomy `AnatomyBundle` artifacts;
- fibre/sheet/normal anisotropic graph-Eikonal propagation;
- scar-core and border-zone conduction modifiers;
- explicit ventricular roots, root activation delays, and Purkinje-distance timing;
- endocardial farthest-point root generation as a declared heuristic;
- ventricular-coordinate APD gradients and repolarization maps;
- deterministic standard 12-lead pseudo-ECG generation from electrode geometry;
- ECG/activation/repolarization discrepancy functions;
- bounded deterministic calibration for fast reference fits;
- ElectroTrace calibration-artifact ingestion;
- pluggable heavyweight EP backends through the `cardiep.backends` entry-point group;
- a safe subprocess contract for wrappers around openCARP, MonoAlg3D, FEniCSx-beat, or other engines;
- provenance-linked output artifacts and deterministic reference checks.

> CardiEP is research software. The native backend is deliberately a **fast integration/reference model**, not a replacement for validated monodomain/bidomain solvers and not a clinical device.

## Install

Core engine:

```bash
python -m pip install -e .
```

Development:

```bash
python -m pip install -e '.[dev]'
pytest -q
cardiep doctor
cardiep validate-reference
```

Optional VTK/VTU/Gmsh loading:

```bash
python -m pip install -e '.[io]'
```

## Native backend

The built-in backend is:

```text
numpy-eikonal-v1
```

It performs fast organ-level activation/repolarization simulation without requiring PETSc, MPI, FEniCSx, CUDA, or an external solver binary. This makes it useful for:

- HeartTwin integration tests;
- rapid parameter sweeps;
- initialization of more expensive inverse problems;
- sensitivity analysis;
- proposal screening inside CardiInfer;
- debugging anatomy, fibre, scar, root, and electrode plumbing;
- generating reproducible software-reference outputs before escalation to a PDE backend.

It is **not** intended to claim numerical equivalence to openCARP, MonoAlg3D, Chaste, FEniCSx-beat, or other biophysical PDE engines.

## Minimal simulation

The anatomy artifact must state its coordinate units explicitly.

```json
{
  "subject_id": "S1",
  "anatomy_ref": {
    "artifact_id": "s1-ep-geometry",
    "kind": "ep_geometry",
    "uri": "file:///absolute/path/geometry.json"
  },
  "backend": "numpy-eikonal-v1",
  "parameters": {
    "values": {
      "fibre_speed": 0.065,
      "sheet_speed": 0.051,
      "normal_speed": 0.048,
      "purkinje_speed": 0.30,
      "apd_min_ms": 240.0,
      "apd_max_ms": 320.0,
      "apd_gradient_tm": 1.0
    },
    "source": "fixed"
  },
  "settings": {
    "root_nodes": [0],
    "ecg_sample_rate_hz": 500.0,
    "output_dir": "runs/S1"
  }
}
```

Run it:

```bash
cardiep simulate request.json
```

Outputs are file-backed `ArtifactRef` objects for activation, repolarization, pseudo-ECG (when electrodes exist), and an EP summary, each with a SHA-256 digest.

## ElectroTrace → CardiEP → CardiInfer

ElectroTrace produces calibration-grade measured electrical artifacts. CardiEP validates them as `EPObservation` objects, and CardiInfer uses the same artifact as `LikelihoodTerm.observation_ref`.

```text
ECG / EAM
   │
   ▼
ElectroTrace
   │ measured waveform/map + QC + uncertainty + hash
   ▼
CardiEP EPObservation
   │
   ├──► numpy-eikonal-v1 / external PDE backend
   │         │
   │         └──► activation + repolarization + ECG
   │
   ▼
CardiInfer likelihood / posterior inference
```

CardiEP also exposes `ep.calibrate` with a deterministic bounded coordinate-pattern search. That is useful for fast reference fits, but posterior inference belongs in CardiInfer.

## Anatomy contract

The native loader accepts:

1. compact CardiEP JSON:
   `node_xyz`, `tetrahedra`, explicit `units`, and optional fibres, scar, ventricular coordinates, electrodes and roots;
2. NPZ with the equivalent arrays;
3. VTK / VTU / Gmsh when `meshio` is installed;
4. a CardiAnatomy `AnatomyBundle` containing a `volume_mesh` plus optional `fiber_field`, `coordinate_field`, and `scar_map` artifacts.

Coordinates are converted internally to **cm**. Conduction velocity parameters are in **cm/ms**. Activation, APD and repolarization are in **ms**.

## Backends

```bash
cardiep backends
cardiep ecosystem
```

Heavy engines are intentionally not dependencies of the core package. A backend plugin implements the existing `EPBackend` protocol and registers under the `cardiep.backends` entry-point group.

The recommended ecosystem includes:

- **Cardiac-Digital-Twin** — digital-twinning architecture, Eikonal/reaction-Eikonal, ECG personalization;
- **FEniCSx-beat** — finite-element monodomain;
- **MonoAlg3D_C** — high-performance finite-volume monodomain/GPU;
- **openCARP** — mature multi-scale mono/bidomain and lead-field workflows;
- **pyCEPS** — CARTO/EnSite EAM translation;
- **Cobiveco / LDRB** — coordinates and myocardial microstructure.

See `docs/RESEARCH_MAP.md` and `docs/BACKENDS.md`.

## Reference validation

```bash
cardiep validate-reference
```

This checks deterministic software invariants including analytic edge travel times, orthotropic directionality, repolarization ordering, standard 12-lead construction, and exact repeatability.

Those checks are **software tests**, not physiological validation.

## Architecture

```text
CardiAnatomy
    │
    ▼
EPGeometry
    │
    ├── Conduction roots / Purkinje timing
    ▼
Anisotropic Eikonal propagation
    │
    ├── scar / border-zone scaling
    ▼
Activation map
    │
    ├── APD / ventricular-coordinate gradients
    ▼
Repolarization map
    │
    ├── electrode geometry
    ▼
Pseudo ECG
    │
    ├── ElectroTrace / EAM observations
    ▼
Discrepancy
    │
    ├── native fast calibration
    └── CardiInfer posterior inference
```


## Scientific validation tier

CardiEP now includes a solver-neutral numerical validation layer around the native engine and external PDE backends:

- canonical activation-profile JSON with strict units and sample-point identity;
- the Niederer 2011 N-version benchmark specification;
- a pinned FEniCSx-beat 0.7.0 interoperability fixture;
- CSV ingestion for openCARP, MonoAlg3D, FEniCSx-beat or other solver post-processing;
- cross-solver RMSE/MAE/max-error/bias/limits-of-agreement/correlation/regression metrics;
- explicit, configurable acceptance gates;
- mesh self-convergence, observed order and fine-grid GCI-style estimates.

Examples:

```bash
cardiep niederer-spec --output niederer.json
cardiep niederer-fenicsx-reference --output fenicsx-reference.json

cardiep profile-from-csv openCARP.csv opencarp.json \
  --benchmark-id niederer-2011 \
  --solver-name openCARP \
  --coordinate-unit mm --time-unit ms \
  --x-column x_mm --y-column y_mm --z-column z_mm \
  --activation-column activation_ms

cardiep compare-activation fenicsx-reference.json opencarp.json \
  --rmse-ms-max 2 --max-abs-ms-max 5 \
  --correlation-min 0.995 --abs-bias-ms-max 2
```

See `docs/SCIENTIFIC_VALIDATION.md` for the cross-solver, convergence, provenance and claim-boundary protocol. Passing these numerical checks does **not** establish physiological or clinical validity.

## Scientific boundary

The built-in pseudo-ECG is an inverse-distance nodal-dipole proxy in arbitrary normalized units. The built-in propagation model is graph-based Eikonal propagation. Both are useful for fast digital-twin plumbing and inverse-loop screening, but neither establishes clinical validity.

Escalate publication-critical claims to an independently verified PDE/ionic backend and perform numerical and empirical validation against the intended dataset and population.

## License

AGPL-3.0-or-later. External projects retain their own licenses and are not silently vendored.
