# Scientific validation tier

This directory separates **software correctness**, **numerical verification**, **cross-solver agreement**, and **empirical validation**. Passing one tier never upgrades a claim into a higher tier automatically.

## Validation ladder

1. **Software reference checks**
   - deterministic tetrahedral Eikonal edge travel times;
   - typed contracts, units, checksums, roots, ECG timing;
   - regression tests.
   - Command: `cardiep validate-reference`.

2. **Numerical verification**
   - mesh/time-step refinement;
   - observed convergence order;
   - Grid Convergence Index (GCI)-style fine-grid estimate;
   - manufactured/exact solutions where available.

3. **Cross-solver verification**
   - same benchmark geometry, ionic model, conductivities, stimulus, activation threshold, sampling points, and discretization;
   - compare openCARP, FEniCSx-beat, MonoAlg3D or another independent PDE solver through the canonical activation-profile contract.

4. **Synthetic inverse recovery**
   - generate synthetic observations from held-out parameters;
   - infer parameters without exposing the truth;
   - quantify bias, RMSE, interval coverage, rank calibration, and failure rate over repeated seeds.

5. **Empirical validation**
   - measured ECG/EAM data;
   - held-out observables and/or held-out mapping points;
   - no parameter tuning on the evaluation target;
   - patient-level splits and explicit population scope.

Clinical validation requires an additional study design and is outside this repository.

---

## Niederer N-version benchmark

The canonical benchmark ID is:

```
niederer-2011
```

The contract follows the classic N-version monodomain benchmark:

- slab: **20 × 7 × 3 mm**;
- fibre direction: +X;
- corner stimulus cube: **1.5 × 1.5 × 1.5 mm**;
- zero-flux boundaries;
- ten Tusscher–Panfilov 2006 epicardial model;
- local activation time: first crossing of **0 mV**;
- points P1–P9 defined by `cardiep niederer-spec`;
- standard resolution grid: `dx={0.5,0.2,0.1} mm`, `dt={0.05,0.01,0.005} ms`.

Primary benchmark reference:

- Niederer SA et al. *Verification of cardiac tissue electrophysiology simulators using an N-version benchmark.* Phil Trans R Soc A. 2011. DOI: 10.1098/rsta.2011.0139.

The interoperability fixture in
`validation/niederer/fenicsx-beat-0.7.0-dx0.1-dt0.005.json`
is transcribed from `finsberg/fenicsx-beat` commit
`7ab18453ae57c28d798b744234157f36297113d5`,
which implements the same benchmark and publishes the P1–P9 activation table.

That fixture is a **pipeline/reference artifact**, not a substitute for rerunning the external solver in a publication.

---

## Canonical activation profile

Every solver is normalized to:

```json
{
  "schema_version": "cardiep-activation-profile-v1",
  "benchmark_id": "niederer-2011",
  "solver": {
    "name": "solver-name",
    "version": "version-or-null",
    "commit": "commit-or-null"
  },
  "equation": "monodomain",
  "ionic_model": "tenTusscher-Panfilov-2006-epicardial",
  "coordinate_unit": "mm",
  "time_unit": "ms",
  "sample_points": [
    {"id": "P1", "xyz": [0, 0, 0]}
  ],
  "activation": {
    "P1": 1.2
  },
  "metadata": {}
}
```

CardiEP converts coordinates internally to cm and activation times to ms. Sample IDs must match exactly and coordinates must agree within the requested tolerance. A solver cannot silently compare a different point set.

### CSV bridge

For solvers that emit CSV after post-processing:

```bash
cardiep profile-from-csv activation.csv activation-profile.json \
  --benchmark-id niederer-2011 \
  --solver-name openCARP \
  --solver-version 20 \
  --coordinate-unit mm \
  --time-unit ms \
  --id-column id \
  --x-column x_mm --y-column y_mm --z-column z_mm \
  --activation-column activation_ms
```

`validation/niederer/profile-template.csv` already contains the P1–P9 coordinates.

---

## External solver recipes

### FEniCSx-beat

Current FEniCSx-beat includes both a Python demo and CLI template for the Niederer benchmark.

Typical CLI route:

```bash
beat init case/config.toml --template niederer_benchmark
beat validate-config case/config.toml
beat run case/config.toml
beat post case/config.toml
```

For a resolution study, override the spatial and temporal resolution for every run and export P1–P9 activation times. Preserve the exact resolved configuration beside the result.

### openCARP

openCARP provides a Niederer numerical-discretization example plus `carputils` workflows for local activation time extraction.

Run the same geometry/ionic/conductivity/stimulus problem for each `dx` and `dt`, extract LAT at P1–P9, then fill the common CSV template or produce the canonical JSON directly.

Do not compare an openCARP Eikonal/reaction-Eikonal run to a monodomain reference and call it monodomain cross-solver agreement. Equation class must match.

### MonoAlg3D

MonoAlg3D is a finite-volume monodomain solver and can export text/VTK results. For a valid comparison:

- disable adaptive spatial refinement unless it is explicitly part of the study;
- document PDE/ODE time steps and CUDA/CPU solver configuration;
- use the same slab, conductivities, TP06 epicardial model, stimulus, and 0 mV LAT definition;
- post-process P1–P9 into the same CSV/JSON contract.

The comparison layer is intentionally solver-neutral; solver-specific scripts should only translate output, never alter the scientific benchmark.

---

## Pairwise comparison

Without a gate:

```bash
cardiep compare-activation \
  reference.json candidate.json \
  --alignment absolute \
  --output comparison.json
```

With an engineering acceptance gate:

```bash
cardiep compare-activation \
  reference.json candidate.json \
  --alignment absolute \
  --rmse-ms-max 2.0 \
  --max-abs-ms-max 5.0 \
  --correlation-min 0.995 \
  --abs-bias-ms-max 2.0 \
  --output comparison.json
```

Those thresholds are **study choices**, not universal physiological truth. Publication thresholds must be justified before looking at the candidate result.

Metrics include RMSE, MAE, maximum absolute error, mean bias, 95% limits of agreement, correlation, linear slope/intercept, and normalized RMSE.

`p1-relative` alignment is available for studies that explicitly separate stimulus-onset latency from propagation pattern. Do not use it to hide an unexplained absolute timing error.

---

## Mesh-convergence manifest

```json
{
  "schema_version": "cardiep-convergence-manifest-v1",
  "levels": [
    {"label": "coarse", "h": 0.05, "dt_ms": 0.05, "profile": "coarse.json"},
    {"label": "medium", "h": 0.02, "dt_ms": 0.01, "profile": "medium.json"},
    {"label": "fine", "h": 0.01, "dt_ms": 0.005, "profile": "fine.json"}
  ],
  "exact_profile": null
}
```

Run:

```bash
cardiep validate-convergence convergence.json --output convergence-report.json
```

The report contains:

- consecutive-grid RMSE;
- exact-solution RMSE when an exact/manufactured profile is supplied;
- observed order where defined;
- approximate uniform-refinement check;
- fine-grid GCI-style estimate;
- monotone self-convergence flag.

Spatial and temporal convergence should be separated when making numerical claims: hold `dt` sufficiently fine while varying `dx`, then hold `dx` sufficiently fine while varying `dt`.

---

## Required publication bundle

For each solver/run preserve:

- solver name, release, commit/container digest;
- benchmark specification;
- resolved configuration;
- mesh generator and mesh checksum;
- geometry coordinate units;
- ionic model source/version and initial state;
- conductivities and membrane parameters;
- stimulus definition;
- PDE and ODE time steps;
- linear/nonlinear solver tolerances;
- hardware/backend;
- raw activation map or voltage output;
- P1–P9 extracted activation profile;
- conversion script/version;
- comparison and convergence reports.

A result should not be labeled `numerically_checked` simply because a process exited successfully.

## Comparison safeguards (0.3.0)

Profiles must agree on declared equation class and ionic model as well as benchmark, sample IDs, and coordinates. Two undeclared model fields remain accepted for legacy numerical comparisons; this does not verify model equivalence. Constant profiles have undefined correlation and fail a correlation gate.

Self-convergence order is omitted for nonuniform refinement ratios. GCI-style estimates require an approximately uniform sequence, a positive latest order, decreasing consecutive differences, and unchanged declared time steps. If every time step is absent, the estimate is conditional on a stationary model or unreported fixed time control; `time_step_metadata_complete` is false. Such an estimate must not be presented as verified isolation of spatial error in a time-dependent PDE.

The pinned Niederer fixture was checked against its upstream source table during the CPU audit. This is a transcription check. No independent monodomain/bidomain solver was rerun in that audit.
