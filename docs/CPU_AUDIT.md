# CardiEP 0.3.0 CPU audit

Audited 2026-10-07, starting from main commit `889f8ce942a6062b5a023a20daa2c4cc671ddede`. All computations use CPU; no GPU is required.

## Evidence

The baseline suite passed 74 tests. New regression cases reproduced scientific-contract and product defects before the fixes. The expanded suite contains 122 tests, including 23 optional independent SciPy reference checks and a meshio multi-block import check. Core-only installations skip optional dependency checks; the dedicated CI reference job installs those dependencies.

| Numerical check | Result | Interpretation |
| --- | --- | --- |
| Tetrahedral local Hopf–Lax update | 12 randomized positive-definite metrics agree with independent constrained SciPy minimization within 2e-7 ms | Local update verification; not a whole-solver proof |
| Bounded surface calibration | 10 randomized cases agree with SciPy box-constrained least squares within 1e-7 | Speed and offset bounds are both honored |
| Surface propagation | Agrees with independent SciPy Dijkstra | Graph implementation verification |
| Native isotropic refinement | Errors 0.051947, 0.035379, 0.027617 ms at grids 4, 8, 12 | Monotone improvement at one off-axis sample; finest error is 53.16% of coarsest |
| Orthotropic planar front | Maximum error 0 ms over 64 nodes / 162 tetrahedra | Manufactured aligned-front verification |
| Synthetic joint inverse fit | Converged; maximum relative parameter error below 1e-9; objective 5.42e-8 | Noiseless self-model recovery of three speeds and constant APD |
| Output artifacts | All synthetic fitted artifact hashes verified | File integrity |

The complete deterministic numerical summary is [results.json](../validation/cpu/results.json). Reproduce it with `python scripts/run_cpu_validation.py`. Python/NumPy versions are recorded in that file; CI regenerates its own report rather than trusting the committed result.

## Fixed defects

- Cross-solver comparison accepted different equation classes and ionic models. Identical constant profiles were incorrectly assigned perfect correlation.
- Self-convergence order used a uniform-ratio formula on nonuniform grids. GCI-style estimates could reuse an older positive order despite later divergence, or mix declared time steps.
- Mesh loading discarded additional tetrahedral blocks; scale-dependent volume thresholds rejected valid small tetrahedra. Higher-order tetrahedra now require explicit linearization.
- Negative root indices and fractional surface indices were mishandled. Geometrically collinear surface triangles were accepted.
- Surface fitting ignored offset bounds and returned an arbitrary speed for unidentifiable equal-distance observations. Unsupported surface parameters/settings and contradictory declared units now fail explicitly.
- Nonfinite repolarization, cellular, and typed result values were accepted along several public paths.
- A backend could return conflicting anatomy provenance, including through nested calibration simulations.
- External wrapper artifact references could point into a deleted temporary directory. Local wrapper outputs now persist and their declared hashes are verified.
- Artifact writes now validate JSON before mutation and replace files atomically; unsafe artifact IDs and remote file-URI authorities are rejected.
- Package, public module, and citation versions were inconsistent; all now declare 0.3.0. Architecture documentation now describes the actual tetrahedral solver.

## Scientific limits

This is numerical and software verification, not complete electrophysiological validation. The volumetric backend is a stationary tetrahedral Eikonal approximation with phenomenological APD; it does not integrate ionic currents or solve monodomain/bidomain equations. The surface backend is isotropic graph propagation. Pseudo-ECG values remain an arbitrary normalized dipole proxy without calibrated torso physics.

No independent PDE solver, measured patient cohort, noise robustness study, restitution benchmark, or clinical outcome study was run. Synthetic inverse recovery uses the same forward model to generate observations and cannot establish real-data identifiability. Spatial APD gradient coefficients are normalized to the prescribed APD range: the magnitude of a lone positive gradient is unidentifiable.

The Niederer fixture was checked against the table in the [pinned upstream FEniCSx-beat demo](https://github.com/finsberg/fenicsx-beat/blob/7ab18453ae57c28d798b744234157f36297113d5/demos/niederer_benchmark.py). This confirms transcription, not an independent PDE rerun. Missing equation/ionic metadata in both comparison profiles still permits legacy comparisons and does not verify equation equivalence. GCI with absent time-step metadata remains conditional; the report explicitly records incomplete time metadata.

## Continuous verification

CI covers Python 3.10–3.14 on Linux, Python 3.12 on Windows, independent SciPy/meshio references, installed wheel behavior, and CPU numerical reproduction. Local verification: **122 tests passed**, **74.10% statement coverage**, and **14 CardiInfer backend integration tests passed**. Source lint and source/wheel builds passed.

The core coverage floor is 70%; it is a regression guard, not a scientific-quality score.

Publication provenance: implementation and CPU evidence committed as `377e150a133a12560940ab4a76d0a5bf1f0efab0`. A clean wheel installation outside the source checkout passed the reference CLI and dependency checks.
