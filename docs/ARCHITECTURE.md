# CardiEP architecture

CardiEP separates the electrophysiology problem into explicit scientific modules so that fast and high-fidelity implementations can share one contract.

## Module graph

```text
ArtifactRef / AnatomyBundle
          │
          ▼
      EPGeometry
          │
          ├──── ventricular coordinates ───► APD map
          │
          ├──── fibres/sheets/normals
          │
          ├──── scar labels
          │
          ├──── electrodes
          │
          └──── roots/endocardium
          │
          ▼
  RootSchedule / Purkinje timing
          │
          ▼
  Propagation model
          │
          ▼
     activation_ms
          │
          ├────► Repolarization model ───► apd_ms / repolarization_ms
          │
          └────► ECG observation model
                           │
                           ▼
                     pseudo ECG
                           │
Measured EPObservation ────┴──► DiscrepancyReport
                                      │
                                      ├── native fast calibration
                                      └── CardiInfer posterior inference
```

## Why this split

The structure is intentionally similar to successful cardiac digital-twinning systems: geometry, conduction, propagation, cellular/repolarization, ECG, discrepancy, and inference are separate concerns. CardiEP keeps inference itself outside the solver core because CardiInfer is the stack-wide uncertainty/inference service.

## Native propagation

`numpy-eikonal-v1` turns each tetrahedral mesh into an undirected edge graph. For an edge displacement **d**, a local fibre-sheet-normal basis is built and the travel time is computed from the orthotropic velocity components. Border-zone and dense-scar labels reduce the local speed through explicit multipliers.

A multi-source Dijkstra solve produces local activation times. Explicit root activation offsets may encode conduction-system delay. When root-to-PMJ path lengths are available, Purkinje delay is added using `purkinje_speed`.

This is a graph-Eikonal approximation rather than a finite-element solution of the continuous Eikonal PDE.

## Repolarization

CardiEP supports:

- constant APD;
- APD min/max bounds;
- linear gradients over any named ventricular coordinate, e.g. `tm`, `ab`, `rt`, `tv`, or Cobiveco fields.

Repolarization is `activation + APD`.

## ECG model

The native ECG layer computes a deterministic nodal-dipole proxy at electrode locations and derives the conventional limb and precordial leads when RA/LA/LL/V1–V6 positions are supplied.

It intentionally reports arbitrary normalized units. Physical torso conductivity, lead fields, inhomogeneous lungs/blood, and electrode transfer physics belong in a dedicated high-fidelity backend.

## Calibration

The native calibration implementation is a bounded coordinate-pattern search. It is deterministic and transparent, making it useful for smoke tests and initialization.

For scientific posterior inference, use CardiInfer and call `ep.simulate` as the forward model.

## Artifact policy

Every native run writes file-backed JSON artifacts with SHA-256 hashes. Large external backends may return their own file-backed artifacts as long as they satisfy the same typed result contract.

No backend is allowed to silently change the subject identity or its own backend identity.
