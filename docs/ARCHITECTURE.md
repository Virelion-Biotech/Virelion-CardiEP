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

`numpy-eikonal-v1` uses tetrahedral Hopf–Lax relaxation. At each vertex, it minimizes interpolated arrival time plus metric travel time over the opposite triangular face. The local metric uses the fibre/sheet/normal speeds; scar multipliers modify cell speeds. This permits wavefront updates through cell interiors rather than only along edges.

Explicit root activation offsets may encode conduction-system delay. When root-to-PMJ path lengths are available, Purkinje delay is added using `purkinje_speed`. The earliest root defines time zero.

This stationary Eikonal approximation has no ionic currents, diffusion time stepping, or monodomain/bidomain PDE solve. `surface-eikonal-v1` separately uses graph shortest paths on triangular surfaces.

## Repolarization

CardiEP supports:

- constant APD;
- APD min/max bounds;
- linear gradients over any named ventricular coordinate, e.g. `tm`, `ab`, `rt`, `tv`, or Cobiveco fields.

Repolarization is `activation + APD`. Coordinate gradients are normalized to the prescribed APD range: a lone positive gradient coefficient therefore controls direction but its magnitude is not identifiable. Synthetic recovery of constant APD does not validate spatial gradients or restitution.

## ECG model

The native ECG layer computes a deterministic nodal-dipole proxy at electrode locations and derives the conventional limb and precordial leads when RA/LA/LL/V1–V6 positions are supplied.

It intentionally reports arbitrary normalized units. Physical torso conductivity, lead fields, inhomogeneous lungs/blood, and electrode transfer physics belong in a dedicated high-fidelity backend.

## Calibration

The native calibration implementation is a bounded coordinate-pattern search. It is deterministic and transparent, making it useful for smoke tests and initialization.

For scientific posterior inference, use CardiInfer and call `ep.simulate` as the forward model.

## Artifact policy

Every native run writes file-backed JSON artifacts with SHA-256 hashes. Large external backends may return their own file-backed artifacts as long as they satisfy the same typed result contract.

No backend is allowed to silently change the subject identity or its own backend identity.
