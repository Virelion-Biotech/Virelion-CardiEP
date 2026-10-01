# HeartTwin integration

CardiEP is a first-class native HeartTwin service.

## Capabilities

- `ep.health`
- `ep.backends`
- `ep.simulate`
- `ep.calibrate`
- `ep.validate.reference`
- `ep.ecosystem`

HeartTwin passes a ready CardiAnatomy artifact reference plus explicit EP parameters and, when relevant, measured ElectroTrace/EAM observations. CardiEP never infers subject identity, coordinate registration, units, or measurement provenance from filenames.

## Registry entry

```yaml
- name: CardiEP
  repository: Virelion-Biotech/Virelion-CardiEP
  capabilities:
    [ep.health, ep.backends, ep.simulate, ep.calibrate, ep.validate.reference, ep.ecosystem]
  builtin: cardiep
  endpoint: ${CARDIEP_URL}
```

When the Python package is installed, HeartTwin invokes `EPAPI` in-process. The same capability names may be served over HTTP by a remote CardiEP deployment.

## Data path

```text
CardiAnatomy AnatomyBundle
          │
          ▼
     CardiEP ep.simulate ◄──── parameter proposal
          │                         ▲
          │                         │
          ├── activation            │
          ├── repolarization        │
          └── ECG                   │
          │                         │
          ▼                         │
ElectroTrace / EAM observations ─► CardiInfer
          │
          └──────── likelihood / discrepancy
```

The native `numpy-eikonal-v1` backend means HeartTwin can now execute an EP forward model without an external numerical engine. Publication- or clinical-grade workflows should still select and validate a high-fidelity backend appropriate to the intended claim.

## Calibration boundary

`ep.calibrate` is a deterministic fast optimizer. It is useful for software verification, initialization, and rapid fits.

Posterior uncertainty, identifiability and uncertainty propagation remain CardiInfer responsibilities. The same measured artifact is carried from ElectroTrace into CardiInfer likelihood terms rather than being converted into an unexplained scalar feature.

## State and provenance

HeartTwin should preserve:

- CardiEP backend name and version;
- parameter values, units and source;
- anatomy and observation artifact IDs/hashes;
- output artifact IDs/hashes;
- validation status;
- warnings and numerical diagnostics;
- inference/posterior artifacts when CardiInfer is used.

The HeartTwin integration test exercises `ep.backends`, `ep.validate.reference`, and a complete `ep.simulate` call through the registry.
