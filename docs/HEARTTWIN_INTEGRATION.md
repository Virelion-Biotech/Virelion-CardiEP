# HeartTwin integration

Initial capabilities:

- `ep.health`
- `ep.simulate`
- `ep.calibrate`

HeartTwin should pass a ready CardiAnatomy artifact reference plus measured EP observations. CardiEP must never infer subject identity, coordinate registration, or measurement provenance from filenames.

Calibration uncertainty belongs in CardiInfer and should be linked back as a posterior artifact rather than collapsed into one unexplained parameter vector.

## Proposed HeartTwin registry entry

```yaml
- name: CardiEP
  repository: Virelion-Biotech/Virelion-CardiEP
  capabilities: [ep.health, ep.simulate, ep.calibrate]
  builtin: cardiep
  endpoint: ${CARDIEP_URL}
```

HeartTwin should store `EPSimulationResult` and `EPCalibrationResult` as typed artifacts and preserve the backend name, parameter provenance, validation status, and artifact digests.
