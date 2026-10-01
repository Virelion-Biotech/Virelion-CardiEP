# CardiEP architecture

CardiEP separates scientific contracts from numerical implementations.

```text
CardiAnatomy geometry/fibers/scar
        + ECG / EAM observations
                  |
                  v
            EPSimulationRequest
                  |
          registered EP backend
                  |
      activation/repolarization/ECG
                  |
                  +----> CardiInfer calibration / UQ
```

Planned backend families include the Virelion reference EP model, wrappers around published Cardiac-Digital-Twin workflows, Eikonal/monodomain/bidomain engines, and solver-specific exporters. Solver licensing and validation status must remain explicit.

## Ownership

CardiEP owns electrophysiology model execution and EP-specific contracts. It does not own anatomical reconstruction, general uncertainty inference, mechanics, hemodynamics, or intervention selection.

CardiAnatomy must establish geometry and spatial frames before patient-specific EP execution. CardiInfer should own posterior inference and uncertainty quantification when calibration goes beyond a backend-specific deterministic fit.

## Validation ladder

1. Contract and software behavior.
2. Numerical convergence/equivalence.
3. Synthetic parameter-recovery tests.
4. Held-out empirical observation agreement.
5. External patient/cohort validation.

A backend must not promote itself to a higher validation level merely because it satisfies the CardiEP API.
