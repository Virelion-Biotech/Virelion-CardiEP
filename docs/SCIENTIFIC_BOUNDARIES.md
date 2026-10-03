# Scientific boundaries

## What the native engine establishes

The native engine establishes that the Virelion stack can carry a patient-specific EP model end to end:

- anatomy;
- conduction roots;
- fibre anisotropy;
- scar modifiers;
- activation;
- APD/repolarization;
- electrode geometry;
- simulated ECG;
- measured EP observations;
- discrepancy;
- calibration;
- provenance.

Its deterministic reference tests protect those software contracts.

## What it does not establish

It does not establish:

- equivalence to a continuous Eikonal PDE implementation;
- equivalence to monodomain or bidomain electrophysiology;
- validated ionic-current dynamics;
- torso-conductivity or electrode-transfer fidelity;
- patient-specific identifiability;
- population generalization;
- diagnostic accuracy;
- ablation-planning performance;
- clinical safety or effectiveness.

## Escalation path

Use the native backend for fast screening and integration. Escalate important conclusions in this order:

1. numerical comparison against a second Eikonal implementation;
2. monodomain comparison on the same anatomy;
3. mesh/time-step/conductivity convergence;
4. calibration on held-out synthetic truth;
5. retrospective empirical validation on measured ECG/EAM;
6. external-center validation;
7. prospective clinical validation if the intended use requires it.

## Inference boundary

The native `ep.calibrate` optimizer is deterministic. It produces a fitted parameter set, not a posterior distribution.

CardiInfer owns posterior sampling, uncertainty propagation, identifiability, and sensitivity-analysis contracts.
