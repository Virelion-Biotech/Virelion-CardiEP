# PhysioNet/CinC 2007 Case 3 patient pilot

This directory contains a reproducible, research-only patient-specific EP pilot for the public PhysioNet/CinC Challenge 2007 Case 3.

## Frozen evidence policy

The ECG split was fixed before model fitting:

- calibration: I, II, V1, V2, V4
- primary spatial holdout: V5, V6
- secondary consistency checks: III, aVR, aVL, aVF, V3

This is a **same-acquisition spatial lead holdout**, not an independent repeat ECG or independent visit. No primary holdout waveform may be used for parameter selection, root selection, meshing choices, solver settings, stopping rules, or profile tuning.

## Initial model boundary

The first fast inverse model is intentionally restricted to isotropic ventricular propagation because the Case 3 challenge geometry does not provide a measured myocardial fibre field. The fitted scalar will be `isotropic_speed`; activation roots are selected by a fixed anatomy-derived endocardial heuristic rather than fitted to the held-out leads. The native pseudo-ECG is used only for the fast inverse loop.

High-fidelity confirmation uses the registered CardiEP experiment `virelion-patient-qrs-recorded-lf12` v1.0.0: patient-specific torso, recorded electrode geometry, prescribed calibrated activation, openCARP Vm generation, and lead-field ECG projection. That profile does not independently re-solve propagation.

## Data provenance

Source: PhysioNet/Computing in Cardiology Challenge 2007, Case 3. The source site documents the detailed heart/lung/torso geometry as prepared by A. van Oosterom and P. van Dam in a consistent millimetre coordinate system. It also documents 120 measured body-surface electrodes plus limb potentials and a single averaged PQRST complex. Source files are downloaded at runtime and SHA-256 pinned; raw patient files are not committed to this repository.

The preflight must resolve the documented sampling-interval discrepancy from the actual source file before calibration is enabled. The reusable preflight workflow is installed on CardiEP `main`, so subsequent synchronization commits on this campaign branch execute the audit against the public source rather than relying on local copies.
