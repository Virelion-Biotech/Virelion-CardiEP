# openCARP external confirmation backend

`opencarp-external-v1` is a strict, confirmation-only CardiEP backend for user-installed openCARP workflows. CardiEP does **not** vendor or redistribute openCARP and does not claim that registering this adapter validates an openCARP experiment for patient use.

## Why a wrapper boundary

openCARP supports direct simulator parameterization and carputils experiments. For a patient workflow, geometry preparation, experiment configuration, lead-field construction, torso/electrode handling, post-processing, and artifact collection are multi-step operations. CardiEP therefore uses a JSON wrapper contract rather than pretending that one generic raw `openCARP` command can safely represent every patient experiment.

Official openCARP documentation recommends carputils for multi-step, reproducible experiments and also supports direct `openCARP +F parameter.par` execution. See:

- https://opencarp.org/getting-started
- https://opencarp.org/documentation/examples
- https://opencarp.org/download/installation

## Availability

The backend is always discoverable so HeartTwin can report that it exists, but it is unavailable unless the configured wrapper executable exists.

By default CardiEP looks for:

```text
cardiep-opencarp-wrapper --request {request} --output {output}
```

A different command can be supplied through `CARDIEP_OPENCARP_WRAPPER` or directly when constructing `OpenCARPExternalBackend`.

The command must contain the literal `{request}` and `{output}` tokens. CardiEP writes a complete `EPSimulationRequest` JSON to the request path. The wrapper must write one `EPSimulationResult` JSON to the output path.

## Request requirements

The current adapter is intentionally limited to patient ECG confirmation. It requires:

```json
{
  "settings": {
    "ecg_forward_model": "lead_field",
    "torso_mode": "patient_specific | standardized",
    "electrode_mode": "recorded | protocol_derived",
    "expected_leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
  }
}
```

If `torso_mode` is `patient_specific`, `settings.torso_ref` must be a SHA-256-pinned artifact reference. If `electrode_mode` is `recorded`, `settings.electrode_ref` must likewise be pinned.

Generic electrode placement and an absent torso are deliberately not accepted by this scientific confirmation adapter.

## Required result artifacts

The wrapper result must contain exactly one:

- `activation_map`; and
- `lead_field_ecg` or `ecg_12lead`.

The ECG artifact metadata must declare all standard 12 lead names, a positive sample rate, and:

```json
{"forward_model": "lead_field"}
```

Any `pseudo_ecg` output is rejected.

## Required provenance

The wrapper result must report:

```json
{
  "provenance": {
    "engine": "openCARP",
    "opencarp_version": "...",
    "wrapper_version": "...",
    "request_sha256": "..."
  }
}
```

`request_sha256` must exactly equal CardiEP's canonical hash of the request JSON. This prevents a wrapper from returning a result generated for a different parameterization while appearing to satisfy the current request.

The surrounding `CardiEPService` additionally preserves the exact anatomy artifact ID, anatomy hash, coordinate frame, and HeartTwin anatomy-bundle fingerprint at the service boundary.

## Calibration policy

`opencarp-external-v1` does not implement `ep.calibrate`. It raises an explicit unavailable error for calibration requests.

The intended first workflow is:

```text
fast activation/inference backend
        |
        +--> parameter/posterior inference
                    |
                    +--> selected supported parameter vector
                                  |
                                  +--> openCARP high-fidelity replay
                                                |
                                                +--> lead-field 12-lead ECG
```

A separate parameter-mapping layer is still required when the fast model and openCARP do not use identical parameter semantics. HeartTwin must not silently copy incompatible parameter names across backends.

## Scientific boundary

A successful wrapper execution demonstrates software interoperability, not patient-model validity. Before patient-level claims, the selected experiment still requires appropriate validation of anatomy, fibres, conductivities, stimulation/activation assumptions, volume-conductor construction, electrode placement, lead-field computation, numerical settings, and held-out ECG endpoints.
