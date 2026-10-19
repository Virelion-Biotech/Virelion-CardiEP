# Camps Cardiac-Digital-Twin external calibration backend

`camps-cdt-external-v1` is a strict CardiEP boundary for the MIT-licensed `juliacamps/Cardiac-Digital-Twin` research pipeline.

The upstream project implements a modular ventricular digital-twinning workflow for ECG/CMR personalization with geometry, conduction/Purkinje, Eikonal/reaction-Eikonal propagation, ECG generation, discrepancy evaluation and sampling. Published studies use it to personalize ventricular activation from standard clinical 12-lead ECG and CMR, and subsequently extend personalization to repolarization.

CardiEP does **not** vendor or silently modify that scientific implementation. A separately installed wrapper translates the stable CardiEP JSON contract into a pinned upstream environment.

## Backend identity

```text
camps-cdt-external-v1
```

Default wrapper command:

```text
cardiep-camps-cdt-wrapper --request {request} --output {output}
```

Override it with `CARDIEP_CAMPS_CDT_WRAPPER`. The command must retain `{request}` and `{output}` placeholders.

## Current upstream reference

At adapter development time, upstream `main` resolved to:

```text
juliacamps/Cardiac-Digital-Twin
816d51fab0837cfe9e20c7d3a318429e9acf0733
```

That is an audit reference, **not** an automatically trusted rolling dependency. Every patient request must explicitly provide `settings.upstream_commit`, and the wrapper result must return the same 40-character commit SHA. A changed upstream revision therefore cannot silently enter a run.

Primary publication DOI required in wrapper provenance:

```text
10.1016/j.media.2024.103361
```

## v1 intended use

The adapter is deliberately narrow:

```text
patient ventricular anatomy
+ standard 12-lead ECG
→ activation/QRS calibration
→ uncertainty-bearing posterior
→ representative activation + ECG prediction
```

It is not a generic gateway to every script in the upstream repository.

For v1, requests require:

- hash-pinned patient anatomy;
- an explicit anatomy coordinate frame;
- standard 12-lead ECG observations for calibration;
- hash-pinned ECG artifacts;
- `calibration_target = "activation_qrs"`;
- `inference_method = "smc_abc"`;
- an explicit pinned `upstream_commit`; and
- standard 12-lead expectation metadata.

Forward replay is unconditional: observations are rejected from `ep.simulate` so held-out measurements cannot accidentally condition a prediction.

## Required calibration result

A wrapper calibration result is rejected unless it includes:

- calibrated parameter values;
- a SHA-256-pinned posterior artifact;
- a representative/best simulation;
- a SHA-256-pinned activation map;
- a SHA-256-pinned standard 12-lead ECG artifact;
- positive ECG sampling-rate metadata;
- an explicit ECG artifact role;
- `posterior_population_size >= 1`;
- `inference_method = "smc_abc"`;
- a named discrepancy metric; and
- exact provenance for upstream repository, upstream commit, wrapper version, paper DOI and CardiEP request hash.

The representative simulation is useful for diagnostics, but the posterior artifact is mandatory because patient-specific inverse problems should not collapse uncertainty into one apparently exact biological parameter vector.

## Wrapper responsibilities

The external wrapper must perform all upstream-specific work, including any conversion from CardiAnatomy geometry into the exact files/classes required by the selected Camps pipeline scripts. It must also make explicit which upstream parameterization and script configuration are used.

A wrapper must not claim compatibility merely because parameter names look similar. HeartTwin performs any later fast-model → detailed-model conversion through its separate versioned EP parameter-mapping contract.

## Scientific boundary

Registration of this backend means only that CardiEP has a fail-closed software contract for a published external research pipeline. It does not establish:

- correct upstream installation;
- anatomical validity;
- equivalence between the wrapper configuration and a published experiment;
- identifiability of inferred parameters;
- validity of translation into openCARP;
- held-out patient accuracy;
- cohort generalization; or
- clinical qualification.

Those must be established independently for the intended use.
