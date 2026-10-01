# Virelion-CardiEP

Patient-specific cardiac electrophysiology and personalization layer for the Virelion HeartTwin stack.

CardiEP owns stable contracts for conduction/activation/repolarization simulation, ECG/EAM observations, scar-aware electrophysiology parameters, forward solver backends, and calibration handoff to CardiInfer. Heavy numerical engines remain swappable backends.

## Scope

- consume CardiAnatomy meshes, fibers, scar maps, and spatial registrations;
- represent ECG, EAM and activation-map observations explicitly;
- run backend-neutral electrophysiology forward models;
- expose calibration problems without hiding uncertainty;
- retain solver/output provenance and validation state;
- provide HeartTwin-native `ep.*` capabilities.

The initial package intentionally contains no fake physiological solver. A run requires an explicitly registered backend.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
cardiep doctor
```

## Scientific boundary

Contract validation and software tests do not establish numerical equivalence, physiological validity, patient personalization, or clinical performance.

## License

AGPL-3.0-or-later.
