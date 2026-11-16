# Provenance-aware genotype and disease context v1

CardiEP accepts an optional `EPDiseaseContext` on simulation and calibration requests. It is metadata, not a hidden parameter source. Disease labels, variants and ECG-derived features are never used to silently change EP parameters.

## Supported descriptive contexts

PKP2-associated arrhythmogenic cardiomyopathy; PLN-associated cardiomyopathy; BAG3-associated dilated cardiomyopathy; LAMP2/Danon disease; and CASQ2-associated arrhythmia context. A disease label requires independent phenotype-label provenance. Reported/curated genotype requires source record IDs and source provenance. Unknown remains unknown.

## Explicit calibrated parameter mapping

A `parameter_mapping` is optional. When present it must declare mapping ID, backend, calibration evidence IDs, source ECG bundle hashes, calibration cohort, subject split and explicit parameter values. The mapping backend must equal the requested backend and each mapped value must exactly match the explicit `EPParameterSet` passed to CardiEP. CardiEP does not auto-derive or auto-apply parameters from genotype labels.

## Backend capability boundary

The fast `numpy-eikonal-v1` engine supports conduction, repolarization and scar-tissue-property context only. It does not simulate intracellular calcium cycling, variant-specific ionic currents, desmosomal integrity, proteostasis or autophagic-lysosomal function. Other backends must declare `supported_mechanisms` before such mechanisms can be requested; otherwise CardiEP rejects the request.

CASQ2-associated calcium-release/triggered-activity context remains descriptive unless an appropriate ionic or reduced-order backend is registered. A disease label alone does not establish a modeled mechanism.

## Provenance

Disease context is preserved in simulation/calibration result provenance with a deterministic SHA-256 digest. The context carries source record IDs, ElectroTrace measurement bundle hashes, baseline/comparator definitions, uncertainty references, calibration cohort and subject-level split metadata. A digest establishes record identity, not scientific truth.

This contract is research metadata, not a diagnostic or clinical prediction model.
