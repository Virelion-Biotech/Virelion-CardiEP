from __future__ import annotations

import math
import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_unique_observations(observations: list[EPObservation]) -> None:
    ids = [item.observation_id for item in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("EP observation IDs must be unique")


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    coordinate_frame: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if any(character not in string.hexdigits for character in lowered):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return lowered


class EPObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    observation_id: str
    kind: Literal[
        "ecg",
        "eam_activation",
        "activation_map",
        "repolarization_map",
        "other",
    ]
    artifact: ArtifactRef
    coordinate_frame: str | None = None
    units: str | None = None
    acquired_at: str | None = None


class EPParameterSet(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    values: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    source: Literal["prior", "calibrated", "fixed", "unknown"] = "unknown"

    @model_validator(mode="after")
    def require_finite_values(self) -> EPParameterSet:
        for name, value in self.values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"EP parameter {name!r} must be finite")
        return self


class EPDiseaseContext(BaseModel):
    """Provenance-linked disease/variant context kept separate from EP parameters."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    disease_label: Literal["PKP2_ACM","PLN_cardiomyopathy","BAG3_DCM","LAMP2_Danon","CASQ2_arrhythmia","other","unknown"] = "unknown"
    genotype_status: Literal["unknown","reported","curated"] = "unknown"
    variants: list[dict[str, Any]] = Field(default_factory=list)
    source_record_ids: list[str] = Field(default_factory=list)
    source_provenance: list[dict[str, Any]] = Field(default_factory=list)
    phenotype_label: str | None = None
    phenotype_label_source_record_ids: list[str] = Field(default_factory=list)
    baseline_definition: dict[str, Any] = Field(default_factory=dict)
    comparator_definition: dict[str, Any] = Field(default_factory=dict)
    uncertainty_reference: dict[str, Any] = Field(default_factory=dict)
    calibration_cohort: dict[str, Any] = Field(default_factory=dict)
    subject_split: dict[str, Any] = Field(default_factory=dict)
    electrotrace_bundle_sha256: list[str] = Field(default_factory=list)
    requested_mechanisms: list[Literal["conduction","repolarization","scar_tissue_properties","desmosomal_integrity","intracellular_calcium_cycling","variant_specific_ionic_currents","proteostasis","autophagic_lysosomal_function"]] = Field(default_factory=list)
    parameter_mapping: dict[str, Any] | None = None
    inferred_from_ecg: bool = False

    @model_validator(mode="after")
    def validate_disease_context(self):
        if self.inferred_from_ecg:
            raise ValueError("genotype and disease labels must never be inferred from ECG")
        if self.genotype_status in {"reported","curated"}:
            if not self.source_record_ids or not self.source_provenance:
                raise ValueError("reported/curated genotype requires source record IDs and provenance")
        if self.genotype_status=="unknown" and self.variants:
            raise ValueError("unknown genotype cannot contain asserted variant records")
        for i, source in enumerate(self.source_provenance):
            for key in ("source_identifier","source_type","source_url","source_sha256"):
                if not isinstance(source.get(key),str) or not source[key].strip():
                    raise ValueError(f"source_provenance[{i}].{key} is required")
            digest=source["source_sha256"]
            if len(digest)!=64 or any(ch not in string.hexdigits for ch in digest):
                raise ValueError(f"source_provenance[{i}].source_sha256 must be SHA-256")
        for digest in self.electrotrace_bundle_sha256:
            if len(digest)!=64 or any(ch not in string.hexdigits for ch in digest):
                raise ValueError("electrotrace_bundle_sha256 entries must be SHA-256")
        if self.parameter_mapping is not None:
            required=("mapping_id","backend","calibration_evidence_ids","source_bundle_sha256","calibration_cohort","subject_split","parameters")
            missing=[k for k in required if k not in self.parameter_mapping]
            if missing: raise ValueError("parameter_mapping missing: "+", ".join(missing))
            if not self.parameter_mapping["calibration_evidence_ids"]:
                raise ValueError("parameter_mapping requires calibration evidence")
            if not self.parameter_mapping["parameters"]:
                raise ValueError("parameter_mapping requires explicit parameter values")
            for digest in self.parameter_mapping["source_bundle_sha256"]:
                if len(digest)!=64 or any(ch not in string.hexdigits for ch in digest):
                    raise ValueError("parameter_mapping source_bundle_sha256 must contain SHA-256 values")
        return self


class EPSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    subject_id: str
    anatomy_ref: ArtifactRef
    backend: str
    parameters: EPParameterSet
    observations: list[EPObservation] = Field(default_factory=list)
    disease_context: EPDiseaseContext | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_observations(self) -> EPSimulationRequest:
        _require_unique_observations(self.observations)
        if self.disease_context is not None:
            mapping=self.disease_context.parameter_mapping
            if mapping is not None:
                if mapping["backend"] != self.backend:
                    raise ValueError("calibrated disease parameter mapping backend mismatch")
                for name,value in mapping["parameters"].items():
                    if self.parameters.values.get(name) != value:
                        raise ValueError(f"EP parameter {name!r} differs from explicit calibrated disease mapping")
            if self.backend=="numpy-eikonal-v1":
                unsupported={"intracellular_calcium_cycling","variant_specific_ionic_currents","desmosomal_integrity","proteostasis","autophagic_lysosomal_function"}
                requested=unsupported.intersection(self.disease_context.requested_mechanisms)
                if requested:
                    raise ValueError("numpy-eikonal-v1 does not support disease mechanisms: "+", ".join(sorted(requested)))
        return self


class EPSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract_version: str = "1.0"
    subject_id: str
    backend: str
    parameters: EPParameterSet
    outputs: list[ArtifactRef] = Field(default_factory=list)
    validation_status: Literal[
        "unvalidated",
        "software_checked",
        "numerically_checked",
        "empirically_checked",
    ] = "unvalidated"
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class EPCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    subject_id: str
    anatomy_ref: ArtifactRef
    observations: list[EPObservation]
    backend: str
    disease_context: EPDiseaseContext | None = None
    parameter_bounds: dict[str, tuple[float, float]]
    initial_parameters: EPParameterSet | None = None
    inference_backend: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_problem_definition(self) -> EPCalibrationRequest:
        if not self.observations:
            raise ValueError("At least one calibration observation is required")
        if self.disease_context is not None and self.disease_context.parameter_mapping is not None and self.disease_context.parameter_mapping["backend"] != self.backend:
            raise ValueError("calibrated disease parameter mapping backend mismatch")
        _require_unique_observations(self.observations)
        for name, (lo, hi) in self.parameter_bounds.items():
            if not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
                raise ValueError(f"Parameter bounds for {name!r} must be finite")
            if lo >= hi:
                raise ValueError("Parameter bounds must satisfy lower < upper")
        return self


class EPCalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract_version: str = "1.0"
    subject_id: str
    backend: str
    parameters: EPParameterSet
    objective: float | None = None
    converged: bool | None = None
    posterior_ref: ArtifactRef | None = None
    simulated: EPSimulationResult | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
