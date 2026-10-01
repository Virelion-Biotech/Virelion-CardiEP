from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EPObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    values: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    source: Literal["prior", "calibrated", "fixed", "unknown"] = "unknown"


class EPSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    anatomy_ref: ArtifactRef
    backend: str
    parameters: EPParameterSet
    observations: list[EPObservation] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class EPSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    anatomy_ref: ArtifactRef
    observations: list[EPObservation]
    backend: str
    parameter_bounds: dict[str, tuple[float, float]]
    initial_parameters: EPParameterSet | None = None
    inference_backend: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_observations(self) -> EPCalibrationRequest:
        if not self.observations:
            raise ValueError("At least one calibration observation is required")
        for lo, hi in self.parameter_bounds.values():
            if lo >= hi:
                raise ValueError("Parameter bounds must satisfy lower < upper")
        return self


class EPCalibrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
