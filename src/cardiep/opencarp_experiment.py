"""Versioned scientific contract for patient-specific openCARP confirmation.

The external runtime can stage immutable files, but a scientifically meaningful
openCARP run also needs explicit modeling choices. This module makes those
choices part of the request rather than relying on hidden driver defaults.

The contract is deliberately solver-facing rather than a claim of universal
best practice. A study must choose and justify its own tissue regions, ionic
models, conductivities / conduction velocities, numerical controls, calibrated
activation translation, torso volume-conductor properties, and ECG output grid.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import EPSimulationRequest
from .provenance import sha256_json

_STANDARD_12 = {
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
}
_JSON_SCALAR = str | int | float | bool


class OpenCARPTimeGrid(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start_ms: float
    end_ms: float
    integration_dt_ms: float = Field(gt=0)
    output_dt_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_grid(self) -> OpenCARPTimeGrid:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.output_dt_ms < self.integration_dt_ms:
            raise ValueError("output_dt_ms cannot be smaller than integration_dt_ms")
        ratio = self.output_dt_ms / self.integration_dt_ms
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("output_dt_ms must be an integer multiple of integration_dt_ms")
        return self


class OpenCARPTissueRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    region_id: str
    element_tags: list[int]
    ionic_model: str
    ionic_parameters: dict[str, float] = Field(default_factory=dict)
    conduction_velocity_m_per_s: tuple[float, float, float] | None = None
    intracellular_conductivity_s_per_m: tuple[float, float, float] | None = None
    extracellular_conductivity_s_per_m: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def validate_region(self) -> OpenCARPTissueRegion:
        if not self.region_id.strip() or not self.ionic_model.strip():
            raise ValueError("region_id and ionic_model must be non-empty")
        if not self.element_tags or len(self.element_tags) != len(set(self.element_tags)):
            raise ValueError("element_tags must be non-empty and unique per region")
        for label, vector in (
            ("conduction_velocity_m_per_s", self.conduction_velocity_m_per_s),
            ("intracellular_conductivity_s_per_m", self.intracellular_conductivity_s_per_m),
            ("extracellular_conductivity_s_per_m", self.extracellular_conductivity_s_per_m),
        ):
            if vector is not None and any(
                (not math.isfinite(value) or value <= 0) for value in vector
            ):
                raise ValueError(f"{label} values must be positive and finite")
        return self


class OpenCARPParameterBinding(BaseModel):
    """Map one HeartTwin/CardiEP parameter onto one explicit experiment target."""

    model_config = ConfigDict(extra="forbid")

    parameter_name: str
    parameter_unit: str
    target_path: str
    rationale: str = Field(min_length=8)

    @model_validator(mode="after")
    def validate_binding(self) -> OpenCARPParameterBinding:
        if (
            not self.parameter_name.strip()
            or not self.parameter_unit.strip()
            or not self.target_path.strip()
        ):
            raise ValueError("parameter binding name, unit and target_path must be explicit")
        return self


class OpenCARPActivationTranslation(BaseModel):
    """Version the transformation from calibrated activation to openCARP stimulation."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    strategy_version: str
    settings: dict[str, _JSON_SCALAR]
    rationale: str = Field(min_length=8)

    @model_validator(mode="after")
    def validate_translation(self) -> OpenCARPActivationTranslation:
        if not self.strategy_id.strip() or not self.strategy_version.strip():
            raise ValueError("activation translation strategy_id/version must be explicit")
        if not self.settings:
            raise ValueError("activation translation settings must be explicit and non-empty")
        return self


class OpenCARPVolumeConductorRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    region_id: str
    element_tags: list[int]
    conductivity_s_per_m: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_region(self) -> OpenCARPVolumeConductorRegion:
        if not self.region_id.strip():
            raise ValueError("volume-conductor region_id must be non-empty")
        if not self.element_tags or len(self.element_tags) != len(set(self.element_tags)):
            raise ValueError("volume-conductor element_tags must be non-empty and unique")
        return self


class OpenCARPECGSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    forward_model: Literal["lead_field"]
    torso_mode: Literal["patient_specific", "standardized"]
    electrode_mode: Literal["recorded", "protocol_derived"]
    expected_leads: list[str]
    output_voltage_unit: Literal["V", "mV", "uV"]
    output_sample_rate_hz: float = Field(gt=0)
    time_axis_start_ms: float
    time_axis_end_ms: float
    simulation_zero_relative_to_ecg_reference_ms: float
    volume_conductor_regions: list[OpenCARPVolumeConductorRegion]

    @model_validator(mode="after")
    def validate_ecg(self) -> OpenCARPECGSpec:
        if set(self.expected_leads) != _STANDARD_12 or len(self.expected_leads) != len(
            _STANDARD_12
        ):
            raise ValueError(
                "openCARP patient confirmation currently requires exactly standard 12 leads"
            )
        if self.time_axis_end_ms <= self.time_axis_start_ms:
            raise ValueError("ECG time axis end must be greater than start")
        if not self.volume_conductor_regions:
            raise ValueError("lead-field ECG requires explicit volume-conductor conductivities")
        all_tags = [
            tag for region in self.volume_conductor_regions for tag in region.element_tags
        ]
        if len(all_tags) != len(set(all_tags)):
            raise ValueError("volume-conductor element tags cannot overlap across regions")
        return self


class OpenCARPExperimentSpec(BaseModel):
    """Complete declared experiment semantics consumed by the package runtime."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract_version: Literal["1.0"] = "1.0"
    experiment_id: str
    experiment_version: str
    intended_use: str = Field(min_length=12)
    field_model: Literal["monodomain", "bidomain", "pseudo_bidomain"]
    activation_model: Literal[
        "reaction_diffusion",
        "reaction_eikonal_minus",
        "reaction_eikonal_plus",
    ]
    tissue_regions: list[OpenCARPTissueRegion]
    fibre_field_required: bool
    parameter_bindings: list[OpenCARPParameterBinding]
    activation_translation: OpenCARPActivationTranslation
    time_grid: OpenCARPTimeGrid
    ecg: OpenCARPECGSpec
    solver_options: dict[str, _JSON_SCALAR]
    provenance_reference: str

    @model_validator(mode="after")
    def validate_experiment(self) -> OpenCARPExperimentSpec:
        if not self.experiment_id.strip() or not self.experiment_version.strip():
            raise ValueError("experiment_id/version must be explicit")
        if not self.tissue_regions:
            raise ValueError("at least one tissue region is required")
        if not self.parameter_bindings:
            raise ValueError("at least one explicit parameter binding is required")
        if not self.solver_options:
            raise ValueError("solver_options must be explicit and non-empty")
        if not self.provenance_reference.strip():
            raise ValueError(
                "provenance_reference must identify the experiment rationale/source"
            )

        region_ids = [region.region_id for region in self.tissue_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("tissue region IDs must be unique")
        tissue_tags = [tag for region in self.tissue_regions for tag in region.element_tags]
        if len(tissue_tags) != len(set(tissue_tags)):
            raise ValueError("tissue element tags cannot overlap across regions")

        bindings = [binding.parameter_name for binding in self.parameter_bindings]
        if len(bindings) != len(set(bindings)):
            raise ValueError("each source parameter may be bound at most once")
        targets = [binding.target_path for binding in self.parameter_bindings]
        if len(targets) != len(set(targets)):
            raise ValueError("each experiment target_path may be bound at most once")

        for region in self.tissue_regions:
            if self.activation_model == "reaction_diffusion":
                if region.intracellular_conductivity_s_per_m is None:
                    raise ValueError(
                        "reaction_diffusion requires intracellular conductivities"
                    )
            elif self.activation_model == "reaction_eikonal_minus":
                if region.conduction_velocity_m_per_s is None:
                    raise ValueError(
                        "reaction_eikonal_minus requires conduction velocities"
                    )
            elif self.activation_model == "reaction_eikonal_plus":
                if (
                    region.conduction_velocity_m_per_s is None
                    or region.intracellular_conductivity_s_per_m is None
                ):
                    raise ValueError(
                        "reaction_eikonal_plus requires both velocities and intracellular conductivities"
                    )

            if (
                self.field_model in {"bidomain", "pseudo_bidomain"}
                and region.extracellular_conductivity_s_per_m is None
            ):
                raise ValueError(
                    f"{self.field_model} requires extracellular conductivity for every tissue region"
                )
        return self

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def validate_experiment_request(
    request: EPSimulationRequest,
    spec: OpenCARPExperimentSpec | dict[str, Any],
) -> OpenCARPExperimentSpec:
    """Bind an experiment spec to one immutable CardiEP confirmation request."""
    selected = (
        spec
        if isinstance(spec, OpenCARPExperimentSpec)
        else OpenCARPExperimentSpec.model_validate(spec)
    )
    settings = request.settings
    if settings.get("ecg_forward_model") != selected.ecg.forward_model:
        raise ValueError("experiment ECG forward model disagrees with request")
    if settings.get("torso_mode") != selected.ecg.torso_mode:
        raise ValueError("experiment torso_mode disagrees with request")
    if settings.get("electrode_mode") != selected.ecg.electrode_mode:
        raise ValueError("experiment electrode_mode disagrees with request")
    if set(map(str, settings.get("expected_leads") or [])) != set(
        selected.ecg.expected_leads
    ):
        raise ValueError("experiment expected leads disagree with request")

    incoming = set(request.parameters.values)
    bound = {binding.parameter_name for binding in selected.parameter_bindings}
    if incoming != bound:
        missing = sorted(incoming - bound)
        extra = sorted(bound - incoming)
        details = []
        if missing:
            details.append("unbound request parameters: " + ", ".join(missing))
        if extra:
            details.append("bindings for absent parameters: " + ", ".join(extra))
        raise ValueError("; ".join(details))
    for binding in selected.parameter_bindings:
        actual_unit = request.parameters.units.get(binding.parameter_name)
        if actual_unit != binding.parameter_unit:
            raise ValueError(
                f"parameter {binding.parameter_name!r} unit {actual_unit!r} does not match "
                f"experiment binding {binding.parameter_unit!r}"
            )

    calibration = settings.get("hearttwin_calibration_reference")
    if not isinstance(calibration, dict) or not isinstance(
        calibration.get("activation_ref"), dict
    ):
        raise ValueError(
            "experiment requires a hash-pinned HeartTwin calibration activation reference"
        )
    activation_ref = calibration["activation_ref"]
    if not activation_ref.get("sha256") or len(str(activation_ref["sha256"])) != 64:
        raise ValueError("calibration activation reference must be SHA-256 pinned")
    return selected
