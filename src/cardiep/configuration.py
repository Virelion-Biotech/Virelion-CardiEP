from __future__ import annotations

import math
from typing import Any

from .geometry import EPGeometry

_SPEED_GROUPS = {
    "fibre_speed": (
        "fibre_speed",
        "fibre_speed_cm_per_ms",
        "fiber_speed",
        "fiber_speed_cm_per_ms",
    ),
    "sheet_speed": ("sheet_speed", "sheet_speed_cm_per_ms"),
    "normal_speed": ("normal_speed", "normal_speed_cm_per_ms"),
    "isotropic_speed": ("isotropic_speed", "isotropic_speed_cm_per_ms"),
    "transverse_speed": ("transverse_speed", "transverse_speed_cm_per_ms"),
    "purkinje_speed": ("purkinje_speed", "purkinje_speed_cm_per_ms"),
}

_BASE_PARAMETERS = {alias for aliases in _SPEED_GROUPS.values() for alias in aliases} | {
    "border_zone_speed_scale",
    "scar_speed_scale",
    "apd_ms",
    "apd_min_ms",
    "apd_min",
    "apd_max_ms",
    "apd_max",
}

_NATIVE_SETTINGS = {
    "geometry_unit",
    "root_nodes",
    "auto_root_count",
    "root_activation_ms",
    "purkinje_root_distance_cm",
    "with_ecg",
    "ecg_sample_rate_hz",
    "ecg_pre_activation_ms",
    "duration_ms",
    "qrs_sigma_ms",
    "t_sigma_ms",
    "repolarization_scale",
    "ecg_chunk_size",
    "output_dir",
    "likelihood_hints",
    "step_fraction",
    "max_iterations",
    "step_tolerance",
}


def _provided(parameters: dict[str, float], group: str) -> list[str]:
    return [name for name in _SPEED_GROUPS[group] if name in parameters]


def validate_native_configuration(
    geometry: EPGeometry,
    settings: dict[str, Any],
    parameters: dict[str, float],
) -> None:
    """Fail closed on native-engine configuration that would be ignored or ambiguous."""
    dynamic_apd = {f"apd_gradient_{name}" for name in geometry.ventricular_coordinates}
    unknown_parameters = sorted(set(parameters) - _BASE_PARAMETERS - dynamic_apd)
    if unknown_parameters:
        raise ValueError("Unknown numpy-eikonal-v1 parameter(s): " + ", ".join(unknown_parameters))

    unknown_settings = sorted(set(settings) - _NATIVE_SETTINGS)
    if unknown_settings:
        raise ValueError("Unknown numpy-eikonal-v1 setting(s): " + ", ".join(unknown_settings))

    for group, aliases in _SPEED_GROUPS.items():
        supplied = [name for name in aliases if name in parameters]
        if len(supplied) > 1:
            raise ValueError(f"Provide only one alias for {group}: {', '.join(supplied)}")

    if "apd_ms" in parameters:
        conflicting = sorted(
            name
            for name in parameters
            if name in {"apd_min_ms", "apd_min", "apd_max_ms", "apd_max"}
            or name.startswith("apd_gradient_")
        )
        if conflicting:
            raise ValueError(
                "apd_ms cannot be combined with APD bounds/gradients: " + ", ".join(conflicting)
            )
    if "apd_min" in parameters and "apd_min_ms" in parameters:
        raise ValueError("Provide only one of apd_min or apd_min_ms")
    if "apd_max" in parameters and "apd_max_ms" in parameters:
        raise ValueError("Provide only one of apd_max or apd_max_ms")

    if geometry.fibre is None:
        unused = sorted(
            name
            for group in ("fibre_speed", "sheet_speed", "normal_speed", "transverse_speed")
            for name in _provided(parameters, group)
        )
        if unused:
            raise ValueError(
                "Anisotropic conduction parameters have no effect without a fibre field: "
                + ", ".join(unused)
            )
    else:
        isotropic = _provided(parameters, "isotropic_speed")
        if isotropic:
            raise ValueError("isotropic_speed has no effect when a fibre field is present")
        transverse = _provided(parameters, "transverse_speed")
        if transverse and (geometry.sheet is not None or geometry.normal is not None):
            raise ValueError(
                "transverse_speed is only used when fibre is present without sheet/normal fields"
            )
        if transverse:
            redundant = sorted(
                _provided(parameters, "sheet_speed") + _provided(parameters, "normal_speed")
            )
            if redundant:
                raise ValueError(
                    "sheet/normal speeds are unused when transverse_speed is explicit "
                    "for fibre-only geometry: " + ", ".join(redundant)
                )

    scar_parameters = [
        name for name in ("border_zone_speed_scale", "scar_speed_scale") if name in parameters
    ]
    if scar_parameters and geometry.scar_labels is None:
        raise ValueError(
            "Scar conduction modifiers have no effect without scar_labels: "
            + ", ".join(scar_parameters)
        )

    purkinje_parameters = _provided(parameters, "purkinje_speed")
    if purkinje_parameters and settings.get("purkinje_root_distance_cm") is None:
        raise ValueError("purkinje_speed has no effect without purkinje_root_distance_cm")

    for name, value in parameters.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"Native EP parameter {name!r} must be finite")

    if "auto_root_count" in settings and settings["auto_root_count"] is not None:
        value = settings["auto_root_count"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("auto_root_count must be a positive integer")

    if "ecg_chunk_size" in settings:
        value = settings["ecg_chunk_size"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("ecg_chunk_size must be a positive integer")

    if "with_ecg" in settings and not isinstance(settings["with_ecg"], bool):
        raise TypeError("with_ecg must be boolean")

    finite_positive = (
        "ecg_sample_rate_hz",
        "duration_ms",
        "qrs_sigma_ms",
        "t_sigma_ms",
        "step_fraction",
        "step_tolerance",
    )
    for name in finite_positive:
        if name in settings and settings[name] is not None:
            value = float(settings[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    if "repolarization_scale" in settings:
        value = float(settings["repolarization_scale"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("repolarization_scale must be non-negative and finite")

    if "ecg_pre_activation_ms" in settings:
        value = float(settings["ecg_pre_activation_ms"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("ecg_pre_activation_ms must be non-negative and finite")

    if "max_iterations" in settings:
        value = settings["max_iterations"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_iterations must be a positive integer")


def validate_parameter_units(parameter_set):
    """Native values use fixed canonical units; reject contradictory declarations."""
    for name, unit in parameter_set.units.items():
        if name not in parameter_set.values:
            raise ValueError(f"Units declared for absent parameter {name!r}")
        if "speed" in name:
            expected = "cm/ms"
        elif (
            name.startswith("apd") and not name.startswith("apd_gradient_")
        ) or name == "activation_offset_ms":
            expected = "ms"
        else:
            expected = "dimensionless"
        normalized = str(unit).strip().lower()
        accepted = {"dimensionless", "1", "unitless"} if expected == "dimensionless" else {expected}
        if normalized not in accepted:
            raise ValueError(f"Parameter {name!r} must use {expected} units; got {unit!r}")
