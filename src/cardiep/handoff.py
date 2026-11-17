"""Adapters from measurement-service handoffs into CardiEP contracts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import (
    ArtifactRef,
    EPCalibrationRequest,
    EPDiseaseContext,
    EPObservation,
    EPParameterSet,
)


def _validate_handoff_subject(handoff: Mapping[str, Any], subject_id: str) -> None:
    entity_id = handoff.get("entity_id")
    if entity_id is not None and str(entity_id) != str(subject_id):
        raise ValueError(
            f"ElectroTrace handoff entity_id {entity_id!r} does not match "
            f"requested subject_id {subject_id!r}"
        )


def observations_from_electrotrace(handoff: Mapping[str, Any]) -> list[EPObservation]:
    """Validate ElectroTrace calibration observations without weakening CardiEP types."""
    raw = handoff.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("ElectroTrace handoff must contain a non-empty observations array")
    observations = [EPObservation.model_validate(item) for item in raw]
    ids = [item.observation_id for item in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("ElectroTrace handoff contains duplicate observation IDs")
    artifact_ids = [item.artifact.artifact_id for item in observations]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("ElectroTrace handoff contains duplicate artifact IDs")
    return observations


def calibration_request_from_electrotrace(
    handoff: Mapping[str, Any],
    *,
    subject_id: str,
    anatomy_ref: ArtifactRef | Mapping[str, Any],
    backend: str,
    parameter_bounds: Mapping[str, tuple[float, float]],
    initial_parameters: EPParameterSet | Mapping[str, Any] | None = None,
    inference_backend: str | None = None,
    settings: Mapping[str, Any] | None = None,
    disease_context: EPDiseaseContext | Mapping[str, Any] | None = None,
) -> EPCalibrationRequest:
    """Create a typed CardiEP calibration request from an ElectroTrace handoff."""
    _validate_handoff_subject(handoff, subject_id)
    anatomy = (
        anatomy_ref
        if isinstance(anatomy_ref, ArtifactRef)
        else ArtifactRef.model_validate(anatomy_ref)
    )
    initial = None
    if initial_parameters is not None:
        initial = (
            initial_parameters
            if isinstance(initial_parameters, EPParameterSet)
            else EPParameterSet.model_validate(initial_parameters)
        )
    observations = observations_from_electrotrace(handoff)
    context = None
    if disease_context is not None:
        context = disease_context if isinstance(disease_context, EPDiseaseContext) else EPDiseaseContext.model_validate(disease_context)
        handoff_hashes = sorted({item.artifact.sha256 for item in observations if item.artifact.sha256})
        if context.electrotrace_bundle_sha256 and not set(handoff_hashes).issubset(set(context.electrotrace_bundle_sha256)):
            raise ValueError("disease context ElectroTrace hashes do not cover the supplied handoff artifacts")
        if handoff_hashes:
            context = context.model_copy(update={"electrotrace_bundle_sha256": sorted(set(context.electrotrace_bundle_sha256) | set(handoff_hashes))})
    merged_settings = dict(settings or {})
    if handoff.get("schema_version"):
        merged_settings.setdefault(
            "measurement_handoff_schema", str(handoff["schema_version"])
        )
    if handoff.get("likelihood_hints"):
        merged_settings.setdefault("likelihood_hints", list(handoff["likelihood_hints"]))
    return EPCalibrationRequest(
        subject_id=subject_id,
        anatomy_ref=anatomy,
        observations=observations,
        backend=backend,
        disease_context=context,
        parameter_bounds=dict(parameter_bounds),
        initial_parameters=initial,
        inference_backend=inference_backend,
        settings=merged_settings,
    )
