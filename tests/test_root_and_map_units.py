import json
from pathlib import Path

import numpy as np
import pytest

from cardiep import (
    ArtifactRef,
    EPObservation,
    RootSchedule,
    evaluate_observations,
    resolve_root_schedule,
    synthetic_tetra_geometry,
)


def test_root_schedule_rejects_fractional_node_ids() -> None:
    with pytest.raises(ValueError, match="integer node indices"):
        RootSchedule(
            nodes=np.asarray([0.0, 1.9]),
            activation_ms=np.asarray([0.0, 5.0]),
            method="test",
        )


def test_resolve_roots_rejects_fractional_ids_and_bad_auto_count() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)

    with pytest.raises(ValueError, match="integer node indices"):
        resolve_root_schedule(
            geometry,
            {"root_nodes": [0, 1.5]},
            {},
        )

    with pytest.raises(ValueError, match="positive integer"):
        resolve_root_schedule(
            geometry,
            {"auto_root_count": 1.5},
            {},
        )

    with pytest.raises(ValueError, match="positive integer"):
        resolve_root_schedule(
            geometry,
            {"auto_root_count": 0},
            {},
        )


def test_root_activation_mapping_must_cover_every_root() -> None:
    geometry = synthetic_tetra_geometry(with_fibres=True)
    with pytest.raises(ValueError, match="missing root nodes"):
        resolve_root_schedule(
            geometry,
            {
                "root_nodes": [0, 1],
                "root_activation_ms": {"0": 0.0},
            },
            {},
        )


def _observation(
    tmp_path: Path,
    *,
    payload: dict,
    units: str | None,
    kind: str = "activation_map",
) -> EPObservation:
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return EPObservation(
        observation_id="obs",
        kind=kind,
        artifact=ArtifactRef(
            artifact_id="obs-artifact",
            kind=kind,
            uri=path.as_uri(),
        ),
        units=units,
    )


def test_activation_map_in_seconds_is_converted_to_ms(tmp_path: Path) -> None:
    observation = _observation(
        tmp_path,
        payload={"values": [0.0, 0.01, 0.02, 0.04]},
        units="s",
    )
    report = evaluate_observations(
        [observation],
        activation_ms=np.asarray([0.0, 10.0, 20.0, 40.0]),
        repolarization_ms=np.asarray([250.0, 260.0, 270.0, 290.0]),
        ecg=None,
        hints=[
            {
                "term_id": "lat",
                "observation_id": "obs",
                "model_output": "activation_map",
                "discrepancy": "rmse",
                "weight": 1.0,
            }
        ],
    )
    assert report.objective == pytest.approx(0.0)


def test_generic_time_map_requires_declared_units(tmp_path: Path) -> None:
    observation = _observation(
        tmp_path,
        payload={"values": [0.0, 10.0, 20.0, 40.0]},
        units=None,
    )
    with pytest.raises(ValueError, match="must declare time units"):
        evaluate_observations(
            [observation],
            activation_ms=np.asarray([0.0, 10.0, 20.0, 40.0]),
            repolarization_ms=np.asarray([250.0, 260.0, 270.0, 290.0]),
            ecg=None,
            hints=[
                {
                    "term_id": "lat",
                    "observation_id": "obs",
                    "model_output": "activation_map",
                    "discrepancy": "rmse",
                    "weight": 1.0,
                }
            ],
        )


def test_ms_named_field_rejects_contradictory_seconds_units(tmp_path: Path) -> None:
    observation = _observation(
        tmp_path,
        payload={"values_ms": [0.0, 10.0, 20.0, 40.0]},
        units="s",
    )
    with pytest.raises(ValueError, match="implies ms"):
        evaluate_observations(
            [observation],
            activation_ms=np.asarray([0.0, 10.0, 20.0, 40.0]),
            repolarization_ms=np.asarray([250.0, 260.0, 270.0, 290.0]),
            ecg=None,
            hints=[
                {
                    "term_id": "lat",
                    "observation_id": "obs",
                    "model_output": "activation_map",
                    "discrepancy": "rmse",
                    "weight": 1.0,
                }
            ],
        )


def test_repolarization_map_microseconds_is_converted(tmp_path: Path) -> None:
    observation = _observation(
        tmp_path,
        payload={"values": [250000.0, 260000.0, 270000.0, 290000.0]},
        units="us",
        kind="repolarization_map",
    )
    report = evaluate_observations(
        [observation],
        activation_ms=np.asarray([0.0, 10.0, 20.0, 40.0]),
        repolarization_ms=np.asarray([250.0, 260.0, 270.0, 290.0]),
        ecg=None,
        hints=[
            {
                "term_id": "repol",
                "observation_id": "obs",
                "model_output": "repolarization_map",
                "discrepancy": "rmse",
                "weight": 1.0,
            }
        ],
    )
    assert report.objective == pytest.approx(0.0)
