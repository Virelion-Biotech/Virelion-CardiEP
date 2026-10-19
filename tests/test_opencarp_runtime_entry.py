from __future__ import annotations

import pytest

from cardiep.opencarp_runtime import OpenCARPRuntimeError
from cardiep.opencarp_runtime_entry import declared_experiment

from .test_opencarp_experiment import _request, _spec


def test_package_runtime_accepts_complete_declared_experiment() -> None:
    request = _request(_spec())
    experiment = declared_experiment(request)
    assert experiment.experiment_id == "patient-ventricular-qrs-confirmation"
    assert len(experiment.fingerprint) == 64


def test_package_runtime_rejects_missing_experiment_spec() -> None:
    with pytest.raises(OpenCARPRuntimeError, match="opencarp_experiment_spec"):
        declared_experiment(_request())


def test_package_runtime_rejects_incompatible_parameter_binding() -> None:
    payload = _spec()
    payload["parameter_bindings"][0]["parameter_unit"] = "mS/cm"
    with pytest.raises(OpenCARPRuntimeError, match="invalid openCARP experiment spec"):
        declared_experiment(_request(payload))
