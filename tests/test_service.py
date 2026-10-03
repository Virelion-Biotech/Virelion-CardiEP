import pytest

from cardiep import ArtifactRef, CardiEPService, EPParameterSet, EPSimulationRequest
from cardiep.backends import BackendUnavailable


def test_service_fails_closed_without_backend() -> None:
    request = EPSimulationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(
            artifact_id="m",
            kind="mesh",
            uri="file:///mesh",
        ),
        backend="missing",
        parameters=EPParameterSet(values={"speed": 0.6}),
    )
    with pytest.raises(BackendUnavailable):
        CardiEPService().simulate(request)
