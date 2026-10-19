from __future__ import annotations

import os

from cardiep.camps_runtime_entry import _configure_default_driver


def test_default_driver_uses_packaged_qualified_executable(monkeypatch) -> None:
    monkeypatch.delenv("CARDIEP_CAMPS_DRIVER", raising=False)
    monkeypatch.setattr(
        "cardiep.camps_runtime_entry.shutil.which",
        lambda name: "/opt/cardiep/bin/cardiep-camps-qualified-driver"
        if name == "cardiep-camps-qualified-driver"
        else None,
    )
    _configure_default_driver()
    value = os.environ["CARDIEP_CAMPS_DRIVER"]
    assert "cardiep-camps-qualified-driver" in value
    assert "{plan}" in value
    assert "{output}" in value


def test_explicit_driver_is_never_overwritten(monkeypatch) -> None:
    monkeypatch.setenv(
        "CARDIEP_CAMPS_DRIVER",
        "custom-driver {plan} {output}",
    )
    monkeypatch.setattr(
        "cardiep.camps_runtime_entry.shutil.which",
        lambda name: "/should/not/be/used",
    )
    _configure_default_driver()
    assert os.environ["CARDIEP_CAMPS_DRIVER"] == "custom-driver {plan} {output}"
