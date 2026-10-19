"""Versioned registry for concrete openCARP patient experiment implementations.

A declared experiment specification describes scientific intent; it must not be
silently translated by a generic catch-all driver.  This registry binds one
``experiment_id`` + ``experiment_version`` pair to one explicit implementation.

Third parties may register implementations through the
``cardiep.opencarp_experiment_drivers`` entry-point group.  A missing,
unavailable, duplicate, or identity-mismatched driver fails closed.
"""
from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from .opencarp_experiment import OpenCARPExperimentSpec


class OpenCARPExperimentDriverError(RuntimeError):
    pass


@runtime_checkable
class OpenCARPExperimentDriver(Protocol):
    experiment_id: str
    experiment_version: str
    driver_version: str

    def available(self) -> bool: ...

    def run(
        self,
        *,
        plan: dict[str, Any],
        experiment: OpenCARPExperimentSpec,
        output_path: str,
    ) -> None: ...


def driver_key(experiment_id: str, experiment_version: str) -> tuple[str, str]:
    selected_id = str(experiment_id).strip()
    selected_version = str(experiment_version).strip()
    if not selected_id or not selected_version:
        raise OpenCARPExperimentDriverError(
            "openCARP experiment driver identity requires non-empty ID and version"
        )
    return selected_id, selected_version


class OpenCARPExperimentDriverRegistry:
    """Registry of explicit openCARP experiment implementations."""

    def __init__(self, drivers: list[OpenCARPExperimentDriver] | None = None) -> None:
        self._drivers: dict[tuple[str, str], OpenCARPExperimentDriver] = {}
        for driver in drivers or []:
            self.register(driver)

    def register(self, driver: OpenCARPExperimentDriver) -> None:
        key = driver_key(driver.experiment_id, driver.experiment_version)
        if not str(driver.driver_version).strip():
            raise OpenCARPExperimentDriverError("driver_version must be explicit")
        if key in self._drivers:
            raise OpenCARPExperimentDriverError(
                f"duplicate openCARP experiment driver for {key[0]!r} version {key[1]!r}"
            )
        self._drivers[key] = driver

    def resolve(self, experiment: OpenCARPExperimentSpec) -> OpenCARPExperimentDriver:
        key = driver_key(experiment.experiment_id, experiment.experiment_version)
        driver = self._drivers.get(key)
        if driver is None:
            raise OpenCARPExperimentDriverError(
                "no registered openCARP implementation for experiment "
                f"{experiment.experiment_id!r} version {experiment.experiment_version!r}"
            )
        try:
            available = bool(driver.available())
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise OpenCARPExperimentDriverError(
                f"openCARP experiment driver availability check failed: {exc}"
            ) from exc
        if not available:
            raise OpenCARPExperimentDriverError(
                "registered openCARP experiment driver is unavailable: "
                f"{experiment.experiment_id!r} version {experiment.experiment_version!r}"
            )
        return driver

    def status(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for (experiment_id, experiment_version), driver in sorted(self._drivers.items()):
            try:
                available = bool(driver.available())
                error = None
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                available = False
                error = f"{type(exc).__name__}: {exc}"
            output.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_version": experiment_version,
                    "driver_version": str(driver.driver_version),
                    "available": available,
                    "implementation": f"{type(driver).__module__}.{type(driver).__name__}",
                    "error": error,
                }
            )
        return output


def discover_opencarp_experiment_drivers() -> OpenCARPExperimentDriverRegistry:
    """Discover explicit experiment implementations from Python entry points."""
    registry = OpenCARPExperimentDriverRegistry()
    eps = entry_points()
    selected = (
        eps.select(group="cardiep.opencarp_experiment_drivers")
        if hasattr(eps, "select")
        else eps.get("cardiep.opencarp_experiment_drivers", ())
    )
    for entry_point in selected:
        try:
            candidate: Any = entry_point.load()
            driver = candidate() if isinstance(candidate, type) else candidate
            if not isinstance(driver, OpenCARPExperimentDriver):
                raise TypeError("entry point does not satisfy OpenCARPExperimentDriver")
            registry.register(driver)
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            warnings.warn(
                f"Could not load openCARP experiment driver {entry_point.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return registry
