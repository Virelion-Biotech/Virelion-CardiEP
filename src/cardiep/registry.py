from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Any

from .backends import EPBackend
from .native_backend import NativeEikonalBackend


def discover_backends() -> dict[str, EPBackend]:
    backends: dict[str, EPBackend] = {NativeEikonalBackend.name: NativeEikonalBackend()}
    eps = entry_points()
    selected = (
        eps.select(group="cardiep.backends")
        if hasattr(eps, "select")
        else eps.get("cardiep.backends", ())
    )
    for ep in selected:
        try:
            backend: Any = ep.load()
            backend = backend() if isinstance(backend, type) else backend
            name = str(getattr(backend, "name"))
            if not name:
                continue
            backends[name] = backend
        except Exception as exc:
            warnings.warn(
                f"Could not load CardiEP backend plugin {ep.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return backends


def backend_status(backends: dict[str, EPBackend] | None = None) -> list[dict[str, Any]]:
    selected = backends or discover_backends()
    output = []
    for name in sorted(selected):
        backend = selected[name]
        try:
            available = bool(backend.available())
            error = None
        except Exception as exc:
            available = False
            error = f"{type(exc).__name__}: {exc}"
        output.append(
            {
                "name": name,
                "available": available,
                "implementation": f"{type(backend).__module__}.{type(backend).__name__}",
                "error": error,
            }
        )
    return output
