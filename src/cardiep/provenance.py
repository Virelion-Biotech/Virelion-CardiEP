from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uri_to_path(uri: str, *, relative_to: str | Path | None = None) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in {"", None}:
        path = Path(unquote(uri))
    elif parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    else:
        raise ValueError(f"Only local file artifacts are supported by the native backend: {uri}")
    if not path.is_absolute() and relative_to is not None:
        path = Path(relative_to) / path
    return path.expanduser().resolve()


def verify_file_sha256(path: str | Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    actual = file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Artifact SHA-256 mismatch for {Path(path)}: "
            f"expected {expected_sha256.lower()}, got {actual.lower()}"
        )



def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def package_source_sha256() -> str:
    """Hash installed CardiEP Python sources so code changes cannot reuse run IDs."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(root.glob("*.py"), key=lambda item: item.name)
    if not paths:
        raise RuntimeError(f"No CardiEP Python sources found under {root}")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def native_runtime_fingerprint() -> dict[str, Any]:
    details = {
        "cardiep_version": _distribution_version("virelion-cardiep"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_machine": platform.machine(),
        "numpy_version": _distribution_version("numpy"),
        "pydantic_version": _distribution_version("pydantic"),
        "source_sha256": package_source_sha256(),
    }
    return {
        **details,
        "fingerprint_sha256": sha256_json(details),
    }
