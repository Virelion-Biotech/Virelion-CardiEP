from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any


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
