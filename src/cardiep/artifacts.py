from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import ArtifactRef
from .provenance import file_sha256


def write_json_artifact(
    output_dir: str | Path,
    *,
    artifact_id: str,
    kind: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> ArtifactRef:
    if not artifact_id.strip() or artifact_id in {".", ".."} or any(c in artifact_id for c in ("/", "\\", "\x00")):
        raise ValueError("artifact_id must be a safe non-empty filename")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    fd, temporary = tempfile.mkstemp(prefix=".cardiep-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_uri(),
        sha256=file_sha256(path),
        metadata=dict(metadata or {}),
    )
