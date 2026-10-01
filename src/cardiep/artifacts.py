from __future__ import annotations

import json
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
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_uri(),
        sha256=file_sha256(path),
        metadata=dict(metadata or {}),
    )
