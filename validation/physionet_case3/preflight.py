#!/usr/bin/env python3
"""Public-data preflight for the PhysioNet/CinC 2007 Case 3 pilot.

Downloads only the open Case 3 text assets, records hashes, and reports their
raw numerical structure. This stage deliberately performs no fitting: the goal
is to resolve source-format/sample-rate ambiguity and freeze the eventual
calibration/holdout split before any model sees the held-out leads.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://physionet.org/files/challenge-2007/1.0.0/data"
FILES = {
    "case3_ventr.tri": f"{BASE}/case0003-tri/case3_ventr.tri",
    "case3_torso.tri": f"{BASE}/case0003-tri/case3_torso.tri",
    "case3_lendo.tri": f"{BASE}/case0003-tri/case3_lendo.tri",
    "case3_rendo.tri": f"{BASE}/case0003-tri/case3_rendo.tri",
    "elnodes_120_352.lst": f"{BASE}/case0003-tri/elnodes_120_352.lst",
    "elnodes_123_370.lst": f"{BASE}/case0003-tri/elnodes_123_370.lst",
    "ecgs_case3.asc": f"{BASE}/case0003-tri/ecgs_case3.asc",
    "case0003.dat": f"{BASE}/case0003.dat",
    "readme.txt": f"{BASE}/readme.txt",
}

CALIBRATION_LEADS = ["I", "II", "V1", "V2", "V4"]
PRIMARY_HOLDOUT_LEADS = ["V5", "V6"]
SECONDARY_CHECK_LEADS = ["III", "aVR", "aVL", "aVF", "V3"]


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Virelion-CardiEP-Case3-preflight/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_tokens(line: str) -> list[float] | None:
    text = line.strip()
    if not text:
        return None
    parts = re.split(r"[\s,]+", text)
    values: list[float] = []
    try:
        for part in parts:
            if part:
                values.append(float(part))
    except ValueError:
        return None
    return values or None


def inspect_text(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    numeric = [(index + 1, values) for index, line in enumerate(lines) if (values := numeric_tokens(line))]
    widths = Counter(len(values) for _, values in numeric)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "line_count": len(lines),
        "numeric_line_count": len(numeric),
        "numeric_width_histogram": dict(sorted(widths.items())),
        "first_text_lines": lines[:8],
        "last_text_lines": lines[-12:],
        "first_numeric_rows": [
            {"line": line, "values": values[:16], "width": len(values)}
            for line, values in numeric[:8]
        ],
        "last_numeric_rows": [
            {"line": line, "values": values[:16], "width": len(values)}
            for line, values in numeric[-8:]
        ],
    }


def infer_dimension_headers(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    candidates = []
    for index, line in enumerate(lines):
        values = numeric_tokens(line)
        if values is None or len(values) != 2:
            continue
        if not all(float(item).is_integer() and item >= 1 for item in values):
            continue
        rows, cols = map(int, values)
        if rows > 100000 or cols > 10000:
            continue
        following_widths = []
        for following in lines[index + 1 : index + 1 + min(rows, 5)]:
            parsed = numeric_tokens(following)
            if parsed is not None:
                following_widths.append(len(parsed))
        candidates.append(
            {
                "line": index + 1,
                "rows": rows,
                "cols": cols,
                "next_numeric_widths": following_widths,
            }
        )
    return candidates[:20]


def main() -> None:
    root = Path("case3_preflight").resolve()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "cardiep-physionet-case3-preflight-v1",
        "source": "PhysioNet/CinC Challenge 2007 Case 3",
        "coordinate_unit": "mm",
        "calibration_leads_frozen_before_fit": CALIBRATION_LEADS,
        "primary_holdout_leads_frozen_before_fit": PRIMARY_HOLDOUT_LEADS,
        "secondary_check_leads": SECONDARY_CHECK_LEADS,
        "holdout_semantics": (
            "same-acquisition spatial lead holdout; not an independent visit or repeat ECG"
        ),
        "files": {},
    }
    for name, url in FILES.items():
        path = raw / name
        print(f"Downloading {url}")
        download(url, path)
        info = inspect_text(path)
        info["url"] = url
        info["dimension_header_candidates"] = infer_dimension_headers(path)
        report["files"][name] = info
        print(f"{name}: bytes={info['bytes']} sha256={info['sha256']}")
        print("  widths:", info["numeric_width_histogram"])
        print("  first:", json.dumps(info["first_text_lines"], ensure_ascii=False))
        print("  last:", json.dumps(info["last_text_lines"], ensure_ascii=False))
        print("  dim candidates:", info["dimension_header_candidates"][:5])

    report_path = root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
