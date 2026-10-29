"""Checks used when writing the release tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SITE_YEAR_REQUIRED = (
    "evidence_id", "assemblage", "site_id", "sample_year", "sample_count",
    "easting", "northing", "latitude", "longitude",
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    severity: str
    detail: str


# Run a few clear checks on the finished site-year table. These catch duplicate
# rows and missing keys before a release is written.
def validate_site_year(df: pd.DataFrame) -> list[Check]:
    missing = [c for c in SITE_YEAR_REQUIRED if c not in df]
    checks = [Check("required_columns", not missing, "error", "missing: " + ", ".join(missing) if missing else "all present")]
    if missing:
        return checks
    checks.extend([
        Check("non_empty", len(df) > 0, "error", f"{len(df):,} rows"),
        Check("unique_evidence_id", not df["evidence_id"].duplicated().any(), "error", f"{int(df['evidence_id'].duplicated().sum()):,} duplicates"),
        Check("valid_year", pd.to_numeric(df["sample_year"], errors="coerce").between(1900, 2100).all(), "error", "years must be 1900-2100"),
        Check("positive_sample_count", (pd.to_numeric(df["sample_count"], errors="coerce") > 0).all(), "error", "all rows must represent observations"),
        Check("coordinate_pair_consistency", (df["easting"].notna() == df["northing"].notna()).all(), "error", "OSGB coordinate pairs must be complete"),
        Check("assemblage_coverage", df["assemblage"].nunique() >= 4, "warning", f"{df['assemblage'].nunique()} assemblages"),
    ])
    return checks


# Keep the check results beside the data rather than relying on terminal output
# that will disappear after the run.
def write_qa(checks: list[Check], path: str | Path, *, fail_on_error: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "passed" if all(c.passed or c.severity != "error" for c in checks) else "failed", "checks": [asdict(c) for c in checks]}
    path.write_text(
        json.dumps(payload, indent=2, default=lambda value: value.item()),
        encoding="utf-8",
    )
    failures = [c for c in checks if not c.passed and c.severity == "error"]
    if failures and fail_on_error:
        raise ValueError("Release QA failed: " + "; ".join(f"{c.name}: {c.detail}" for c in failures))
    return path
