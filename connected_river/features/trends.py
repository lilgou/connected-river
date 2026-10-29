"""Repeat-site descriptive trends with monitoring-effort diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _slope(year: pd.Series, value: pd.Series) -> float:
    valid = pd.to_numeric(year, errors="coerce").notna() & pd.to_numeric(value, errors="coerce").notna()
    if valid.sum() < 3 or pd.to_numeric(year[valid]).nunique() < 3:
        return np.nan
    return float(np.polyfit(pd.to_numeric(year[valid]), pd.to_numeric(value[valid]), 1)[0])


# Summarise change only where a site has repeated observations. A simple slope
# is easier to interpret here than a model fitted to sparse national coverage.
def build_repeated_site_trends(
    site_year: pd.DataFrame, output_path: str | Path, *, start_year: int = 2015, minimum_years: int = 4,
) -> Path:
    """Describe repeated observations; slopes are not automatically ecological condition trends."""
    recent = site_year[pd.to_numeric(site_year["sample_year"], errors="coerce").ge(start_year)].copy()
    rows = []
    for (assemblage, site_id), group in recent.groupby(["assemblage", "site_id"], dropna=False):
        if group["sample_year"].nunique() < minimum_years:
            continue
        group = group.sort_values("sample_year")
        rows.append({
            "assemblage": assemblage, "site_id": site_id, "site_label": group["site_label"].dropna().iloc[0] if group["site_label"].notna().any() else None,
            "catchment": group["catchment"].dropna().iloc[0] if group["catchment"].notna().any() else None,
            "water_body": group["water_body"].dropna().iloc[0] if group["water_body"].notna().any() else None,
            "first_year": int(group["sample_year"].min()), "last_year": int(group["sample_year"].max()),
            "observed_years": int(group["sample_year"].nunique()),
            "taxon_richness_slope_per_year": _slope(group["sample_year"], group["taxon_richness_max"]),
            "abundance_slope_per_year": _slope(group["sample_year"], group["abundance_total"]),
            "sample_count_slope_per_year": _slope(group["sample_year"], group["sample_count"]),
            "median_taxon_richness": pd.to_numeric(group["taxon_richness_max"], errors="coerce").median(),
            "median_annual_samples": pd.to_numeric(group["sample_count"], errors="coerce").median(),
            "trend_interpretation": "descriptive repeat-site observation; monitoring effort and method may confound ecological change",
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path
