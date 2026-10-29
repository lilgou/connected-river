"""Machine-readable temporal coverage for every evidence stream in a release."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _year(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.year.astype("Int64")
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.notna().any():
        return parsed.astype("Int64")
    return pd.to_datetime(values, errors="coerce").dt.year.astype("Int64")


def _rows(
    table: pd.DataFrame,
    *,
    source: str,
    evidence_type: str,
    year_column: str,
    group_column: str | None = None,
    site_column: str | None = None,
) -> list[dict[str, object]]:
    if table.empty or year_column not in table:
        return []
    work = table.copy()
    work["_coverage_year"] = _year(work[year_column])
    groups = work.groupby(group_column, dropna=False) if group_column and group_column in work else [("all", work)]
    result: list[dict[str, object]] = []
    for component, frame in groups:
        years = frame["_coverage_year"].dropna()
        result.append({
            "source": source,
            "evidence_type": evidence_type,
            "component": str(component),
            "first_year": int(years.min()) if len(years) else pd.NA,
            "latest_year": int(years.max()) if len(years) else pd.NA,
            "record_count": len(frame),
            "distinct_sites": (
                int(frame[site_column].nunique())
                if site_column and site_column in frame else pd.NA
            ),
        })
    return result


def _parquet_rows(
    path: str | Path,
    *,
    source: str,
    evidence_type: str,
    year_column: str,
    group_column: str | None = None,
    site_column: str | None = None,
) -> list[dict[str, object]]:
    """Aggregate very large release tables in DuckDB instead of pandas."""
    sql_path = str(Path(path).resolve()).replace("'", "''")

    def quote(column: str) -> str:
        return '"' + column.replace('"', '""') + '"'

    year_value = (
        f"coalesce(try_cast({quote(year_column)} AS INTEGER), "
        f"year(try_cast({quote(year_column)} AS TIMESTAMP)))"
    )
    component = quote(group_column) if group_column else "'all'"
    distinct_sites = f"count(DISTINCT {quote(site_column)})" if site_column else "NULL"
    group_by = f"GROUP BY {component}" if group_column else ""
    connection = duckdb.connect()
    try:
        table = connection.sql(f"""
            SELECT
                cast({component} AS VARCHAR) AS component,
                min({year_value}) AS first_year,
                max({year_value}) AS latest_year,
                count(*) AS record_count,
                {distinct_sites} AS distinct_sites
            FROM read_parquet('{sql_path}')
            {group_by}
            ORDER BY component
        """).df()
    finally:
        connection.close()
    return [{
        "source": source,
        "evidence_type": evidence_type,
        "component": str(row.component),
        "first_year": int(row.first_year) if pd.notna(row.first_year) else pd.NA,
        "latest_year": int(row.latest_year) if pd.notna(row.latest_year) else pd.NA,
        "record_count": int(row.record_count),
        "distinct_sites": int(row.distinct_sites) if pd.notna(row.distinct_sites) else pd.NA,
    } for row in table.itertuples(index=False)]


# Report the years actually found in each prepared source. Configured dates are
# useful expectations, but they are not a substitute for checking the files.
def build_source_temporal_coverage(
    *,
    ecology_samples_path: str | Path,
    water_quality_observations_path: str | Path,
    nrfa_daily_flows_path: str | Path,
    edm_asset_year_path: str | Path,
    output_path: str | Path,
    rsn_site_year_path: str | Path | None = None,
) -> Path:
    """Summarise the years actually observed in the completed run."""
    rows: list[dict[str, object]] = []
    rows += _rows(
        pd.read_parquet(ecology_samples_path), source="EA Ecology & Fish bulk archive",
        evidence_type="biological survey", year_column="sample_year",
        group_column="assemblage", site_column="site_id",
    )
    rows += _parquet_rows(
        water_quality_observations_path, source="EA Water Quality Archive",
        evidence_type="river chemistry observation", year_column="year",
        group_column="determinand_key", site_column="wq_site_id",
    )
    rows += _parquet_rows(
        nrfa_daily_flows_path, source="National River Flow Archive",
        evidence_type="quality-controlled daily flow", year_column="date",
        site_column="nrfa_station_id",
    )
    edm = pd.read_parquet(edm_asset_year_path)
    if {"reporting_year", "asset_id"}.issubset(edm.columns):
        valid_assets = edm.groupby("reporting_year")["asset_id"].nunique(dropna=True)
        complete_years = valid_assets[valid_assets.ge(100)].index
        edm = edm[edm["reporting_year"].isin(complete_years)].copy()
    rows += _rows(
        edm, source="EA EDM annual returns",
        evidence_type="storm-overflow asset-year", year_column="reporting_year",
        site_column="asset_id",
    )
    if rsn_site_year_path and Path(rsn_site_year_path).exists():
        rsn = pd.read_parquet(rsn_site_year_path)
        year_column = next((column for column in ("year", "sample_year", "survey_year") if column in rsn), None)
        if year_column:
            rows += _rows(
                rsn, source="NCEA River Surveillance Network",
                evidence_type="integrated surveillance evidence", year_column=year_column,
                site_column=next((column for column in ("network_id", "site_id") if column in rsn), None),
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(rows).sort_values(["evidence_type", "source", "component"])
    if output_path.suffix.casefold() == ".parquet":
        coverage.to_parquet(output_path, index=False)
    else:
        coverage.to_csv(output_path, index=False)
    return output_path
