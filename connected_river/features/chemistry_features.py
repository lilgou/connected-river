"""Survey-date chemistry features with a fixed target-determinand view."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from connected_river.chemistry_registry import exact_feature_columns


def _stable_mean_sql(value: str) -> str:
    """Return an order-independent mean for decimal environmental results."""
    fixed = f"CAST({value} AS DECIMAL(38, 10))"
    return f"CAST(SUM({fixed}) / COUNT({value}) AS DOUBLE)"


def _stable_sample_std_sql(value: str) -> str:
    """Return an order-independent sample SD using fixed-point moments."""
    fixed = f"CAST({value} AS DECIMAL(38, 10))"
    variance = (
        f"(SUM({fixed} * {fixed}) - SUM({fixed}) * SUM({fixed}) "
        f"/ COUNT({value})) / (COUNT({value}) - 1)"
    )
    return (
        f"CASE WHEN COUNT({value}) > 1 THEN SQRT(CAST(GREATEST("
        f"CAST(0 AS DECIMAL(38, 20)), {variance}) AS DOUBLE)) ELSE NULL END"
    )


def _ensure_exact_target_schema(table: pd.DataFrame) -> pd.DataFrame:
    """Add absent exact-parity fields so retired determinands remain explicit."""
    missing = [column for column in exact_feature_columns() if column not in table]
    if not missing:
        return table
    empty = pd.DataFrame(float("nan"), index=table.index, columns=missing)
    return pd.concat([table, empty], axis=1)


def _validate_window_months(window_months: int) -> int:
    if isinstance(window_months, bool) or int(window_months) != window_months:
        raise ValueError("window_months must be a whole number")
    window_months = int(window_months)
    if not 1 <= window_months <= 12:
        raise ValueError(
            "window_months must be between 1 and 12; longer windows require an "
            "required source-download settings"
        )
    return window_months


def _finalize_window_schema(table: pd.DataFrame, window_months: int) -> pd.DataFrame:
    """Keep the published six-month schema but label alternative windows honestly."""

    table = _ensure_exact_target_schema(table)
    if window_months == 6:
        return table
    return table.rename(
        columns={
            column: f"{column[:-4]}_window"
            for column in table.columns
            if column.endswith("_6mo")
        }
    )


# Only samples collected before the biological survey are included. The window
# length is a setting, so the same code can be used for a different study design.
def build_antecedent_chemistry_features(
    surveys: pd.DataFrame,
    selected_matches: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    window_months: int = 6,
    value_column: str = "result_standardized_loq_half",
    assemblages: set[str] | tuple[str, ...] | None = ("macroinvertebrate",),
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Calculate per-survey chemistry statistics for preceding calendar months.

    This implements the cited study's temporal processing: each
    biological survey is a separate observation; measurements at its
    selected chemistry site from the inclusive six-calendar-month antecedent
    window are summarised as minimum, maximum, median, mean and *sample* standard
    deviation.  Overlapping survey windows intentionally reuse measurements.
    Values reported below LOQ use LOQ/2 by default.
    """
    window_months = _validate_window_months(window_months)
    required_surveys = {"site_id", "sample_id", "sample_date", "sample_year"}
    missing = required_surveys - set(surveys)
    if missing:
        raise KeyError(f"Survey table is missing required columns: {sorted(missing)}")
    if value_column not in observations:
        raise KeyError(f"Water-quality observations do not contain {value_column!r}")

    survey = surveys.copy()
    if "assemblage" not in survey:
        survey["assemblage"] = "macroinvertebrate"
    if assemblages is not None:
        survey = survey[survey["assemblage"].isin(assemblages)].copy()
    survey["bio_site_id"] = survey["assemblage"].astype("string") + ":" + survey["site_id"].astype("string")
    analysis_key = survey.get("analysis_id", pd.Series("", index=survey.index)).astype("string").fillna("")
    survey["integration_record_id"] = (
        survey["assemblage"].astype("string") + ":" + survey["sample_id"].astype("string") + ":" + analysis_key
    )
    survey["year"] = pd.to_numeric(survey["sample_year"], errors="coerce").astype("Int64")
    survey["survey_date"] = pd.to_datetime(survey["sample_date"], errors="coerce").dt.normalize()
    # An explicit release year set preserves biological surveys even when a
    # whole year has no defensible chemistry match. Falling back to the match
    # table retains compatibility for direct analytical use of this function.
    eligible_years = (
        [int(year) for year in years]
        if years is not None
        else pd.to_numeric(selected_matches["year"], errors="coerce").dropna().astype(int).unique()
    )
    survey = survey[survey["year"].isin(eligible_years)].copy()
    match_columns = [
        column for column in (
            "bio_site_id", "year", "wq_site_id", "wq_site_name", "match_rank",
            "network_distance_m", "relationship", "selection_confidence",
        ) if column in selected_matches
    ]
    matched = survey.merge(
        selected_matches[match_columns].drop_duplicates(["bio_site_id", "year"]),
        on=["bio_site_id", "year"], how="left", validate="many_to_one",
    )

    chemistry = observations.copy()
    chemistry["sample_datetime"] = pd.to_datetime(chemistry["sample_datetime"], errors="coerce").dt.normalize()
    chemistry["chemistry_value"] = pd.to_numeric(chemistry[value_column], errors="coerce")
    if "target_determinand_key" not in chemistry:
        chemistry["target_determinand_key"] = pd.NA
    chemistry = chemistry[
        chemistry["determinand_key"].notna()
        & chemistry["chemistry_value"].notna()
        & chemistry["wq_site_id"].notna()
    ].copy()
    by_station = {
        str(key): group.sort_values("sample_datetime")
        for key, group in chemistry.groupby("wq_site_id", sort=False)
    }

    long_rows: list[dict[str, object]] = []
    survey_rows: list[dict[str, object]] = []
    # Preserve all survey/site fields and interpreted biological indices.  The
    # matching merge has already selected the match columns needed for checking, and the
    # only helper that should not escape into the product is ``survey_date``.
    identity = [column for column in matched.columns if column != "survey_date"]
    for row in matched.itertuples(index=False):
        values = row._asdict()
        survey_date = values.get("survey_date")
        start = survey_date - pd.DateOffset(months=window_months) if pd.notna(survey_date) else pd.NaT
        base = {column: values.get(column) for column in identity}
        base.update({
            "chemistry_window_start": start,
            "chemistry_window_end": survey_date,
            "chemistry_window_months": int(window_months),
            "chemistry_temporal_method": "antecedent_calendar_months",
            "below_loq_treatment": "LOQ/2" if value_column.endswith("loq_half") else value_column,
        })
        station = by_station.get(str(values.get("wq_site_id")))
        if station is None or pd.isna(survey_date):
            base["chemistry_determinand_count_6mo"] = 0
            base["target_exact_determinand_count_6mo"] = 0
            survey_rows.append(base)
            continue
        window = station[station["sample_datetime"].between(start, survey_date, inclusive="both")]
        matching_window = (
            window[window["is_matching_determinand"].fillna(False)]
            if "is_matching_determinand" in window else window
        )
        # The general support score retains its declared chemistry scope. The
        # additional Table A.1 codes are still summarised below
        # in both readable determinand fields and the exact-code family, but do
        # not silently change the meaning of the public evidence score.
        base["chemistry_determinand_count_6mo"] = int(
            matching_window["determinand_key"].nunique()
        )
        base["target_exact_determinand_count_6mo"] = int(
            window["target_determinand_key"].nunique()
        )
        survey_rows.append(base)
        for determinand, group in window.groupby("determinand_key"):
            measured = group["chemistry_value"].dropna()
            long_rows.append({
                "integration_record_id": values.get("integration_record_id"), "determinand_key": determinand,
                "observation_count_6mo": int(measured.count()),
                "minimum_6mo": measured.min(), "maximum_6mo": measured.max(),
                "median_6mo": measured.median(), "mean_6mo": measured.mean(),
                "std_6mo": measured.std(ddof=1),
                "below_loq_count_6mo": int(group["result_qualifier"].eq("<").sum()),
                "below_loq_fraction_6mo": float(group["result_qualifier"].eq("<").mean()),
                "above_range_count_6mo": int(group["result_qualifier"].eq(">").sum()),
                "above_range_fraction_6mo": float(group["result_qualifier"].eq(">").mean()),
            })
        exact_window = window[window["target_determinand_key"].notna()]
        for determinand, group in exact_window.groupby("target_determinand_key"):
            measured = group["chemistry_value"].dropna()
            long_rows.append({
                "integration_record_id": values.get("integration_record_id"),
                "determinand_key": f"target_exact_{determinand}",
                "observation_count_6mo": int(measured.count()),
                "minimum_6mo": measured.min(), "maximum_6mo": measured.max(),
                "median_6mo": measured.median(), "mean_6mo": measured.mean(),
                "std_6mo": measured.std(ddof=1),
                "below_loq_count_6mo": int(group["result_qualifier"].eq("<").sum()),
                "below_loq_fraction_6mo": float(group["result_qualifier"].eq("<").mean()),
                "above_range_count_6mo": int(group["result_qualifier"].eq(">").sum()),
                "above_range_fraction_6mo": float(group["result_qualifier"].eq(">").mean()),
            })

    result = pd.DataFrame(survey_rows)
    if not long_rows:
        return _finalize_window_schema(result, window_months)
    long = pd.DataFrame(long_rows)
    metrics = [
        "observation_count_6mo", "minimum_6mo", "maximum_6mo", "median_6mo", "mean_6mo",
        "std_6mo", "below_loq_count_6mo", "below_loq_fraction_6mo",
        "above_range_count_6mo", "above_range_fraction_6mo",
    ]
    wide = long.pivot(index="integration_record_id", columns="determinand_key", values=metrics)
    wide.columns = [f"{determinand}_{metric}" for metric, determinand in wide.columns]
    merged = result.merge(
        wide.reset_index(), on="integration_record_id", how="left", validate="one_to_one",
    )
    return _finalize_window_schema(merged, window_months)


def build_antecedent_chemistry_features_from_path(
    surveys: pd.DataFrame,
    selected_matches: pd.DataFrame,
    observations_path: str | Path,
    *,
    window_months: int = 6,
    value_column: str = "result_standardized_loq_half",
    assemblages: set[str] | tuple[str, ...] | None = ("macroinvertebrate",),
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Disk-backed equivalent of :func:`build_antecedent_chemistry_features`.

    DuckDB performs the station/date range join and grouped summaries directly
    against Parquet. This preserves the survey-specific windows while
    avoiding an in-memory copy and per-station dictionary of the national
    observation archive.
    """
    window_months = _validate_window_months(window_months)
    required_surveys = {"site_id", "sample_id", "sample_date", "sample_year"}
    missing = required_surveys - set(surveys)
    if missing:
        raise KeyError(f"Survey table is missing required columns: {sorted(missing)}")
    observations_path = Path(observations_path)
    available_columns = set(pq.ParquetFile(observations_path).schema_arrow.names)
    if value_column not in available_columns:
        raise KeyError(f"Water-quality observations do not contain {value_column!r}")

    survey = surveys.copy()
    if "assemblage" not in survey:
        survey["assemblage"] = "macroinvertebrate"
    if assemblages is not None:
        survey = survey[survey["assemblage"].isin(assemblages)].copy()
    survey["bio_site_id"] = (
        survey["assemblage"].astype("string") + ":" + survey["site_id"].astype("string")
    )
    analysis_key = survey.get("analysis_id", pd.Series("", index=survey.index)).astype("string").fillna("")
    survey["integration_record_id"] = (
        survey["assemblage"].astype("string") + ":"
        + survey["sample_id"].astype("string") + ":" + analysis_key
    )
    survey["year"] = pd.to_numeric(survey["sample_year"], errors="coerce").astype("Int64")
    survey["survey_date"] = pd.to_datetime(survey["sample_date"], errors="coerce").dt.normalize()
    eligible_years = (
        [int(year) for year in years]
        if years is not None
        else pd.to_numeric(selected_matches["year"], errors="coerce").dropna().astype(int).unique()
    )
    survey = survey[survey["year"].isin(eligible_years)].copy()
    match_columns = [
        column for column in (
            "bio_site_id", "year", "wq_site_id", "wq_site_name", "match_rank",
            "network_distance_m", "relationship", "selection_confidence",
        ) if column in selected_matches
    ]
    matched = survey.merge(
        selected_matches[match_columns].drop_duplicates(["bio_site_id", "year"]),
        on=["bio_site_id", "year"], how="left", validate="many_to_one",
    )
    matched["chemistry_window_start"] = matched["survey_date"] - pd.DateOffset(months=window_months)

    identity = [column for column in matched if column not in {"survey_date", "chemistry_window_start"}]
    result = matched[identity].copy()
    result["chemistry_window_start"] = matched["chemistry_window_start"]
    result["chemistry_window_end"] = matched["survey_date"]
    result["chemistry_window_months"] = int(window_months)
    result["chemistry_temporal_method"] = "antecedent_calendar_months"
    result["below_loq_treatment"] = (
        "LOQ/2" if value_column.endswith("loq_half") else value_column
    )
    result["chemistry_determinand_count_6mo"] = 0
    result["target_exact_determinand_count_6mo"] = 0

    survey_windows = matched[
        ["integration_record_id", "wq_site_id", "survey_date", "chemistry_window_start"]
    ].dropna(subset=["wq_site_id", "survey_date", "chemistry_window_start"])
    if survey_windows.empty:
        return _finalize_window_schema(result, window_months)

    sql_path = str(observations_path.resolve()).replace("'", "''")
    safe_value_column = value_column.replace('"', '""')
    sql_value = f'observations."{safe_value_column}"'
    stable_mean = _stable_mean_sql(sql_value)
    stable_std = _stable_sample_std_sql(sql_value)
    connection = duckdb.connect()
    try:
        # Floating-point AVG/STDDEV reductions are order-sensitive at the last
        # few binary digits. Parallel group aggregation allowed otherwise
        # identical release builds to visit observations in a different order.
        # Use the stable Parquet scan order for release-valued statistics; the
        # small throughput cost is preferable to scientifically identical but
        # byte-different public products.
        connection.execute("PRAGMA threads=1")
        connection.execute("PRAGMA preserve_insertion_order=true")
        connection.register("survey_windows", survey_windows)
        exact_union = ""
        if "target_determinand_key" in available_columns:
            exact_union = f"""
                UNION ALL
                SELECT
                    windows.integration_record_id,
                    'target_exact_' || observations.target_determinand_key
                        AS determinand_key,
                    'target_exact' AS feature_family,
                    count(observations."{safe_value_column}") AS observation_count_6mo,
                    min(observations."{safe_value_column}") AS minimum_6mo,
                    max(observations."{safe_value_column}") AS maximum_6mo,
                    median(observations."{safe_value_column}") AS median_6mo,
                    {stable_mean} AS mean_6mo,
                    {stable_std} AS std_6mo,
                    sum(CASE WHEN observations.result_qualifier = '<' THEN 1 ELSE 0 END)
                        AS below_loq_count_6mo,
                    avg(CASE WHEN observations.result_qualifier = '<' THEN 1.0 ELSE 0.0 END)
                        AS below_loq_fraction_6mo,
                    sum(CASE WHEN observations.result_qualifier = '>' THEN 1 ELSE 0 END)
                        AS above_range_count_6mo,
                    avg(CASE WHEN observations.result_qualifier = '>' THEN 1.0 ELSE 0.0 END)
                        AS above_range_fraction_6mo
                FROM survey_windows AS windows
                JOIN read_parquet('{sql_path}') AS observations
                  ON windows.wq_site_id = observations.wq_site_id
                 AND observations.sample_datetime BETWEEN
                     windows.chemistry_window_start AND windows.survey_date
                WHERE observations.target_determinand_key IS NOT NULL
                  AND observations."{safe_value_column}" IS NOT NULL
                GROUP BY windows.integration_record_id, observations.target_determinand_key
            """
        feature_family_case = (
            "CASE WHEN bool_or(coalesce(observations.is_matching_determinand, false)) "
            "THEN 'matching_target' ELSE 'additional_target' END"
            if "is_matching_determinand" in available_columns else "'matching_target'"
        )
        long = connection.sql(f"""
            SELECT
                windows.integration_record_id,
                observations.determinand_key,
                {feature_family_case} AS feature_family,
                count(observations."{safe_value_column}") AS observation_count_6mo,
                min(observations."{safe_value_column}") AS minimum_6mo,
                max(observations."{safe_value_column}") AS maximum_6mo,
                median(observations."{safe_value_column}") AS median_6mo,
                {stable_mean} AS mean_6mo,
                {stable_std} AS std_6mo,
                sum(CASE WHEN observations.result_qualifier = '<' THEN 1 ELSE 0 END)
                    AS below_loq_count_6mo,
                avg(CASE WHEN observations.result_qualifier = '<' THEN 1.0 ELSE 0.0 END)
                    AS below_loq_fraction_6mo,
                sum(CASE WHEN observations.result_qualifier = '>' THEN 1 ELSE 0 END)
                    AS above_range_count_6mo,
                avg(CASE WHEN observations.result_qualifier = '>' THEN 1.0 ELSE 0.0 END)
                    AS above_range_fraction_6mo
            FROM survey_windows AS windows
            JOIN read_parquet('{sql_path}') AS observations
              ON windows.wq_site_id = observations.wq_site_id
             AND observations.sample_datetime BETWEEN
                 windows.chemistry_window_start AND windows.survey_date
            WHERE observations.determinand_key IS NOT NULL
              AND observations."{safe_value_column}" IS NOT NULL
            GROUP BY windows.integration_record_id, observations.determinand_key
            {exact_union}
        """).df()
    finally:
        connection.close()

    if long.empty:
        return _finalize_window_schema(result, window_months)
    matching_long = long[long["feature_family"].eq("matching_target")]
    exact_long = long[long["feature_family"].eq("target_exact")]
    determinand_counts = matching_long.groupby("integration_record_id")["determinand_key"].nunique()
    result["chemistry_determinand_count_6mo"] = (
        result["integration_record_id"].map(determinand_counts).fillna(0).astype(int)
    )
    exact_counts = exact_long.groupby("integration_record_id")["determinand_key"].nunique()
    result["target_exact_determinand_count_6mo"] = (
        result["integration_record_id"].map(exact_counts).fillna(0).astype(int)
    )
    metrics = [
        "observation_count_6mo", "minimum_6mo", "maximum_6mo", "median_6mo", "mean_6mo",
        "std_6mo", "below_loq_count_6mo", "below_loq_fraction_6mo",
        "above_range_count_6mo", "above_range_fraction_6mo",
    ]
    wide = long.pivot(index="integration_record_id", columns="determinand_key", values=metrics)
    wide.columns = [f"{determinand}_{metric}" for metric, determinand in wide.columns]
    merged = result.merge(
        wide.reset_index(), on="integration_record_id", how="left", validate="one_to_one",
    )
    return _finalize_window_schema(merged, window_months)
