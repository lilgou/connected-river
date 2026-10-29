"""Antecedent hydrological context for biological observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from connected_river.features.flow_regime import build_flow_regime_features


# Flow summaries use the same before-survey idea as chemistry: only information
# available up to the biological sample date is included.
def build_antecedent_flow_features(
    biology_site_year: pd.DataFrame, selected_gauges: pd.DataFrame, daily_flows: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise flow on and before the first biological survey in each site-year."""
    # Selections are year-aware even when NRFA has no observation in that year.
    year_aware = "year" in selected_gauges
    merge_keys = ["bio_site_id", "year"] if year_aware else ["bio_site_id"]
    biology = biology_site_year.merge(
        selected_gauges, on=merge_keys, how="left",
        validate="one_to_one" if year_aware else "many_to_one",
    )
    biology["survey_date"] = pd.to_datetime(biology["first_sample_date_in_year"], errors="coerce")
    flows = daily_flows.copy()
    flows["date"] = pd.to_datetime(flows["date"], errors="coerce")
    flows["flow_m3s"] = pd.to_numeric(flows["flow_m3s"], errors="coerce")
    if "flow_station_id" not in flows:
        flows["flow_station_id"] = flows["nrfa_station_id"].astype("Int64").astype("string")
    if "nrfa_station_id" not in flows:
        flows["nrfa_station_id"] = pd.to_numeric(flows["flow_station_id"], errors="coerce").astype("Int64")
    if "flow_source" not in flows:
        flows["flow_source"] = "nrfa"
    grouped = {
        (str(source), str(station)): value.set_index("date").sort_index()["flow_m3s"]
        for (source, station), value in flows.groupby(["flow_source", "flow_station_id"])
    }
    rows = []
    for row in biology.itertuples(index=False):
        base = {
            "bio_site_id": row.bio_site_id, "year": row.year,
            "flow_source": getattr(row, "flow_source", "nrfa"),
            "flow_station_id": getattr(row, "flow_station_id", getattr(row, "nrfa_station_id", None)),
            "flow_station_name": getattr(row, "flow_station_name", getattr(row, "nrfa_station_name", None)),
            "flow_river": getattr(row, "flow_river", getattr(row, "nrfa_river", None)),
            "nrfa_station_id": getattr(row, "nrfa_station_id", None),
            "nrfa_station_name": getattr(row, "nrfa_station_name", None),
            "nrfa_river": getattr(row, "nrfa_river", None),
            "gauge_network_distance_m": getattr(row, "gauge_network_distance_m", np.nan),
            "gauge_relationship": getattr(row, "gauge_relationship", None),
            "flow_selection_reason": getattr(row, "flow_selection_reason", None),
            "flow_quality_flags_30d": getattr(row, "flow_quality_flags_30d", None),
            "flow_station_distinct_count_for_site": getattr(row, "flow_station_distinct_count_for_site", pd.NA),
            "flow_station_changed_from_previous_selected_year": getattr(row, "flow_station_changed_from_previous_selected_year", pd.NA),
            "flow_previous_selected_year": getattr(row, "flow_previous_selected_year", pd.NA),
            "flow_previous_station_id": getattr(row, "flow_previous_station_id", pd.NA),
        }
        station_id = base["flow_station_id"]
        source = base["flow_source"]
        key = (str(source), str(station_id))
        if pd.isna(station_id) or pd.isna(row.survey_date) or key not in grouped:
            rows.append(base)
            continue
        series = grouped[key]
        survey_date = pd.Timestamp(row.survey_date).normalize()
        six_month_start = survey_date - pd.DateOffset(months=6)
        history = series.loc[six_month_start:survey_date]
        q95 = pd.to_numeric(pd.Series([getattr(row, "gdf_q95_flow", np.nan)]), errors="coerce").iloc[0]
        q10 = pd.to_numeric(pd.Series([getattr(row, "gdf_q10_flow", np.nan)]), errors="coerce").iloc[0]
        for days in (7, 30, 90):
            window = history.loc[survey_date - pd.Timedelta(days=days - 1):survey_date]
            base[f"flow_mean_{days}d_m3s"] = window.mean()
            base[f"flow_min_{days}d_m3s"] = window.min()
            base[f"flow_max_{days}d_m3s"] = window.max()
            base[f"flow_coverage_{days}d"] = window.notna().sum() / days
        six_month = history.dropna()
        expected_days = (survey_date - six_month_start).days + 1
        base["flow_mean_6mo_m3s"] = six_month.mean()
        base["flow_median_6mo_m3s"] = six_month.median()
        base["flow_min_6mo_m3s"] = six_month.min()
        base["flow_max_6mo_m3s"] = six_month.max()
        base["flow_std_6mo_m3s"] = six_month.std(ddof=1)
        base["flow_q05_6mo_m3s"] = six_month.quantile(0.05)
        base["flow_q95_6mo_m3s"] = six_month.quantile(0.95)
        base["flow_coverage_6mo"] = six_month.count() / expected_days
        base["flow_window_start"] = six_month_start
        base["flow_window_end"] = survey_date
        nearest = series.loc[survey_date - pd.Timedelta(days=3):survey_date + pd.Timedelta(days=3)]
        if len(nearest):
            nearest_date = (nearest.index.to_series() - survey_date).abs().idxmin()
            base["flow_on_or_near_survey_m3s"] = nearest.loc[nearest_date]
            base["flow_date_offset_days"] = int((nearest_date - survey_date).days)
        window30 = history.loc[survey_date - pd.Timedelta(days=29):survey_date]
        base["low_flow_days_30d"] = int((window30 <= q95).sum()) if pd.notna(q95) else np.nan
        base["high_flow_days_30d"] = int((window30 >= q10).sum()) if pd.notna(q10) else np.nan
        base["flow_relative_to_q95"] = base.get("flow_on_or_near_survey_m3s", np.nan) / q95 if pd.notna(q95) and q95 > 0 else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


# Gauge records are first matched to survey sites, then reduced to a small set
# of recent-flow measures that can be joined without duplicating survey rows.
def build_survey_antecedent_nrfa_features(
    surveys: pd.DataFrame, selected_gauges: pd.DataFrame, daily_flows: pd.DataFrame,
    *, record_id_column: str = "sample_id",
    reference_start: str | pd.Timestamp = "1974-05-20",
    reference_end: str | pd.Timestamp = "2019-03-29",
    minimum_reference_coverage: float = 0.5,
) -> pd.DataFrame:
    """Summarise NRFA daily flow over the six calendar months before each survey.

    This is the hydrological counterpart to the antecedent chemistry window. Gauge
    selection remains site-year specific, while the summary window is anchored
    to each individual macroinvertebrate survey date.
    """
    required = {record_id_column, "bio_site_id", "sample_date"}
    missing = required.difference(surveys)
    if missing:
        raise ValueError(f"Survey flow features require columns: {sorted(missing)}")
    survey = surveys.copy()
    survey["sample_date"] = pd.to_datetime(survey["sample_date"], errors="coerce")
    if "year" not in survey:
        survey["year"] = survey["sample_date"].dt.year
    survey["year"] = pd.to_numeric(survey["year"], errors="coerce").astype("Int64")
    selected = selected_gauges.copy()
    selected["year"] = pd.to_numeric(selected["year"], errors="coerce").astype("Int64")
    joined = survey.merge(selected, on=["bio_site_id", "year"], how="left", validate="many_to_one")

    flows = daily_flows.copy()
    flows["date"] = pd.to_datetime(flows["date"], errors="coerce")
    flows["flow_m3s"] = pd.to_numeric(flows["flow_m3s"], errors="coerce")
    if "flow_station_id" not in flows:
        flows["flow_station_id"] = flows["nrfa_station_id"].astype("Int64").astype("string")
    grouped = {
        str(station): group.set_index("date").sort_index()["flow_m3s"]
        for station, group in flows.groupby("flow_station_id")
    }

    rows: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        station_id = getattr(row, "flow_station_id", None)
        survey_date = pd.Timestamp(row.sample_date).normalize() if pd.notna(row.sample_date) else pd.NaT
        start = survey_date - pd.DateOffset(months=6) if pd.notna(survey_date) else pd.NaT
        result = {
            record_id_column: getattr(row, record_id_column),
            "nrfa_station_id": getattr(row, "nrfa_station_id", None),
            "nrfa_station_name": getattr(row, "nrfa_station_name", None),
            "nrfa_river": getattr(row, "nrfa_river", None),
            "nrfa_network_distance_m": getattr(row, "gauge_network_distance_m", np.nan),
            "nrfa_relationship": getattr(row, "gauge_relationship", None),
            "nrfa_selection_reason": getattr(row, "flow_selection_reason", None),
            "flow_station_distinct_count_for_site": getattr(row, "flow_station_distinct_count_for_site", pd.NA),
            "flow_station_changed_from_previous_selected_year": getattr(row, "flow_station_changed_from_previous_selected_year", pd.NA),
            "flow_previous_selected_year": getattr(row, "flow_previous_selected_year", pd.NA),
            "flow_previous_station_id": getattr(row, "flow_previous_station_id", pd.NA),
            "nrfa_window_start": start,
            "nrfa_window_end": survey_date,
        }
        series = grouped.get(str(station_id))
        if series is None or pd.isna(survey_date):
            rows.append(result)
            continue
        window = series.loc[start:survey_date].dropna()
        expected = (survey_date - start).days + 1
        result.update({
            "nrfa_flow_n_6mo": int(window.size),
            "nrfa_flow_coverage_6mo": window.size / expected,
            "nrfa_flow_min_6mo_m3s": window.min(),
            "nrfa_flow_max_6mo_m3s": window.max(),
            "nrfa_flow_mean_6mo_m3s": window.mean(),
            "nrfa_flow_std_6mo_m3s": window.std(ddof=1),
            "nrfa_flow_median_6mo_m3s": window.median(),
            "nrfa_flow_q05_6mo_m3s": window.quantile(0.05),
            "nrfa_flow_q95_6mo_m3s": window.quantile(0.95),
        })
        rows.append(result)
    result = pd.DataFrame(rows)
    if result.duplicated(record_id_column).any():
        raise ValueError(f"Survey-level NRFA features must contain one row per {record_id_column}")
    regime = build_flow_regime_features(
        joined[[record_id_column, "nrfa_station_id", "sample_date"]],
        flows,
        record_id_column=record_id_column,
        reference_start=reference_start,
        reference_end=reference_end,
        minimum_reference_coverage=minimum_reference_coverage,
    )
    return result.merge(regime, on=record_id_column, how="left", validate="one_to_one")
