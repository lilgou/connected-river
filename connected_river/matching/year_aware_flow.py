"""Select an NRFA gauge separately for every biological site-year."""

from __future__ import annotations

import pandas as pd


def standardise_daily_flows(nrfa: pd.DataFrame) -> pd.DataFrame:
    """Normalise cached NRFA gauged daily-flow records for feature building."""
    if nrfa.empty:
        return pd.DataFrame(columns=["flow_source", "flow_station_id", "date", "flow_m3s", "quality_flag"])
    out = nrfa.copy()
    out["flow_source"] = "nrfa"
    out["flow_station_id"] = out["nrfa_station_id"].astype("Int64").astype("string")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["flow_m3s"] = pd.to_numeric(out["flow_m3s"], errors="coerce")
    if "quality_flag" not in out:
        out["quality_flag"] = "NRFA quality-controlled archive"
    out["quality_flag"] = out["quality_flag"].astype("string")
    return out.dropna(subset=["flow_station_id", "date"]).drop_duplicates(
        ["flow_station_id", "date"], keep="last",
    ).sort_values(["flow_station_id", "date"]).reset_index(drop=True)


# Choose a gauge separately for each year because record availability changes.
# This avoids treating a station as monitored in years when it has no data.
def select_site_year_nrfa_gauges(
    biology_site_year: pd.DataFrame,
    ranked_nrfa: pd.DataFrame,
    daily_flows: pd.DataFrame,
    *,
    minimum_30d_coverage: float = 0.7,
    max_candidates: int = 3,
) -> pd.DataFrame:
    """Select the best connected NRFA gauge with data for each site-year.

    Location candidates are ranked on the river network. Selection is repeated
    for each year because an otherwise suitable gauge may not cover that
    biological survey window. No other flow provider is substituted.
    """
    biology = biology_site_year[["bio_site_id", "year", "first_sample_date_in_year"]].copy()
    biology["survey_date"] = pd.to_datetime(biology["first_sample_date_in_year"], errors="coerce").dt.normalize()
    if ranked_nrfa.empty or "gauge_rank" not in ranked_nrfa:
        candidates = ranked_nrfa.iloc[0:0].copy()
    else:
        candidates = ranked_nrfa[ranked_nrfa["gauge_rank"].le(max_candidates)].copy()
    lookup = {
        str(station): group.set_index("date").sort_index()
        for station, group in daily_flows.groupby("flow_station_id", dropna=False)
    }
    by_site = {key: value for key, value in candidates.groupby("bio_site_id")}
    rows: list[dict[str, object]] = []

    for observation in biology.itertuples(index=False):
        base = {
            "bio_site_id": observation.bio_site_id,
            "year": int(observation.year),
            "flow_source": "nrfa",
        }
        site_candidates = by_site.get(observation.bio_site_id)
        if site_candidates is None or pd.isna(observation.survey_date):
            rows.append({**base, "flow_selection_reason": "no_connected_nrfa_candidate"})
            continue
        survey_date = pd.Timestamp(observation.survey_date)
        start = survey_date - pd.Timedelta(days=29)
        evaluated: list[dict[str, object]] = []
        for candidate in site_candidates.itertuples(index=False):
            station_id = str(candidate.flow_station_id)
            series = lookup.get(station_id)
            if series is None:
                valid, coverage = pd.DataFrame(), 0.0
                record_start = record_end = pd.NaT
            else:
                valid = series.loc[start:survey_date]
                coverage = valid["flow_m3s"].notna().sum() / 30
                record_start, record_end = series.index.min(), series.index.max()
            quality = sorted(set(valid.get("quality_flag", pd.Series(dtype="string")).dropna().astype(str)))
            evaluated.append({
                **base,
                **candidate._asdict(),
                "flow_coverage_30d_at_selection": float(coverage),
                "flow_record_start_date": record_start,
                "flow_record_end_date": record_end,
                "flow_quality_flags_30d": "; ".join(quality) if quality else None,
                "flow_coverage_adequate": bool(coverage >= minimum_30d_coverage),
                "flow_threshold_source": "NRFA period-of-record statistics",
            })
        available = [row for row in evaluated if row["flow_coverage_30d_at_selection"] > 0]
        if not available:
            rows.append({**base, "flow_selection_reason": "no_nrfa_observations_for_survey_year"})
            continue
        adequate = [row for row in available if row["flow_coverage_adequate"]]
        if adequate:
            chosen = min(adequate, key=lambda row: (row["gauge_network_distance_m"], row["gauge_rank"]))
            reason = "nrfa_window_available"
        else:
            chosen = min(available, key=lambda row: (-row["flow_coverage_30d_at_selection"], row["gauge_network_distance_m"]))
            reason = "best_partial_nrfa_window"
        chosen["flow_selection_reason"] = reason
        rows.append(chosen)

    selected = pd.DataFrame(rows)
    for column in (
        "flow_station_id", "flow_station_name", "flow_river", "nrfa_station_id",
        "gauge_network_distance_m", "gauge_relationship",
    ):
        if column not in selected:
            selected[column] = pd.NA
    selected["nrfa_station_id"] = pd.to_numeric(selected["nrfa_station_id"], errors="coerce").astype("Int64")
    selected = selected.sort_values(["bio_site_id", "year"]).reset_index(drop=True)

    # A station can legitimately change between years when the nearest candidate
    # does not cover the survey window. Expose that potential discontinuity so a
    # temporal analysis can filter it or include station identity explicitly.
    selected["flow_station_distinct_count_for_site"] = selected.groupby(
        "bio_site_id", dropna=False,
    )["flow_station_id"].transform(lambda values: values.dropna().nunique()).astype("Int64")
    selected["flow_station_changed_from_previous_selected_year"] = pd.Series(
        pd.NA, index=selected.index, dtype="boolean",
    )
    selected["flow_previous_selected_year"] = pd.Series(
        pd.NA, index=selected.index, dtype="Int64",
    )
    selected["flow_previous_station_id"] = pd.Series(
        pd.NA, index=selected.index, dtype="string",
    )
    for _, group in selected.groupby("bio_site_id", sort=False, dropna=False):
        previous_station: str | None = None
        previous_year: int | None = None
        for index in group.index:
            station = selected.at[index, "flow_station_id"]
            if pd.isna(station):
                continue
            station = str(station)
            if previous_station is not None:
                selected.at[index, "flow_station_changed_from_previous_selected_year"] = (
                    station != previous_station
                )
                selected.at[index, "flow_previous_selected_year"] = previous_year
                selected.at[index, "flow_previous_station_id"] = previous_station
            previous_station = station
            previous_year = int(selected.at[index, "year"])
    return selected
