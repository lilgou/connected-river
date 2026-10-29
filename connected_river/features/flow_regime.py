"""Fixed-reference and survey-window hydrological-regime features.

The method combines long-period gauge descriptors for 20 May 1974 to 29 March
2019 with threshold-event summaries for the six months preceding each
biological survey. This module derives both from NRFA gauged daily flow and
also exposes event duration in days, which is easier to interpret than the
normalised patch-length field.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FLOW_REFERENCE_START = pd.Timestamp("1974-05-20")
FLOW_REFERENCE_END = pd.Timestamp("2019-03-29")
FLOW_REFERENCE_EXPECTED_DAYS = (
    FLOW_REFERENCE_END - FLOW_REFERENCE_START
).days + 1
FLOW_REFERENCE_QUANTILES = (0.05, 0.25, 0.75, 0.95)


def flow_regime_value_columns(
    quantiles: Iterable[float] = FLOW_REFERENCE_QUANTILES,
) -> list[str]:
    """Return the expected value columns for the flow-regime table."""
    columns = [
        "flow_reference_rf_mean_poi", "flow_reference_rf_q50_poi",
        "flow_reference_rf_std_poi", "flow_reference_rf_cov_poi",
        "flow_reference_rf_q5_mean_poi", "flow_reference_rf_q95_mean_poi",
        "flow_iha_reference_start", "flow_iha_reference_end",
        "flow_iha_reference_expected_days", "flow_iha_reference_observed_days",
        "flow_iha_reference_coverage", "flow_iha_reference_minimum_coverage",
        "flow_iha_reference_eligible", "flow_iha_reference_reason",
        "flow_iha_threshold_direction", "flow_iha_window_start_exclusive",
        "flow_iha_window_end_inclusive", "flow_iha_window_expected_days",
        "flow_iha_window_observed_days", "flow_iha_window_coverage",
        "flow_iha_seasonal_reference_coverage",
        "flow_iha_survey_mean_to_poi_mean_ratio",
    ]
    for quantile in quantiles:
        label = str(round(float(quantile) * 100))
        for suffix in ("prec", "poi"):
            columns.extend([
                f"flow_reference_rf_{label}_quantile_{suffix}",
                f"flow_reference_rf_{label}_threshold_{suffix}",
                f"flow_reference_rf_{label}_duration_exceeded_{suffix}",
                f"flow_reference_rf_{label}_n_exceedences_{suffix}",
                f"flow_reference_rf_{label}_avg_patch_length_{suffix}",
                f"flow_iha_q{label}_mean_event_duration_days_{suffix}",
                f"flow_iha_q{label}_extreme_days_{suffix}",
                f"flow_iha_q{label}_event_count_{suffix}",
            ])
    return columns


def _safe_anniversary(value: pd.Timestamp, year: int) -> pd.Timestamp:
    """Move a month/day into ``year``, treating 29 February as 28 February."""
    try:
        return value.replace(year=year)
    except ValueError:
        return value.replace(year=year, day=28)


def _six_month_index(end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return the preceding window: (date - 6 months, date]."""
    start = end - pd.DateOffset(months=6)
    return pd.date_range(start + pd.Timedelta(days=1), end, freq="D")


# Count sustained high- and low-flow runs rather than individual days. Several
# adjacent threshold exceedances usually describe one hydrological event.
def _event_summary(
    daily: pd.Series,
    threshold: float,
    quantile: float,
) -> dict[str, float]:
    """Summarise low/high-flow events using the documented IHA direction.

    Missing daily flow is not treated as normal flow: it contributes to the
    expected-day denominator and breaks an event.  Low-flow events are values
    strictly below the 5th/25th percentile thresholds; high-flow events are
    values strictly above the 75th/95th percentile thresholds.

    ``legacy_avg_patch_length`` retains the established twice-normalised
    duration for continuity, although it has no intuitive unit. The corrected
    ``mean_event_duration_days`` is therefore released alongside it.
    """
    expected_days = int(daily.size)
    observed_days = int(daily.notna().sum())
    if expected_days == 0 or observed_days == 0 or pd.isna(threshold):
        return {
            "duration_exceeded": np.nan,
            "n_exceedences": np.nan,
            "legacy_avg_patch_length": np.nan,
            "mean_event_duration_days": np.nan,
            "extreme_days": np.nan,
            "event_count": np.nan,
        }
    if quantile < 0.5:
        extreme = daily.lt(threshold)
    else:
        extreme = daily.gt(threshold)
    # Missing observations explicitly break a run of consecutive exceedances.
    extreme = extreme.where(daily.notna(), False).astype(bool)
    starts = extreme & ~extreme.shift(1, fill_value=False)
    extreme_days = int(extreme.sum())
    event_count = int(starts.sum())
    duration = extreme_days / expected_days
    frequency = event_count / expected_days
    mean_duration = extreme_days / event_count if event_count else np.nan
    # This matches the numerical scale of the deposited field.  It is retained
    # for historical comparability only; use mean_event_duration_days for new
    # scientific interpretation.
    legacy_patch = mean_duration / (expected_days**2) if event_count else np.nan
    return {
        "duration_exceeded": float(duration),
        "n_exceedences": float(frequency),
        "legacy_avg_patch_length": float(legacy_patch) if pd.notna(legacy_patch) else np.nan,
        "mean_event_duration_days": float(mean_duration) if pd.notna(mean_duration) else np.nan,
        "extreme_days": float(extreme_days),
        "event_count": float(event_count),
    }


# Build expected values by day of year from the historical gauge record. The
# leap-day handling keeps dates aligned between leap and non-leap years.
def _seasonal_reference(
    reference: pd.Series,
    survey_date: pd.Timestamp,
    *,
    reference_start: pd.Timestamp,
    reference_end: pd.Timestamp,
) -> pd.Series:
    """Pool the same antecedent calendar season across the fixed POI."""
    indices: list[pd.DatetimeIndex] = []
    for year in range(reference_start.year, reference_end.year + 1):
        end = _safe_anniversary(survey_date, year)
        index = _six_month_index(end)
        index = index[(index >= reference_start) & (index <= reference_end)]
        if len(index):
            indices.append(index)
    if not indices:
        return pd.Series(dtype="float64")
    # One vectorised index lookup is substantially faster than reindexing the
    # 45-year reference once per year for every observed survey month/day.
    pooled_index = pd.DatetimeIndex(np.concatenate([index.values for index in indices]))
    return reference.reindex(pooled_index).reset_index(drop=True)


def _reference_statistics(reference: pd.Series) -> dict[str, float]:
    valid = pd.to_numeric(reference, errors="coerce").dropna()
    if valid.empty:
        return {}
    mean = valid.mean()
    std = valid.std(ddof=1)
    return {
        "flow_reference_rf_mean_poi": float(mean),
        "flow_reference_rf_q50_poi": float(valid.quantile(0.50)),
        "flow_reference_rf_std_poi": float(std),
        "flow_reference_rf_cov_poi": float(std / mean) if mean else np.nan,
        # Hydrological Q5 is exceeded 5% of the time (statistical 95th
        # percentile); Q95 is the low flow exceeded 95% of the time.
        "flow_reference_rf_q5_mean_poi": float(valid.quantile(0.95) / mean) if mean else np.nan,
        "flow_reference_rf_q95_mean_poi": float(valid.quantile(0.05) / mean) if mean else np.nan,
    }


# Compare the six months before each survey with the gauge's own longer-term
# behaviour, which is more useful than comparing unlike rivers on raw flow alone.
def build_flow_regime_features(
    surveys: pd.DataFrame,
    daily_flows: pd.DataFrame,
    *,
    record_id_column: str = "integration_record_id",
    station_id_column: str = "nrfa_station_id",
    date_column: str = "sample_date",
    reference_start: str | pd.Timestamp = FLOW_REFERENCE_START,
    reference_end: str | pd.Timestamp = FLOW_REFERENCE_END,
    minimum_reference_coverage: float = 0.5,
    quantiles: Iterable[float] = FLOW_REFERENCE_QUANTILES,
) -> pd.DataFrame:
    """Build long-period and survey-specific flow-regime features.

    The function returns one row for every input survey, including surveys
    without a selected gauge or without an eligible reference record.  The
    reference-period fields use a ``flow_reference_`` namespace.
    Corrected event duration, raw event counts and reference diagnostics use a
    ``flow_iha_`` namespace.
    """
    required = {record_id_column, station_id_column, date_column}
    missing = required.difference(surveys)
    if missing:
        raise ValueError(f"Flow-regime surveys require columns: {sorted(missing)}")
    required_flow = {station_id_column, "date", "flow_m3s"}
    missing_flow = required_flow.difference(daily_flows)
    if missing_flow:
        raise ValueError(f"Daily flow table requires columns: {sorted(missing_flow)}")

    reference_start = pd.Timestamp(reference_start).normalize()
    reference_end = pd.Timestamp(reference_end).normalize()
    if reference_end < reference_start:
        raise ValueError("Flow reference end must not precede its start")
    expected_reference_days = (reference_end - reference_start).days + 1
    quantile_values = tuple(float(value) for value in quantiles)
    if any(value <= 0 or value >= 1 for value in quantile_values):
        raise ValueError("Flow quantiles must lie strictly between zero and one")

    survey = surveys[[record_id_column, station_id_column, date_column]].copy()
    survey[date_column] = pd.to_datetime(survey[date_column], errors="coerce").dt.normalize()
    survey[station_id_column] = pd.to_numeric(survey[station_id_column], errors="coerce").astype("Int64")
    flows = daily_flows[[station_id_column, "date", "flow_m3s"]].copy()
    flows[station_id_column] = pd.to_numeric(flows[station_id_column], errors="coerce").astype("Int64")
    flows["date"] = pd.to_datetime(flows["date"], errors="coerce").dt.normalize()
    flows["flow_m3s"] = pd.to_numeric(flows["flow_m3s"], errors="coerce")
    flows = flows.dropna(subset=[station_id_column, "date"]).drop_duplicates(
        [station_id_column, "date"], keep="last",
    )

    survey_by_station = {
        int(station): group for station, group in survey.dropna(subset=[station_id_column]).groupby(station_id_column)
    }
    flow_by_station = {
        int(station): group.set_index("date").sort_index()["flow_m3s"]
        for station, group in flows.groupby(station_id_column, sort=False)
    }
    results: dict[object, dict[str, object]] = {
        row[record_id_column]: {record_id_column: row[record_id_column]}
        for row in survey.to_dict("records")
    }
    for row in survey[survey[station_id_column].isna()].to_dict("records"):
        results[row[record_id_column]].update({
            "flow_iha_reference_eligible": False,
            "flow_iha_reference_reason": "missing_selected_nrfa_gauge",
        })

    for station_id, station_surveys in survey_by_station.items():
        series = flow_by_station.get(station_id)
        if series is None:
            for row in station_surveys.to_dict("records"):
                results[row[record_id_column]].update({
                    "flow_iha_reference_eligible": False,
                    "flow_iha_reference_reason": "no_nrfa_daily_record",
                })
            continue
        reference_index = pd.date_range(reference_start, reference_end, freq="D")
        reference = series.reindex(reference_index)
        reference_observed = int(reference.notna().sum())
        reference_coverage = reference_observed / expected_reference_days
        eligible = bool(reference_coverage >= minimum_reference_coverage)
        reference_stats = _reference_statistics(reference) if eligible else {}
        seasonal_cache: dict[str, tuple[pd.Series, float]] = {}

        for row in station_surveys.to_dict("records"):
            record_id = row[record_id_column]
            survey_date = row[date_column]
            base = results[record_id]
            base.update({
                "flow_iha_reference_start": reference_start,
                "flow_iha_reference_end": reference_end,
                "flow_iha_reference_expected_days": expected_reference_days,
                "flow_iha_reference_observed_days": reference_observed,
                "flow_iha_reference_coverage": float(reference_coverage),
                "flow_iha_reference_minimum_coverage": float(minimum_reference_coverage),
                "flow_iha_reference_eligible": eligible,
                "flow_iha_reference_reason": (
                    "eligible_fixed_1974_2019_reference"
                    if eligible else "below_50_percent_reference_coverage"
                ),
                "flow_iha_threshold_direction": "below_q05_q25;above_q75_q95",
            })
            if pd.isna(survey_date):
                base["flow_iha_reference_reason"] = "missing_survey_date"
                continue
            window_index = _six_month_index(pd.Timestamp(survey_date))
            window = series.reindex(window_index)
            base.update({
                "flow_iha_window_start_exclusive": pd.Timestamp(survey_date) - pd.DateOffset(months=6),
                "flow_iha_window_end_inclusive": pd.Timestamp(survey_date),
                "flow_iha_window_expected_days": int(window.size),
                "flow_iha_window_observed_days": int(window.notna().sum()),
                "flow_iha_window_coverage": float(window.notna().mean()) if len(window) else np.nan,
            })
            if not eligible:
                continue
            base.update(reference_stats)
            month_day = pd.Timestamp(survey_date).strftime("%m-%d")
            if month_day not in seasonal_cache:
                seasonal = _seasonal_reference(
                    reference,
                    pd.Timestamp(survey_date),
                    reference_start=reference_start,
                    reference_end=reference_end,
                )
                expected = int(seasonal.size)
                coverage = float(seasonal.notna().mean()) if expected else np.nan
                seasonal_cache[month_day] = (seasonal, coverage)
            seasonal, seasonal_coverage = seasonal_cache[month_day]
            base["flow_iha_seasonal_reference_coverage"] = seasonal_coverage

            valid_window = window.dropna()
            valid_seasonal = seasonal.dropna()
            for quantile in quantile_values:
                label = str(round(quantile * 100))
                threshold_prec = valid_window.quantile(quantile) if len(valid_window) else np.nan
                threshold_poi = valid_seasonal.quantile(quantile) if len(valid_seasonal) else np.nan
                base[f"flow_reference_rf_{label}_quantile_prec"] = quantile
                base[f"flow_reference_rf_{label}_threshold_prec"] = threshold_prec
                base[f"flow_reference_rf_{label}_quantile_poi"] = quantile
                base[f"flow_reference_rf_{label}_threshold_poi"] = threshold_poi
                for suffix, threshold in (("prec", threshold_prec), ("poi", threshold_poi)):
                    event = _event_summary(window, threshold, quantile)
                    base[f"flow_reference_rf_{label}_duration_exceeded_{suffix}"] = event[
                        "duration_exceeded"
                    ]
                    base[f"flow_reference_rf_{label}_n_exceedences_{suffix}"] = event[
                        "n_exceedences"
                    ]
                    base[f"flow_reference_rf_{label}_avg_patch_length_{suffix}"] = event[
                        "legacy_avg_patch_length"
                    ]
                    base[f"flow_iha_q{label}_mean_event_duration_days_{suffix}"] = event[
                        "mean_event_duration_days"
                    ]
                    base[f"flow_iha_q{label}_extreme_days_{suffix}"] = event["extreme_days"]
                    base[f"flow_iha_q{label}_event_count_{suffix}"] = event["event_count"]

            mean_poi = base.get("flow_reference_rf_mean_poi")
            base["flow_iha_survey_mean_to_poi_mean_ratio"] = (
                float(valid_window.mean() / mean_poi)
                if len(valid_window) and pd.notna(mean_poi) and mean_poi != 0 else np.nan
            )

    result = pd.DataFrame(results.values())
    for column in flow_regime_value_columns(quantile_values):
        if column not in result:
            result[column] = pd.NA
    if result.duplicated(record_id_column).any():
        raise ValueError(f"Flow-regime features must contain one row per {record_id_column}")
    return survey[[record_id_column]].merge(result, on=record_id_column, how="left", validate="one_to_one")
