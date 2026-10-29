"""Date-bounded River Habitat Survey context matching."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


# Convert grid references in one batch and leave failed values missing. Guessing
# a location would be worse than carrying an honest gap into the output.
def os_grid_reference_to_xy(values: pd.Series) -> pd.DataFrame:
    """Convert standard two-letter OS National Grid references to metres."""
    eastings: list[float] = []
    northings: list[float] = []
    for value in values.astype("string"):
        match = re.fullmatch(r"\s*([A-HJ-Z])([A-HJ-Z])\s*(\d{2,10})\s*", str(value).upper())
        if match is None or len(match.group(3)) % 2:
            eastings.append(np.nan)
            northings.append(np.nan)
            continue
        first = _LETTERS.index(match.group(1))
        second = _LETTERS.index(match.group(2))
        e100km = ((first - 2) % 5) * 5 + second % 5
        n100km = 19 - (first // 5) * 5 - second // 5
        if not (0 <= e100km <= 6 and 0 <= n100km <= 12):
            eastings.append(np.nan)
            northings.append(np.nan)
            continue
        digits = match.group(3)
        half = len(digits) // 2
        eastings.append(e100km * 100_000 + int(digits[:half]) * 10 ** (5 - half))
        northings.append(n100km * 100_000 + int(digits[half:]) * 10 ** (5 - half))
    return pd.DataFrame({"rhs_easting": eastings, "rhs_northing": northings}, index=values.index)


# For each survey, keep the nearest suitable habitat record in the allowed time
# range and record why rows without one could not be linked.
def build_habitat_context_ledger(
    specialist: pd.DataFrame,
    rhs: pd.DataFrame,
    *,
    max_distance_m: float = 1_000,
    nearest_candidates: int = 25,
) -> pd.DataFrame:
    """Match the nearest dated RHS survey available by each biology date.

    This is a Euclidean, date-bounded context match. It does not place the
    habitat survey at the biological site or assume that both were sampled on
    the same date.
    """
    required_product = {
        "integration_record_id", "data_origin", "sample_date", "sample_year",
        "assemblage", "easting", "northing",
    }
    missing = sorted(required_product - set(specialist))
    if missing:
        raise ValueError("Habitat context requires product columns: " + ", ".join(missing))
    required_rhs = {"SURVEY_ID", "SURVEY_DATE", "NGR_SITE"}
    if not required_rhs.issubset(rhs):
        raise ValueError("Habitat source requires SURVEY_ID, SURVEY_DATE and NGR_SITE")

    automated = specialist.loc[specialist["data_origin"].eq("connected_river_pipeline")].copy()
    source = rhs.copy()
    coordinates = os_grid_reference_to_xy(source["NGR_SITE"])
    source = pd.concat([source, coordinates], axis=1)
    source["rhs_survey_date"] = pd.to_datetime(source["SURVEY_DATE"], errors="coerce")
    source = source.dropna(subset=["rhs_easting", "rhs_northing", "rhs_survey_date"])
    if source.empty:
        raise ValueError("No dated, georeferenced River Habitat Survey rows are available")
    tree = cKDTree(source[["rhs_easting", "rhs_northing"]].to_numpy(float))
    points = automated[["easting", "northing"]].apply(pd.to_numeric, errors="coerce")
    valid_points = points.notna().all(axis=1)
    k = min(max(int(nearest_candidates), 1), len(source))
    distances = np.full((len(automated), k), np.nan)
    indices = np.full((len(automated), k), -1, dtype=int)
    if valid_points.any():
        queried_distance, queried_index = tree.query(points.loc[valid_points].to_numpy(float), k=k)
        queried_distance = np.atleast_2d(queried_distance)
        queried_index = np.atleast_2d(queried_index)
        if k == 1:
            queried_distance = queried_distance.T
            queried_index = queried_index.T
        positions = np.flatnonzero(valid_points.to_numpy())
        distances[positions] = queried_distance
        indices[positions] = queried_index

    chosen_source = np.full(len(automated), -1, dtype=int)
    chosen_distance = np.full(len(automated), np.nan)
    sample_dates = pd.to_datetime(automated["sample_date"], errors="coerce").to_numpy()
    source_dates = source["rhs_survey_date"].to_numpy()
    for row_position in range(len(automated)):
        if pd.isna(sample_dates[row_position]):
            continue
        for distance, source_position in zip(distances[row_position], indices[row_position]):
            if source_position < 0 or distance > max_distance_m:
                continue
            if source_dates[source_position] <= sample_dates[row_position]:
                chosen_source[row_position] = source_position
                chosen_distance[row_position] = distance
                break

    ledger = automated[[
        "integration_record_id", "assemblage", "sample_date", "sample_year",
        "site_id", "bio_site_id", "easting", "northing",
    ]].reset_index(drop=True)
    ledger["habitat_context_status"] = "source_unavailable"
    matched = chosen_source >= 0
    ledger.loc[matched, "habitat_context_status"] = "documented_method_replacement"
    ledger["rhs_match_method"] = "nearest_euclidean_prior_survey_within_1000m"
    ledger["rhs_match_distance_m"] = chosen_distance
    ledger["rhs_survey_id"] = pd.Series(pd.NA, index=ledger.index, dtype="string")
    ledger["rhs_survey_date"] = pd.NaT
    fields = {
        "HMS": "rhs_hms", "HMS_CLASS": "rhs_hms_class", "HQA": "rhs_hqa",
        "HQA_ADJUSTED": "rhs_hqa_adjusted",
        "HMS_RFORCED_BNK_BED_SUB_SCORE": "rhs_hms_reinforced_bank_bed_subscore",
        "CHANNEL_BANKFULL_WIDTH": "rhs_channel_bankfull_width_m",
        "CHANNEL_WATER_WIDTH": "rhs_channel_water_width_m",
        "CHANNEL_WATER_DEPTH": "rhs_channel_water_depth_m",
        "BED_MATERIAL_DESCRIPTION": "rhs_bed_material_description",
    }
    for target in fields.values():
        ledger[target] = pd.NA
    if matched.any():
        selected = source.iloc[chosen_source[matched]].reset_index(drop=True)
        matched_rows = ledger.index[matched]
        ledger.loc[matched_rows, "rhs_survey_id"] = selected["SURVEY_ID"].astype("string").to_numpy()
        ledger.loc[matched_rows, "rhs_survey_date"] = selected["rhs_survey_date"].to_numpy()
        for source_column, target in fields.items():
            if source_column in selected:
                ledger.loc[matched_rows, target] = selected[source_column].to_numpy()
    ledger["rhs_age_at_biology_days"] = (
        pd.to_datetime(ledger["sample_date"]) - pd.to_datetime(ledger["rhs_survey_date"])
    ).dt.days.astype("Int64")
    ledger["temporal_leakage_check"] = (
        pd.to_datetime(ledger["rhs_survey_date"]).le(pd.to_datetime(ledger["sample_date"]))
        | ledger["rhs_survey_date"].isna()
    ).astype("boolean")
    ledger["source_version"] = "EA national RHS open extract acquired by source manifest"
    ledger["comparability_warning"] = (
        "Euclidean prior-survey context, not an exact site-and-date habitat observation. "
        "Pre/post-2003 RHS methods and published HQA/HMS limitations require stratified use."
    )
    if not ledger["temporal_leakage_check"].all():
        raise AssertionError("A future RHS survey was assigned to an earlier biology survey")
    return ledger
