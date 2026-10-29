"""Normalise official EDM storm-overflow annual returns across workbook layouts."""

from __future__ import annotations

import re
import zipfile
from datetime import datetime, time, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()


def _column(df: pd.DataFrame, *phrases: str) -> str | None:
    columns = {_clean(c).casefold(): c for c in df.columns}
    for phrase in phrases:
        for cleaned, original in columns.items():
            if phrase.casefold() in cleaned:
                return original
    return None


# EDM spreadsheets do not all start their table on the first row, so find the
# real heading row before trying to standardise the columns.
def _header_row(raw: pd.DataFrame) -> int | None:
    for index, row in raw.head(12).iterrows():
        text = " | ".join(_clean(v) for v in row.dropna())
        if "Water Company Name" in text and ("Site Name" in text or "Unique ID" in text):
            return int(index)
    return None


def _duration_hours(series: pd.Series) -> pd.Series:
    def parse(value: object) -> float | None:
        if pd.isna(value):
            return None
        if isinstance(value, time):
            return value.hour + value.minute / 60 + value.second / 3600
        if isinstance(value, datetime):
            return value.hour + value.minute / 60 + value.second / 3600
        if isinstance(value, timedelta):
            return value.total_seconds() / 3600
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        try:
            parsed_duration = pd.to_timedelta(text)
        except (TypeError, ValueError):
            parsed_duration = pd.NaT
        if pd.notna(parsed_duration):
            return float(parsed_duration.total_seconds() / 3600)
        if ":" in text:
            try:
                parts = [float(part) for part in text.split(":")]
                if len(parts) == 3:
                    return parts[0] + parts[1] / 60 + parts[2] / 3600
            except ValueError:
                return None
        try:
            return float(text)
        except ValueError:
            return None
    return series.map(parse).astype(float)


def _identity_text(series: pd.Series) -> pd.Series:
    """Normalise publisher identifiers without turning missing values into 'nan'."""
    return (
        series.astype("string")
        .fillna("")
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


def _clean_ngr(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .replace("", pd.NA)
    )


def _operational_percent(series: pd.Series) -> pd.Series:
    """Normalise publisher fractions and percentages to a 0-100 scale.

    Annual workbooks have mixed Excel percentage fractions (for example 0.99)
    and already-scaled values (99). Values outside the physical range are
    retained as missing rather than clipped into a plausible-looking result.
    """
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(values.ge(0))
    values = values.where(values.le(100))
    return values.where(values.gt(1), values * 100)


def _normalise_sheet(
    data: bytes,
    sheet: str,
    year: int,
    source: str,
    source_workbook: str | None = None,
) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(data), sheet_name=sheet, header=None, nrows=12)
    header = _header_row(raw)
    if header is None:
        return pd.DataFrame()
    df = pd.read_excel(BytesIO(data), sheet_name=sheet, header=header)
    df.columns = [_clean(c) for c in df.columns]
    company = _column(df, "Water Company Name")
    site = _column(df, "Site Name (EA Consents", "Site Name")
    permit = _column(df, "EA Permit Reference", "Permit No.", "Permit Number")
    activity = _column(df, "Activity Reference")
    unique = _column(df, "Unique ID")
    ngr = _column(df, "Outlet Discharge NGR")
    waterbody = _column(df, "WFD Waterbody ID")
    catchment = _column(df, "WFD Waterbody Catchment Name")
    receiving = _column(df, "Receiving Water / Environment")
    spills = _column(df, "Counted spills using 12-24h")
    duration = _column(df, "Total Duration")
    operation = _column(df, "% of reporting period EDM operational")
    out = pd.DataFrame(index=df.index)
    out["reporting_year"] = year
    out["asset_id"] = df[unique].astype("string").str.strip() if unique else pd.Series(pd.NA, index=df.index, dtype="string")
    out["water_company"] = df[company] if company else pd.NA
    out["site_name"] = df[site] if site else pd.NA
    out["permit_reference"] = df[permit] if permit else pd.NA
    out["activity_reference"] = df[activity] if activity else pd.NA
    out["outlet_ngr"] = df[ngr] if ngr else pd.NA
    out["wfd_waterbody_id"] = df[waterbody] if waterbody else pd.NA
    out["wfd_catchment"] = df[catchment] if catchment else pd.NA
    out["receiving_water"] = df[receiving] if receiving else pd.NA
    out["counted_spills"] = pd.to_numeric(df[spills], errors="coerce") if spills else pd.NA
    out["spill_duration_hours"] = _duration_hours(df[duration]) if duration else pd.NA
    out["edm_operational_pct"] = (
        _operational_percent(df[operation]) if operation
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )
    out["source_archive"] = source
    out["source_workbook"] = source_workbook or source
    out["source_sheet"] = sheet
    for column in [
        "asset_id", "water_company", "site_name", "permit_reference", "outlet_ngr",
        "activity_reference", "wfd_waterbody_id", "wfd_catchment", "receiving_water",
        "source_archive", "source_workbook", "source_sheet",
    ]:
        out[column] = out[column].astype("string").str.strip()
    identity = out["asset_id"].notna() | out["permit_reference"].notna() | out["site_name"].notna()
    out = out[identity & out["water_company"].notna()].copy()
    fallback = (
        "historic:"
        + _identity_text(out["water_company"])
        + "|"
        + _identity_text(out["permit_reference"])
        + "|"
        + _identity_text(out["activity_reference"])
        + "|"
        + _identity_text(out["site_name"])
    )
    missing_asset_id = out["asset_id"].isna() | out["asset_id"].str.strip().eq("")
    out["asset_id_source"] = "publisher_unique_id"
    out.loc[missing_asset_id, "asset_id"] = fallback.loc[missing_asset_id]
    out.loc[missing_asset_id, "asset_id_source"] = "constructed_publisher_identity"
    return out


def _unique_location_lookup(
    reference: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> dict[tuple[str, ...], tuple[str, int, object]]:
    eligible = reference.copy()
    for column in key_columns:
        eligible = eligible[eligible[column].ne("")]
    if eligible.empty:
        return {}
    lookup: dict[tuple[str, ...], tuple[str, int, object]] = {}
    for key, group in eligible.groupby(list(key_columns), dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        locations = sorted(set(group["_clean_outlet_ngr"].dropna().astype(str)))
        if len(locations) != 1:
            continue
        asset_ids = sorted(set(group["asset_id"].dropna().astype(str)))
        reference_asset: object = asset_ids[0] if len(asset_ids) == 1 else pd.NA
        lookup[tuple(str(value) for value in key_tuple)] = (
            locations[0],
            int(pd.to_numeric(group["reporting_year"], errors="coerce").min()),
            reference_asset,
        )
    return lookup


def _apply_location_crosswalk(edm: pd.DataFrame) -> pd.DataFrame:
    """Attach coordinates to older EDM rows using unique exact later-year keys.

    The 2020 return has spill evidence but no national outlet grid-reference
    field. It is retained in full. Spatial routing is allowed only where a
    company plus published identifier resolves to exactly one later publisher
    NGR; unmatched or ambiguous records remain explicitly non-spatial.
    """
    out = edm.copy()
    out["_company_key"] = _identity_text(out["water_company"])
    out["_permit_key"] = _identity_text(out["permit_reference"])
    out["_activity_key"] = _identity_text(out["activity_reference"])
    out["_site_key"] = _identity_text(out["site_name"])
    out["_clean_outlet_ngr"] = _clean_ngr(out["outlet_ngr"])

    direct = out["_clean_outlet_ngr"].notna()
    out["location_match_method"] = "not_spatially_locatable_from_published_fields"
    out["location_source_year"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["location_reference_asset_id"] = pd.Series(
        pd.NA, index=out.index, dtype="string"
    )
    out.loc[direct, "location_match_method"] = "publisher_reported_ngr"
    out.loc[direct, "location_source_year"] = pd.to_numeric(
        out.loc[direct, "reporting_year"], errors="coerce"
    ).astype("Int64")
    out.loc[direct, "location_reference_asset_id"] = out.loc[direct, "asset_id"]

    reference = out.loc[direct].copy()
    strategies = (
        (
            ("_company_key", "_permit_key", "_activity_key"),
            "exact_company_permit_activity_to_unique_later_ngr",
        ),
        (
            ("_company_key", "_permit_key"),
            "exact_company_permit_to_unique_later_ngr",
        ),
        (
            ("_company_key", "_site_key"),
            "exact_company_site_to_unique_later_ngr",
        ),
    )
    for key_columns, method in strategies:
        lookup = _unique_location_lookup(reference, key_columns)
        unresolved = out["_clean_outlet_ngr"].isna()
        for index in out.index[unresolved]:
            key = tuple(str(out.at[index, column]) for column in key_columns)
            match = lookup.get(key)
            if match is None:
                continue
            ngr, source_year, reference_asset = match
            out.at[index, "_clean_outlet_ngr"] = ngr
            out.at[index, "location_match_method"] = method
            out.at[index, "location_source_year"] = source_year
            out.at[index, "location_reference_asset_id"] = reference_asset

    out["outlet_ngr"] = out["_clean_outlet_ngr"]
    out["location_available_for_network"] = out["outlet_ngr"].notna()
    return out.drop(columns=[
        "_company_key", "_permit_key", "_activity_key", "_site_key",
        "_clean_outlet_ngr",
    ])


# Work through every available reporting year, but keep the year on each row so
# spill information is never accidentally treated as timeless site context.
def build_edm_index(raw_dir: str | Path, output_dir: str | Path, years: list[int] | tuple[int, ...]) -> dict[str, Path]:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    present_archives: set[int] = set()
    for year in years:
        archive_path = raw_dir / f"EDM_{year}_Storm_Overflow_Annual_Return.zip"
        if not archive_path.exists():
            continue
        present_archives.add(int(year))
        year_frames = []
        with zipfile.ZipFile(archive_path) as archive:
            workbooks = [n for n in archive.namelist() if n.lower().endswith(".xlsx") and "summary" not in n.lower()]
            for workbook in workbooks:
                data = archive.read(workbook)
                excel = pd.ExcelFile(BytesIO(data))
                sheets = ["All WaSC"] if "All WaSC" in excel.sheet_names else excel.sheet_names
                for sheet in sheets:
                    table = _normalise_sheet(
                        data,
                        sheet,
                        year,
                        archive_path.name,
                        source_workbook=workbook,
                    )
                    if not table.empty:
                        year_frames.append(table)
        year_df = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
        if not year_df.empty:
            frames.append(year_df)
    if not frames:
        raise RuntimeError(f"No EDM annual-return tables could be read from {raw_dir}")
    edm = pd.concat(frames, ignore_index=True)
    edm = _apply_location_crosswalk(edm)
    source_rows_by_year = edm.groupby("reporting_year", dropna=False).size().to_dict()
    duplicate_mask = edm.duplicated(["reporting_year", "asset_id"], keep="first")
    duplicate_rows_by_year = (
        edm.loc[duplicate_mask].groupby("reporting_year", dropna=False).size().to_dict()
    )
    edm = edm.drop_duplicates(["reporting_year", "asset_id"], keep="first")
    inventory = []
    for year in years:
        year_df = edm.loc[pd.to_numeric(edm["reporting_year"], errors="coerce").eq(year)]
        valid_assets = int(year_df["asset_id"].notna().sum())
        located_assets = int(year_df["location_available_for_network"].fillna(False).sum())
        if int(year) not in present_archives:
            status = "missing"
        elif valid_assets >= 100:
            status = "loaded"
        elif len(year_df):
            status = "incomplete"
        else:
            status = "no_recognised_tables"
        spatial_status = (
            "complete" if located_assets == valid_assets and valid_assets
            else "partial" if located_assets
            else "unavailable"
        )
        inventory.append({
            "year": year,
            "status": status,
            "rows": len(year_df),
            "source_rows_parsed": int(source_rows_by_year.get(year, 0)),
            "duplicate_asset_rows_removed": int(duplicate_rows_by_year.get(year, 0)),
            "valid_assets": valid_assets,
            "assets_with_network_location": located_assets,
            "network_location_coverage": (
                located_assets / valid_assets if valid_assets else None
            ),
            "spatial_status": spatial_status,
            "national_context_available": located_assets >= 100,
        })
    paths = {"asset_year": output_dir / "edm_asset_year.parquet", "inventory": output_dir / "edm_source_inventory.parquet"}
    edm.to_parquet(paths["asset_year"], index=False)
    pd.DataFrame(inventory).to_parquet(paths["inventory"], index=False)
    return paths
