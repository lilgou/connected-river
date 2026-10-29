"""Downloader/parser for EA EDM storm overflow annual returns."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from connected_river.download.http import (
    download_file,
    extract_zip,
    normalise_columns,
    session_with_retries,
)
from connected_river.utils.config import resolve_path
from connected_river.utils.io import ensure_dir
from connected_river.utils.manifest import write_manifest_entry

DATASET_PAGE = "https://www.data.gov.uk/dataset/19f6064d-7356-466f-844e-d20ea10ae9fd/event-duration-monitoring-storm-overflows-annual-returns"
FALLBACK_URL = "https://environment.data.gov.uk/api/file/download?fileName=EDM_{year}_Storm_Overflow_Annual_Return.zip"


def discover_edm_zip_url(year: int) -> str:
    """Discover the current EDM annual-return ZIP URL from data.gov.uk.

    The Defra file service embeds a fileDataSetId in current links. We scrape the
    public dataset page to avoid hardcoding IDs, and fall back to a filename-only
    URL if the page format changes.
    """
    try:
        with session_with_retries() as s:
            html = s.get(DATASET_PAGE, timeout=60).text
        pattern = rf'https://environment\.data\.gov\.uk/api/file/download\?[^"<> ]*EDM_{year}_Storm_Overflow_Annual_Return\.zip'
        match = re.search(pattern, html)
        if match:
            return match.group(0).replace("&amp;", "&")
    except Exception:
        pass
    return FALLBACK_URL.format(year=year)


def download_edm_year(
    year: int,
    *,
    raw_dir: str | Path = "data/raw/storm_overflows/edm",
    overwrite: bool = False,
    normalise: bool = True,
) -> dict[str, Path | list[Path]]:
    """Download an EDM annual-return ZIP and normalise spreadsheet sheets."""
    raw_dir = resolve_path(raw_dir)
    zip_path = raw_dir / f"EDM_{year}_Storm_Overflow_Annual_Return.zip"
    url = discover_edm_zip_url(year)
    zip_path = download_file(
        url,
        zip_path,
        overwrite=overwrite,
        dataset="ea_edm_storm_overflows",
        metadata={"year": year},
    )
    extracted_dir = raw_dir / str(year)
    extracted = extract_zip(zip_path, extracted_dir, overwrite=overwrite)
    result: dict[str, Path | list[Path]] = {"zip": zip_path, "extracted": extracted}
    if normalise:
        out_path = normalise_edm_year(year, extracted_dir=extracted_dir)
        result["normalised"] = out_path
    return result


def _looks_like_asset_sheet(df: pd.DataFrame) -> bool:
    joined = " ".join(str(c).lower() for c in df.columns)
    return any(term in joined for term in ["overflow", "spill", "duration", "edm", "asset"])


def normalise_edm_year(year: int, *, extracted_dir: str | Path) -> Path:
    """Convert EDM spreadsheets for a year into one Parquet table.

    The annual ZIPs include multiple company sheets and summary sheets. This
    parser keeps any sheet that looks asset/spill related and adds source
    workbook/sheet/year columns for traceability.
    """
    extracted_dir = Path(extracted_dir)
    rows: list[pd.DataFrame] = []
    for workbook in list(extracted_dir.rglob("*.xlsx")) + list(extracted_dir.rglob("*.xls")):
        try:
            sheets = pd.read_excel(workbook, sheet_name=None, dtype=str)
        except Exception:
            continue
        for sheet, df in sheets.items():
            if df.empty:
                continue
            df = normalise_columns(df.dropna(how="all"))
            if df.empty or not _looks_like_asset_sheet(df):
                continue
            df["edm_year"] = year
            df["source_workbook"] = workbook.name
            df["source_sheet"] = str(sheet)
            rows.append(df)
    out_dir = ensure_dir("data/interim/storm_overflows")
    out_path = out_dir / f"edm_storm_overflows_{year}.parquet"
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["edm_year"])
    out.to_parquet(out_path, index=False)
    write_manifest_entry(
        "outputs/logs/download_manifest.json",
        dataset="ea_edm_storm_overflows_normalised",
        source_url=DATASET_PAGE,
        output_path=out_path,
        metadata={"year": year, "rows": len(out), "columns": list(out.columns)},
    )
    return out_path


# EDM files are published by reporting year and can differ slightly in layout.
# Save the originals here and deal with those differences during ingestion.
def download_edm_years(years: Iterable[int], *, overwrite: bool = False) -> list[Path]:
    paths: list[Path] = []
    for year in years:
        result = download_edm_year(int(year), overwrite=overwrite)
        normalised = result.get("normalised")
        if isinstance(normalised, Path):
            paths.append(normalised)
    return paths
