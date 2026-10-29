"""Ingest the national River Habitat Survey summary workbook."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd


# Keep habitat observations as dated evidence. They are useful context, but are
# not assumed to describe a reach indefinitely.
def build_river_habitat_index(zip_path: str | Path, output_path: str | Path) -> Path:
    zip_path, output_path = Path(zip_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".xlsx"))
        table = pd.read_excel(BytesIO(archive.read(member)), sheet_name="Data")
    table["SURVEY_DATE"] = pd.to_datetime(table["SURVEY_DATE"], dayfirst=True, errors="coerce")
    table["SURVEY_YEAR"] = pd.to_numeric(table["SURVEY_YEAR"], errors="coerce").astype("Int64")
    table.to_parquet(output_path, index=False)
    return output_path
