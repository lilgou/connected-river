"""Harmonise treatment works and consented sewage outlets from open extracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyproj import Transformer


# Standardise permit locations once here so the network step can concentrate on
# route relationships rather than source-file column names.
def build_discharge_context(input_path: str | Path, output_path: str | Path) -> Path:
    raw = pd.read_csv(input_path, low_memory=False)
    rename = {
        "PERMIT_NUMBER": "permit_reference", "DISCHARGE_SITE_NAME": "site_name",
        "COMPANY_NAME": "operator", "DSI_TYPE_DESCRIPTION": "site_type",
        "RECEIVING_WATER": "receiving_water", "REC_ENV_CODE_DESCRIPTION": "receiving_environment",
        "OUTLET_TYPE_DESCRIPTION": "outlet_type", "OUTLET_GRID_REF": "outlet_ngr",
        "OUTLET_NUMBER": "outlet_number", "EFFLUENT_NUMBER": "effluent_number",
        "STATUS_OF_PERMIT": "permit_status_raw", "STATUS_DESCRIPTION": "permit_status_description",
        "longitude": "longitude", "latitude": "latitude",
    }
    table = raw.rename(columns={key: value for key, value in rename.items() if key in raw}).copy()
    keep = list(dict.fromkeys(value for value in rename.values() if value in table))
    table = table[keep]
    table = table[table["receiving_environment"].astype("string").str.contains("Freshwater river", case=False, na=False)].copy()
    table["discharge_id"] = (
        table["permit_reference"].astype("string").fillna("unknown") + "|"
        + table.get("outlet_number", pd.Series(index=table.index, dtype="string")).astype("string").fillna("unknown") + "|"
        + table.get("effluent_number", pd.Series(index=table.index, dtype="string")).astype("string").fillna("unknown")
    )
    table = table.drop_duplicates("discharge_id")
    table["is_treatment_works"] = table["site_type"].astype("string").str.contains("WwTW|Treatment Works", case=False, na=False)
    table["is_water_company"] = table["outlet_type"].astype("string").str.contains("water company", case=False, na=False)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    longitude = pd.to_numeric(table["longitude"], errors="coerce")
    latitude = pd.to_numeric(table["latitude"], errors="coerce")
    easting, northing = transformer.transform(longitude.to_numpy(), latitude.to_numpy())
    table["easting"], table["northing"] = easting, northing
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    return output_path
