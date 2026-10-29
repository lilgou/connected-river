from __future__ import annotations

from pathlib import Path

import pandas as pd

from connected_river.chemistry_registry import (
    DETERMINAND_BY_LABEL,
    DETERMINAND_BY_NOTATION,
    additional_target_determinands,
)
from connected_river.utils.config import resolve_path
from connected_river.utils.io import ensure_dir

COLUMN_MAP = {
    "id": "observation_id",
    "samplingPoint.notation": "wq_site_id",
    "samplingPoint.prefLabel": "wq_site_name",
    "samplingPoint.easting": "easting",
    "samplingPoint.northing": "northing",
    "samplingPoint.region": "region",
    "samplingPoint.area": "area",
    "samplingPoint.subArea": "sub_area",
    "samplingPoint.samplingPointStatus": "site_status",
    "samplingPoint.samplingPointType": "site_type",
    "phenomenonTime": "sample_datetime",
    "samplingPurpose": "sampling_purpose",
    "sampleMaterialType": "sample_material",
    "determinand.notation": "determinand_id",
    "determinand.prefLabel": "determinand_name",
    "result": "result_raw",
    "unit": "unit",
}


DETERMINAND_ALIASES = {
    label: item.key
    for item in DETERMINAND_BY_NOTATION.values()
    for label in item.labels
}
DETERMINAND_ALIASES.update({
    "Nitrogen, Total Oxidised as N": "total_oxidised_n_mg_l",
    "Phosphorus, Total as P": "total_phosphorus_p_mg_l",
    "Temperature of Water": "water_temperature_c",
    "Conductivity at 25 C": "conductivity_25c",
    "Mercury, Dissolved": "mercury",
})


def classify_determinand(label: object, notation: object = None) -> str | None:
    """Return the stable feature key, preferring an exact EA notation."""
    if not pd.isna(notation):
        notation_text = str(notation).strip()
        if notation_text.isdigit():
            notation_text = notation_text.zfill(4)
        registered = DETERMINAND_BY_NOTATION.get(notation_text)
        if registered is not None:
            return registered.key
    if pd.isna(label):
        return None
    text = str(label).strip()
    if text in DETERMINAND_ALIASES:
        return DETERMINAND_ALIASES[text]
    lower = text.casefold()
    registered = DETERMINAND_BY_LABEL.get(lower)
    if registered is not None:
        return registered.key
    # Metal form is scientifically material. The cited study's headline Zn/Cu
    # predictors are dissolved concentrations, so those retain the short key;
    # total/unspecified and bioavailable results remain separate sensitivity
    # variables instead of being pooled into one misleading distribution.
    for metal in ("arsenic", "cadmium", "chromium", "copper", "lead", "mercury", "nickel", "zinc"):
        if lower == f"{metal}, dissolved":
            return metal
        if lower == metal:
            return f"{metal}_total"
        if lower == f"{metal}, bioavailable":
            return f"{metal}_bioavailable"
    patterns = [
        ("orthophosphate_p_mg_l", "orthophosphate"), ("total_phosphorus_p_mg_l", "phosphorus, total"),
        ("ammoniacal_n_mg_l", "ammon"), ("nitrite_n_mg_l", "nitrite"), ("nitrate_n_mg_l", "nitrate"),
        ("dissolved_oxygen_percent", "oxygen, dissolved, %"), ("dissolved_oxygen_mg_l", "oxygen, dissolved"),
        ("bod_5day_atu_mg_l", "bod : 5 day atu"), ("suspended_solids_mg_l", "solids, suspended"),
        ("water_temperature_c", "temperature of water"), ("alkalinity_caco3_mg_l", "alkalinity to ph 4.5"),
        ("conductivity_25c", "conductivity"),
    ]
    if lower == "ph":
        return "ph"
    return next((key for key, token in patterns if token in lower), None)


# Results sometimes contain qualifiers such as "<" alongside the number. Keep
# the qualifier rather than silently turning it into an ordinary measurement.
def parse_result(value: object) -> tuple[float | None, str | None]:
    if pd.isna(value):
        return None, None

    s = str(value).strip()
    qualifier = None

    if s.startswith("<"):
        qualifier = "<"
        s = s[1:].strip()
    elif s.startswith(">"):
        qualifier = ">"
        s = s[1:].strip()

    try:
        return float(s), qualifier
    except ValueError:
        return None, "non_numeric"


# Bring archive variants into one sample layout, standardise units where a safe
# conversion is known, and retain the original result fields for checking.
def harmonise_wq_observations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP).copy()

    keep = [v for v in COLUMN_MAP.values() if v in df.columns]
    df = df[keep]

    text_columns = [
        "observation_id", "wq_site_id", "wq_site_name", "region", "area", "sub_area",
        "site_status", "site_type", "sampling_purpose", "sample_material",
        "determinand_id", "determinand_name", "result_raw", "unit",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string")

    df["sample_datetime"] = pd.to_datetime(df["sample_datetime"], errors="coerce")
    df["sample_date"] = df["sample_datetime"].dt.date
    df["year"] = df["sample_datetime"].dt.year
    df["month"] = df["sample_datetime"].dt.month

    parsed = df["result_raw"].apply(parse_result)
    df["result"] = parsed.apply(lambda x: x[0])
    df["result_qualifier"] = parsed.apply(lambda x: x[1])
    # Preserve the reported limit while exposing the sensitivity treatments
    # described in the cited supporting information. The strict compatible
    # product uses LOQ/2; the alternatives remain available for sensitivity
    # analysis rather than silently overwriting the publisher value.
    df["result_loq_half"] = df["result"]
    df.loc[df["result_qualifier"].eq("<"), "result_loq_half"] *= 0.5
    df["result_below_loq_zero"] = df["result"]
    df.loc[df["result_qualifier"].eq("<"), "result_below_loq_zero"] = 0.0

    df["easting"] = pd.to_numeric(df["easting"], errors="coerce")
    df["northing"] = pd.to_numeric(df["northing"], errors="coerce")

    label_keys = df["determinand_name"].map(classify_determinand)
    if "determinand_id" in df:
        notation = df["determinand_id"].astype("string").str.strip().str.zfill(4)
        notation_items = notation.map(lambda value: DETERMINAND_BY_NOTATION.get(str(value)))
        notation_keys = notation_items.map(lambda item: getattr(item, "key", None))
        df["determinand_key"] = notation_keys.combine_first(label_keys)
        # Exact target-determinand comparability is deliberately separate from the broad
        # Target label classifier. A populated non-Table-A.1 notation must not
        # acquire parity merely because another code shares its label.
        missing_notation = df["determinand_id"].isna() | df["determinand_id"].str.strip().eq("")
        label_items = df["determinand_name"].astype("string").str.strip().str.casefold().map(
            DETERMINAND_BY_LABEL,
        )
        exact_items = notation_items.where(~missing_notation, label_items)
        df["target_determinand_key"] = exact_items.map(
            lambda item: getattr(item, "key", None),
        )
    else:
        df["determinand_key"] = label_keys
        label_items = df["determinand_name"].astype("string").str.strip().str.casefold().map(
            DETERMINAND_BY_LABEL,
        )
        df["target_determinand_key"] = label_items.map(
            lambda item: getattr(item, "key", None),
        )
    df["is_target_determinand"] = df["determinand_key"].notna()
    # Matching support uses the core chemistry scope. The 18 additional target
    # fields are released after station selection and do
    # not make a specialist station rank more highly.
    additional_keys = {item.key for item in additional_target_determinands()}
    df["is_matching_determinand"] = (
        df["is_target_determinand"]
        & ~df["determinand_key"].isin(additional_keys)
    )

    units = df["unit"].astype("string").str.upper()
    df["result_standardized"] = df["result"]
    df["result_standardized_loq_half"] = df["result_loq_half"]
    df["result_standardized_below_loq_zero"] = df["result_below_loq_zero"]
    df["unit_standardized"] = df["unit"]
    micrograms = units.str.contains("MICROGRAM PER LITRE", na=False)
    nanograms = units.str.contains("NANOGRAM PER LITRE", na=False)
    df.loc[micrograms, "result_standardized"] = df.loc[micrograms, "result"] / 1000.0
    df.loc[micrograms, "result_standardized_loq_half"] = df.loc[micrograms, "result_loq_half"] / 1000.0
    df.loc[micrograms, "result_standardized_below_loq_zero"] = df.loc[micrograms, "result_below_loq_zero"] / 1000.0
    df.loc[micrograms, "unit_standardized"] = "MILLIGRAM PER LITRE"
    df.loc[nanograms, "result_standardized"] = df.loc[nanograms, "result"] / 1_000_000.0
    df.loc[nanograms, "result_standardized_loq_half"] = df.loc[nanograms, "result_loq_half"] / 1_000_000.0
    df.loc[nanograms, "result_standardized_below_loq_zero"] = df.loc[nanograms, "result_below_loq_zero"] / 1_000_000.0
    df.loc[nanograms, "unit_standardized"] = "MILLIGRAM PER LITRE"

    df["has_valid_coordinates"] = (
        df["easting"].between(0, 700000)
        & df["northing"].between(0, 1300000)
    )
    df["has_numeric_result"] = df["result"].notna()

    return df


def harmonise_wq_file(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = resolve_path(input_path)
    output_path = resolve_path(output_path)
    ensure_dir(output_path.parent)

    df = pd.read_parquet(input_path)
    out = harmonise_wq_observations(df)
    out.to_parquet(output_path, index=False)

    return output_path
