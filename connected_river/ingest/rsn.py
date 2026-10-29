"""Ingest the NCEA River Surveillance Network analysis-ready dataset."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd


def _slug(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def _read_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as stream:
        return pd.read_csv(stream, low_memory=False)


# Bring river surveillance records into one site-year layout before they are
# joined to the other monitoring programmes.
def build_rsn_index(zip_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    zip_path, output_dir = Path(zip_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}
    inventory = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            name = _slug(Path(member).stem.removeprefix("RSN "))
            table = _read_csv(archive, member)
            tables[name] = table
            path = output_dir / f"rsn_{name}.parquet"
            table.to_parquet(path, index=False)
            paths[name] = path
            inventory.append({"table": name, "rows": len(table), "columns": len(table.columns), "source_member": member})

    sites = tables["sites"].copy()
    sites = sites[sites["waterbody_cat"].astype("string").str.casefold().eq("river")].copy()
    sites["site_id"] = sites["network_id"].astype("string")

    wq = tables["water_quality_samples"].copy()
    wq["sample_datetime"] = pd.to_datetime(wq["sample_datetime"], errors="coerce")
    wq["year"] = wq["sample_datetime"].dt.year.astype("Int64")
    wq["meas_result"] = pd.to_numeric(wq["meas_result"], errors="coerce")
    wq_summary = wq.groupby(["network_id", "year"], dropna=False).agg(
        wq_observations=("det_code", "size"), wq_numeric_observations=("meas_result", "count"),
        wq_samples=("sample_id", "nunique"), wq_sampling_dates=("sample_datetime", lambda s: s.dt.date.nunique()),
        wq_determinands=("det_code", "nunique"), first_wq_date=("sample_datetime", "min"), last_wq_date=("sample_datetime", "max"),
    ).reset_index()
    names = wq["det_desc"].astype("string").str.casefold()
    for key, pattern in {
        "nutrients": r"nitrate|nitrite|ammon|phosph|orthophosphate",
        "oxygen_bod": r"oxygen|bod|biochemical oxygen",
        "metals": r"zinc|copper|lead|cadmium|nickel|chromium|mercury|iron|aluminium|manganese",
        "pesticides_pfas": r"pestic|herbicide|insecticide|fungicide|pfas|fluoro",
    }.items():
        counts = wq[names.str.contains(pattern, regex=True, na=False)].groupby(["network_id", "year"])["det_code"].nunique().rename(f"{key}_determinands").reset_index()
        wq_summary = wq_summary.merge(counts, on=["network_id", "year"], how="left")

    evidence = wq_summary
    biology_specs = {
        "macroinvertebrate_samples": ("macroinvertebrate", ["whpt_ntaxa", "whpt_aspt", "life_family_index", "psi_family_score"]),
        "diatom_samples": ("diatom", ["percent_motile", "percent_organic_tolerant", "percent_saline"]),
        "macrophyte_samples": ("macrophyte", ["rmni", "rn_a_taxa", "n_rfg", "rfa_pc"]),
    }
    biology_long = []
    for table_name, (assemblage, metrics) in biology_specs.items():
        table = tables[table_name].copy()
        table["sample_datetime"] = pd.to_datetime(table["sample_datetime"], errors="coerce")
        table["year"] = table["sample_datetime"].dt.year.astype("Int64")
        aggregations = {"biology_samples": ("sample_id", "nunique")}
        aggregations.update({metric: (metric, "mean") for metric in metrics if metric in table})
        summary = table.groupby(["network_id", "year"], dropna=False).agg(**aggregations).reset_index()
        summary["assemblage"] = assemblage
        biology_long.append(summary)
        wide = summary.drop(columns="assemblage").rename(columns={c: f"{assemblage}_{c}" for c in summary.columns if c not in {"network_id", "year", "assemblage"}})
        evidence = evidence.merge(wide, on=["network_id", "year"], how="outer")

    habitat = tables["river_habitat_surveys"].copy()
    habitat["survey_date"] = pd.to_datetime(habitat["survey_date"], errors="coerce")
    habitat["year"] = habitat["survey_date"].dt.year.astype("Int64")
    habitat_summary = habitat.groupby(["network_id", "year"], dropna=False).agg(
        habitat_surveys=("survey_id", "nunique"), hqa_score=("hqa_score", "mean"),
        hms_score=("hms_score", "mean"), habitat_completeness=("completeness_score", "mean"),
    ).reset_index()
    evidence = evidence.merge(habitat_summary, on=["network_id", "year"], how="outer")
    evidence = evidence.merge(sites[["network_id", "current_panel", "site_status", "grts_ngr_easting", "grts_ngr_northing", "grts_wgs84_lat", "grts_wgs84_long", "waterbody_id"]], on="network_id", how="left")
    evidence["evidence_id"] = "rsn:" + evidence["network_id"].astype("string") + ":" + evidence["year"].astype("string")

    for name, table in {"river_sites": sites, "site_year_evidence": evidence, "biology_site_year": pd.concat(biology_long, ignore_index=True, sort=False), "habitat_site_year": habitat_summary, "source_inventory": pd.DataFrame(inventory)}.items():
        path = output_dir / f"rsn_{name}.parquet"
        table.to_parquet(path, index=False)
        paths[name] = path
    return paths
