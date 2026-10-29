"""Read Environment Agency ecology bulk downloads into a consistent set of tables.

The loaders in this module are file-driven. Downloading is a separate stage,
every input is recorded and archived scripts are not executed.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


@dataclass(frozen=True)
class EcologySource:
    assemblage: str
    zip_name: str
    site_member: str
    observation_member: str
    observation_kind: str


SOURCES = (
    EcologySource("macroinvertebrate", "INV_OPEN_DATA.zip", "INV_OPEN_DATA_SITE.csv", "INV_OPEN_DATA_METRICS.csv", "metrics"),
    EcologySource("diatom", "DIAT_OPEN_DATA.zip", "DIAT_OPEN_DATA_SITE.csv", "DIAT_OPEN_DATA_METRICS.csv", "metrics"),
    EcologySource("macrophyte", "MACP_OPEN_DATA.zip", "MACP_OPEN_DATA_SITE.csv", "MACP_OPEN_DATA_METRICS.csv", "metrics"),
    EcologySource("freshwater_fish", "FW_Fish_Counts.zip", "FW_Fish_Sites.csv", "FW_Fish_Counts.csv", "fish"),
    EcologySource("transitional_coastal_fish", "TRAC_Fish_Counts.zip", "TRAC_Fish_Sites.csv", "TRAC_Fish_Counts.csv", "fish"),
)


def _read_member(path: Path, member: str) -> pd.DataFrame:
    """Read a named CSV, allowing the site table to live in a sibling ZIP."""
    candidates = [path]
    if "_Counts.zip" in path.name and "_Sites.csv" in member:
        candidates.insert(0, path.with_name(path.name.replace("_Counts.zip", "_Sites.zip")))
    for candidate in candidates:
        if not candidate.exists():
            continue
        with zipfile.ZipFile(candidate) as archive:
            matches = [n for n in archive.namelist() if Path(n).name.casefold() == member.casefold()]
            if matches:
                with archive.open(matches[0]) as stream:
                    return pd.read_csv(stream, low_memory=False)
    raise FileNotFoundError(f"Could not find {member!r} in: {', '.join(map(str, candidates))}")


def _series(df: pd.DataFrame, name: str, default: object = pd.NA) -> pd.Series:
    return df[name] if name in df else pd.Series(default, index=df.index)


def _text(df: pd.DataFrame, name: str) -> pd.Series:
    return _series(df, name).astype("string").str.strip()


def _number(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_series(df, name), errors="coerce")


def _date(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_datetime(_series(df, name), dayfirst=True, errors="coerce")


# Different ecology programmes use slightly different source columns. They are
# brought into the same small set of site fields here.
def _site_table(source: EcologySource, raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["assemblage"] = source.assemblage
    out["site_id"] = _text(raw, "SITE_ID")
    out["site_label"] = _series(raw, "SITE_NAME", _series(raw, "SITE_ID"))
    out["catchment"] = _series(raw, "CATCHMENT")
    out["water_body"] = _series(raw, "WATER_BODY")
    out["wfd_waterbody_id"] = _series(raw, "WFD_WATERBODY_ID")
    out["reporting_area"] = _series(raw, "REPORTING_AREA", _series(raw, "NEW_AREA_NAME"))
    out["easting"] = _number(raw, "FULL_EASTING").fillna(_number(raw, "SITE_RANKED_EASTING"))
    out["northing"] = _number(raw, "FULL_NORTHING").fillna(_number(raw, "SITE_RANKED_NORTHING"))
    out["first_sample_date"] = _date(raw, "MIN_SAMPLE_DATE").fillna(_date(raw, "FIRST_SURVEY"))
    out["last_sample_date"] = _date(raw, "MAX_SAMPLE_DATE").fillna(_date(raw, "LAST_SURVEY"))
    out["source_sample_count"] = _number(raw, "COUNT_OF_SAMPLES").fillna(_number(raw, "N_SURVEYS"))
    valid = out["easting"].notna() & out["northing"].notna()
    out["longitude"] = np.nan
    out["latitude"] = np.nan
    if valid.any():
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(out.loc[valid, "easting"], out.loc[valid, "northing"])
        out.loc[valid, "longitude"] = lon
        out.loc[valid, "latitude"] = lat
    return out.dropna(subset=["site_id"]).drop_duplicates(["assemblage", "site_id"])


def _metric_samples(source: EcologySource, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _date(raw, "SAMPLE_DATE")
    samples = pd.DataFrame({
        "assemblage": source.assemblage,
        "site_id": _text(raw, "SITE_ID"),
        "sample_id": _text(raw, "SAMPLE_ID"),
        "analysis_id": _text(raw, "ANALYSIS_ID"),
        "sample_date": dates,
        "sample_year": dates.dt.year.astype("Int64"),
        "sample_method": _series(raw, "SAMPLE_METHOD"),
    }).drop_duplicates()
    metadata = {c for c in raw.columns if any(token in c for token in ("SITE_ID", "SAMPLE_ID", "ANALYSIS_ID", "DATE", "METHOD", "TYPE", "REASON", "VERSION", "REPLICATE", "THIRD_PARTY", "WATERBODY"))}
    value_cols = [c for c in raw.columns if c not in metadata]
    ids = [c for c in ["SITE_ID", "SAMPLE_ID", "ANALYSIS_ID", "SAMPLE_DATE"] if c in raw]
    metrics = raw[ids + value_cols].melt(ids, var_name="metric_name", value_name="metric_value")
    metrics["assemblage"] = source.assemblage
    metrics["site_id"] = _text(metrics, "SITE_ID")
    metrics["sample_id"] = _text(metrics, "SAMPLE_ID")
    metrics["analysis_id"] = _text(metrics, "ANALYSIS_ID")
    metrics["sample_date"] = _date(metrics, "SAMPLE_DATE")
    metrics["metric_value"] = pd.to_numeric(metrics["metric_value"], errors="coerce")
    metrics = metrics[["assemblage", "site_id", "sample_id", "analysis_id", "sample_date", "metric_name", "metric_value"]].dropna(subset=["metric_value"])
    return samples, metrics


def _fish_samples(source: EcologySource, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _date(raw, "EVENT_DATE")
    count = _number(raw, "FISH_COUNT").fillna(_number(raw, "ALL_RUNS"))
    samples = pd.DataFrame({
        "assemblage": source.assemblage,
        "site_id": _text(raw, "SITE_ID"),
        "sample_id": _text(raw, "SURVEY_ID"),
        "analysis_id": _text(raw, "SURVEY_ID"),
        "sample_date": dates,
        "sample_year": _number(raw, "EVENT_DATE_YEAR").fillna(dates.dt.year).astype("Int64"),
        "sample_method": _series(raw, "SURVEY_METHOD"),
    }).drop_duplicates()
    work = raw.assign(_count=count)
    summary = work.groupby(["SITE_ID", "SURVEY_ID"], dropna=False).agg(
        fish_taxa_richness=("SPECIES_ID", "nunique"),
        fish_total_count=("_count", "sum"),
        fish_nonzero_species=("_count", lambda s: int((s.fillna(0) > 0).sum())),
    ).reset_index()
    metrics = summary.melt(["SITE_ID", "SURVEY_ID"], var_name="metric_name", value_name="metric_value")
    metrics["assemblage"] = source.assemblage
    metrics["site_id"] = _text(metrics, "SITE_ID")
    metrics["sample_id"] = _text(metrics, "SURVEY_ID")
    metrics["analysis_id"] = metrics["sample_id"]
    metrics["sample_date"] = pd.NaT
    return samples, metrics[["assemblage", "site_id", "sample_id", "analysis_id", "sample_date", "metric_name", "metric_value"]]


def _taxa_table(source: EcologySource, raw_dir: Path, observations: pd.DataFrame) -> pd.DataFrame:
    if source.observation_kind == "fish":
        raw = observations
        out = pd.DataFrame({
            "assemblage": source.assemblage, "site_id": _text(raw, "SITE_ID"),
            "sample_id": _text(raw, "SURVEY_ID"), "analysis_id": _text(raw, "SURVEY_SPECIES_ID"),
            "taxon_id": _text(raw, "SPECIES_ID"), "taxon_name": _series(raw, "SPECIES_NAME"),
            "latin_name": _series(raw, "LATIN_NAME"),
            "abundance": _number(raw, "FISH_COUNT").fillna(_number(raw, "ALL_RUNS")),
            "percentage_cover": np.nan,
        })
        return out
    prefix = {"macroinvertebrate": "INV", "diatom": "DIAT", "macrophyte": "MACP"}[source.assemblage]
    raw = _read_member(raw_dir / source.zip_name, f"{prefix}_OPEN_DATA_TAXA.csv")
    abundance_col = "TOTAL_ABUNDANCE" if source.assemblage == "macroinvertebrate" else "UNITS_FOUND_OR_SEQUENCE_READS"
    out = pd.DataFrame({
        "assemblage": source.assemblage, "site_id": _text(raw, "SITE_ID"),
        "sample_id": _text(raw, "SAMPLE_ID"), "analysis_id": _text(raw, "ANALYSIS_ID"),
        "taxon_id": _text(raw, "TAXON_LIST_ITEM_KEY"), "abundance": _number(raw, abundance_col),
        "percentage_cover": _number(raw, "PERCENTAGE_COVER"),
    })
    out["taxon_name"] = pd.NA
    out["latin_name"] = pd.NA
    return out


def _taxonomy_lookup(raw_dir: Path) -> pd.DataFrame:
    raw = _read_member(raw_dir / "OPEN_DATA_TAXON_INFO.zip", "OPEN_DATA_TAXON_INFO.csv")
    wanted = {
        "TAXON_LIST_ITEM_KEY": "taxon_id", "PREFERRED_TAXON_NAME": "preferred_taxon_name",
        "TAXON_RANK": "taxon_rank", "TAXON_GROUP_NAME": "taxon_group",
        "NON_NATIVE_SP": "is_non_native", "PROTECTED_TAXA": "is_protected",
    }
    return raw[[c for c in wanted if c in raw]].rename(columns=wanted).drop_duplicates("taxon_id")


def _metric_column(value: object) -> str:
    """Create a stable, self-describing survey metric column name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    return f"biology_metric_{slug}"


def _truthy_taxonomy_flag(values: pd.Series) -> pd.Series:
    """Interpret the publisher's T/F and occasional yes/no taxonomy flags."""
    return values.astype("string").str.casefold().isin({"t", "y", "yes", "true", "1"})


def _attach_taxonomy_traits(
    taxa: pd.DataFrame, taxa_presence: pd.DataFrame,
) -> pd.DataFrame:
    """Attach trait richness without turning absent taxonomy into false zeros."""
    keys = ["assemblage", "site_id", "sample_id"]
    taxonomy_fields = [
        column for column in [
            "preferred_taxon_name", "taxon_group", "is_non_native", "is_protected",
        ]
        if column in taxa
    ]
    mapped = (
        taxa[taxonomy_fields].notna().any(axis=1)
        if taxonomy_fields
        else pd.Series(False, index=taxa.index)
    )
    work = taxa.assign(_taxonomy_mapped=mapped)
    mapping = (
        work.loc[work["_taxonomy_mapped"]]
        .groupby(keys, dropna=False)["taxon_id"]
        .nunique()
        .rename("taxonomy_trait_taxa_mapped")
        .reset_index()
    )
    out = taxa_presence.merge(mapping, on=keys, how="left")
    out["taxonomy_trait_taxa_mapped"] = out["taxonomy_trait_taxa_mapped"].fillna(0).astype("Int64")
    out["taxonomy_trait_coverage"] = (
        pd.to_numeric(out["taxonomy_trait_taxa_mapped"], errors="coerce")
        / pd.to_numeric(out["taxon_richness"], errors="coerce").replace(0, np.nan)
    )

    group_text = work.get(
        "taxon_group", pd.Series(pd.NA, index=work.index)
    ).astype("string").str.casefold()
    non_native = work.get("is_non_native", pd.Series(pd.NA, index=work.index))
    protected = work.get("is_protected", pd.Series(pd.NA, index=work.index))
    trait_masks = {
        "ept_taxon_richness": group_text.str.contains(
            "ephemeroptera|plecoptera|trichoptera", regex=True, na=False,
        ),
        "non_native_taxon_richness": _truthy_taxonomy_flag(non_native),
        "protected_taxon_richness": _truthy_taxonomy_flag(protected),
    }
    for column, mask in trait_masks.items():
        trait = (
            work.loc[mask]
            .groupby(keys, dropna=False)["taxon_id"]
            .nunique()
            .rename(column)
            .reset_index()
        )
        out = out.merge(trait, on=keys, how="left")
        applicable = out["taxonomy_trait_taxa_mapped"].gt(0)
        if column == "ept_taxon_richness":
            applicable &= out["assemblage"].astype("string").eq("macroinvertebrate")
        out.loc[applicable & out[column].isna(), column] = 0
        out.loc[~applicable, column] = pd.NA
        out[column] = out[column].astype("Int64")
    return out


# Build the common tables one programme at a time, then combine them. Keeping
# that split makes it easier to spot a problem in one source dataset.
def build_ecology_index(raw_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Build the sites, samples, metrics and site-year tables."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sites, samples, metrics, taxa, inventory = [], [], [], [], []
    for source in SOURCES:
        try:
            site_raw = _read_member(raw_dir / source.zip_name, source.site_member)
            observation_raw = _read_member(raw_dir / source.zip_name, source.observation_member)
            sites.append(_site_table(source, site_raw))
            sample, metric = (_fish_samples(source, observation_raw) if source.observation_kind == "fish" else _metric_samples(source, observation_raw))
            samples.append(sample)
            metrics.append(metric)
            taxa.append(_taxa_table(source, raw_dir, observation_raw))
            inventory.extend([
                {"assemblage": source.assemblage, "table": "sites", "rows": len(site_raw), "status": "loaded"},
                {"assemblage": source.assemblage, "table": source.observation_kind, "rows": len(observation_raw), "status": "loaded"},
            ])
        except (FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
            inventory.append({"assemblage": source.assemblage, "table": source.observation_kind, "rows": 0, "status": f"unavailable: {exc}"})

    if not samples:
        raise RuntimeError(f"No ecology sources could be loaded from {raw_dir}")
    site_df = pd.concat(sites, ignore_index=True)
    sample_df = pd.concat(samples, ignore_index=True).dropna(subset=["site_id", "sample_id"])
    metric_df = pd.concat(metrics, ignore_index=True)
    taxa_df = pd.concat(taxa, ignore_index=True)
    try:
        taxonomy_df = _taxonomy_lookup(raw_dir)
        taxa_df = taxa_df.merge(taxonomy_df, on="taxon_id", how="left")
        inventory.append({"assemblage": "all", "table": "BIOSYS taxonomy", "rows": len(taxonomy_df), "status": "loaded"})
    except (FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        taxonomy_df = pd.DataFrame()
        inventory.append({"assemblage": "all", "table": "BIOSYS taxonomy", "rows": 0, "status": f"unavailable: {exc}"})
    sample_df = sample_df.merge(site_df, on=["assemblage", "site_id"], how="left")

    metric_presence = metric_df.groupby(["assemblage", "site_id", "sample_id"], dropna=False).agg(
        metric_count=("metric_name", "nunique"), metric_value_count=("metric_value", "count")
    ).reset_index()
    sample_df = sample_df.merge(metric_presence, on=["assemblage", "site_id", "sample_id"], how="left")
    # Preserve the publisher's interpreted WHPT, LIFE, PSI, TDI, RMNI and other
    # indices on the survey row alongside the general metric counts.
    metric_keys = ["assemblage", "site_id", "sample_id", "analysis_id"]
    metric_wide = metric_df.pivot_table(
        index=metric_keys, columns="metric_name", values="metric_value", aggfunc="first",
    )
    metric_wide.columns = [_metric_column(column) for column in metric_wide.columns]
    metric_wide = metric_wide.reset_index()
    sample_df = sample_df.merge(metric_wide, on=metric_keys, how="left", validate="many_to_one")
    taxa_presence = taxa_df.groupby(["assemblage", "site_id", "sample_id"], dropna=False).agg(
        taxon_richness=("taxon_id", "nunique"),
        abundance_total=("abundance", "sum"),
        taxon_record_count=("taxon_id", "size"),
    ).reset_index()
    taxa_presence = _attach_taxonomy_traits(taxa_df, taxa_presence)
    sample_df = sample_df.merge(taxa_presence, on=["assemblage", "site_id", "sample_id"], how="left")
    site_year = sample_df.groupby(["assemblage", "site_id", "sample_year"], dropna=False).agg(
        sample_count=("sample_id", "nunique"),
        first_sample_date_in_year=("sample_date", "min"),
        last_sample_date_in_year=("sample_date", "max"),
        metric_count=("metric_count", "max"),
        metric_value_count=("metric_value_count", "sum"),
        taxon_richness_max=("taxon_richness", "max"),
        abundance_total=("abundance_total", "sum"),
        taxon_record_count=("taxon_record_count", "sum"),
        site_label=("site_label", "first"), catchment=("catchment", "first"),
        water_body=("water_body", "first"), wfd_waterbody_id=("wfd_waterbody_id", "first"),
        easting=("easting", "first"), northing=("northing", "first"),
        latitude=("latitude", "first"), longitude=("longitude", "first"),
    ).reset_index()
    site_year["evidence_id"] = site_year["assemblage"].astype(str) + ":" + site_year["site_id"].astype(str) + ":" + site_year["sample_year"].astype("string")

    tables = {"sites": site_df, "samples": sample_df, "metrics": metric_df, "taxa": taxa_df, "taxonomy": taxonomy_df, "site_year": site_year, "source_inventory": pd.DataFrame(inventory)}
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"ecology_{name}.parquet"
        table.to_parquet(path, index=False)
        paths[name] = path
    return paths
