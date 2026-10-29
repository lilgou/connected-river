"""Download the water-column metal determinands used by the matching workflow.

The Environment Agency codelist contains several analytical forms for each
metal.  Selecting them from the current codelist avoids silently relying on a
single notation that may only be populated in part of the archive.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from connected_river.download.core_chemistry import RIVER_OBSERVATION_FILTERS
from connected_river.download.http import session_with_retries
from connected_river.download.water_quality import (
    download_observations,
    is_valid_observation_cache,
)
from connected_river.utils.config import resolve_path

DETERMINAND_ENDPOINT = "https://environment.data.gov.uk/water-quality/codelist/determinand"
PRIORITY_METALS = (
    "arsenic", "cadmium", "chromium", "copper", "lead", "mercury", "nickel", "zinc",
)
def classify_priority_metal(label: object) -> str | None:
    """Return a target metal for a concentration-form determinand label.

    Exact, dissolved and bioavailable forms are retained and separated during
    harmonisation. Loads, solids, extracts, PNECs and BLM diagnostics that
    merely contain a metal name are intentionally excluded.
    """
    if pd.isna(label):
        return None
    text = str(label).strip().casefold()
    for metal in PRIORITY_METALS:
        if text in {metal, f"{metal}, dissolved", f"{metal}, bioavailable"}:
            return metal
    return None


def fetch_determinand_codelist(
    cache_path: str | Path = "data/raw/water_quality/determinand_codelist.csv",
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch (or reuse) the EA determinand codelist with metal classification."""
    cache_path = resolve_path(cache_path)
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, dtype="string")
        if {"notation", "prefLabel"}.issubset(cached):
            cached["priority_metal"] = cached["prefLabel"].map(classify_priority_metal)
            return cached

    rows: list[dict[str, object]] = []
    skip, limit = 0, 250
    with session_with_retries() as session:
        while True:
            response = session.get(
                DETERMINAND_ENDPOINT,
                params={"skip": skip, "limit": limit},
                headers={"Accept": "application/ld+json", "API-Version": "1"},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            members = payload.get("member") or payload.get("members") or []
            if not members:
                break
            rows.extend({
                "notation": item.get("notation"),
                "prefLabel": item.get("prefLabel"),
                "uri": item.get("@id") or item.get("id"),
            } for item in members)
            if len(members) < limit:
                break
            skip += limit

    result = pd.DataFrame(rows).drop_duplicates(["notation", "prefLabel"])
    if result.empty:
        raise RuntimeError("EA determinand codelist returned no records")
    result["priority_metal"] = result["prefLabel"].map(classify_priority_metal)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    return result


# Metals use the same archive as the core chemistry, but have their own entry
# point so they can be left out of a smaller or more focused run.
def download_priority_metals(
    years: list[int] | tuple[int, ...] | range,
    output_dir: str | Path = "data/raw/water_quality/metals_priority_fast",
    *,
    overwrite: bool = False,
    workers: int = 2,
    refresh_codelist: bool = False,
) -> list[Path]:
    """Download every populated analytical form of the eight priority metals.

    Valid CSV or Parquet caches are reused so completed year partitions are not
    downloaded again.
    """
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    codelist = fetch_determinand_codelist(refresh=refresh_codelist)
    targets = (
        codelist[codelist["priority_metal"].notna()]
        .dropna(subset=["notation"])
        .drop_duplicates(["notation", "priority_metal"])
    )
    if targets.empty:
        raise RuntimeError("No priority water-column metal determinands were identified")

    def one(year: int, metal: str, notation: str) -> Path:
        stem = f"{year}_{metal}_{notation}"
        parquet_path = output_dir / f"{stem}.parquet"
        csv_path = output_dir / f"{stem}.csv"
        if not overwrite:
            if is_valid_observation_cache(parquet_path):
                return parquet_path
            if is_valid_observation_cache(csv_path):
                return csv_path
        return download_observations(
            date_from=f"{year}-01-01",
            date_to=f"{year}-12-31",
            output_path=parquet_path,
            extra_params={"determinand": notation, **RIVER_OBSERVATION_FILTERS},
            limit=2500,
            page_workers=1,
            sleep_s=0.05,
        )

    tasks = [
        (int(year), str(row.priority_metal), str(row.notation))
        for year in years
        for row in targets.itertuples(index=False)
    ]
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, *task) for task in tasks]
        for future in as_completed(futures):
            paths.append(future.result())
    return sorted(paths)
