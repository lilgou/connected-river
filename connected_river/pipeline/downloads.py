"""Stage-1 pipeline: raw data discovery and download.

The modules under ``connected_river.download`` already contain the concrete
connector and pagination logic.  This layer keeps acquisition order and
parameters in one place so other projects can reuse the same process.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from connected_river.download.catalog import acquire_source, load_catalog
from connected_river.download.core_chemistry import download_core_chemistry
from connected_river.download.discharge_permits import download_consented_discharges
from connected_river.download.ecology_bulk import download_all_core_ecology
from connected_river.download.edm_storm_overflows import download_edm_years
from connected_river.download.nrfa import download_nrfa_station_catalogue
from connected_river.download.priority_metals import (
    download_priority_metals,
    fetch_determinand_codelist,
)
from connected_river.download.target_determinands import (
    download_additional_target_determinands,
)
from connected_river.utils.config import load_profile

DEFAULT_RELEASE_CATALOG_SOURCES = (
    "catchment_classifications",
    "catchment_objectives",
    "catchment_rnags",
    "catchment_protected_areas",
    "catchment_geometry",
    "river_habitat_surveys",
    "river_surveillance_network",
)


@dataclass(frozen=True)
class PipelineYears:
    """Year parameters that drive first-stage downloads.

    ``chemistry_years`` is the user-visible archive request window.
    ``lookback_chemistry_years`` is the same range with one extra prior
    year, included to make six-month chemistry windows complete for the first
    matched year.
    """

    online_start_year: int
    target_year: int
    public_extension_start_year: int
    chemistry_years: tuple[int, ...]
    chemistry_years_with_lookback: tuple[int, ...]


@dataclass(frozen=True)
class DownloadArtifacts:
    """Material downloaded by the stage-1 pipeline."""

    profile: str
    target_year: int
    online_start_year: int
    public_extension_start_year: int
    chemistry_years: tuple[int, ...]
    chemistry_years_with_lookback: tuple[int, ...]
    downloaded: dict[str, list[Path] | dict[str, Any]]


def _partition_refresh_plan(
    years: list[int] | tuple[int, ...] | range,
    *,
    refresh_recent_years: int,
    overwrite: bool,
) -> tuple[list[int], list[int]]:
    """Split stable cached years from partitions that must be downloaded again."""
    ordered = sorted({int(year) for year in years})
    if refresh_recent_years < 0:
        raise ValueError("refresh_recent_years must be non-negative")
    if overwrite:
        return [], ordered
    recent_count = min(refresh_recent_years, len(ordered))
    if recent_count == 0:
        return ordered, []
    return ordered[:-recent_count], ordered[-recent_count:]


def _download_partitioned(
    downloader,
    years: list[int] | tuple[int, ...] | range,
    *,
    refresh_recent_years: int,
    overwrite: bool,
    **kwargs,
) -> list[Path]:
    """Reuse old partitions and atomically renew only the declared recent window."""
    stable_years, refreshed_years = _partition_refresh_plan(
        years,
        refresh_recent_years=refresh_recent_years,
        overwrite=overwrite,
    )
    paths: list[Path] = []
    if stable_years:
        paths.extend(downloader(stable_years, overwrite=False, **kwargs))
    if refreshed_years:
        paths.extend(downloader(refreshed_years, overwrite=True, **kwargs))
    return sorted(set(paths))


def configure_years(
    profile: str, target_year: int | None = None,
    chemistry_years: list[int] | tuple[int, ...] | None = None,
) -> PipelineYears:
    config = load_profile(profile)
    run_config = config.get("run", {})

    online_start_year = int(run_config.get("online_matching_start_year", 2018))
    default_target = int(run_config.get("target_year", datetime.now(timezone.utc).year - 1))
    effective_target_year = int(target_year if target_year is not None else default_target)
    if effective_target_year < online_start_year:
        raise ValueError("target_year must be >= online_matching_start_year")

    public_extension_start_year = int(run_config.get("public_extension_start_year", 2018))
    if chemistry_years is None:
        chemistry_years = tuple(range(online_start_year, effective_target_year + 1))
    else:
        chemistry_years = tuple(sorted({int(value) for value in chemistry_years}))
        if not chemistry_years:
            raise ValueError("chemistry_years must be non-empty")
        if chemistry_years[0] < online_start_year:
            raise ValueError("chemistry_years cannot include years before online start")
        if chemistry_years[-1] != effective_target_year:
            if effective_target_year > chemistry_years[-1]:
                chemistry_years = tuple(range(chemistry_years[0], effective_target_year + 1))
            else:
                raise ValueError("target_year must not be smaller than chemistry_years")

    lookback_chemistry_years = tuple(
        range(min(chemistry_years) - 1, max(chemistry_years) + 1)
    )
    return PipelineYears(
        online_start_year=online_start_year,
        target_year=effective_target_year,
        public_extension_start_year=public_extension_start_year,
        chemistry_years=chemistry_years,
        chemistry_years_with_lookback=lookback_chemistry_years,
    )


def run_download_stage(
    profile: str = "connected_river",
    *,
    target_year: int | None = None,
    chemistry_years: list[int] | tuple[int, ...] | None = None,
    overwrite: bool = False,
    refresh_snapshot_sources: bool = False,
    refresh_recent_years: int = 0,
    extract_ecology: bool = False,
    convert_ecology_to_parquet: bool = False,
) -> DownloadArtifacts:
    """Download all repeatable data required by the complete workflow.

    This keeps the acquisition sequence in one place for reuse by new projects,
    while preserving the existing download modules and their settings. Set
    ``refresh_snapshot_sources=True`` and ``refresh_recent_years=2`` for the
    normal latest-data workflow. ``overwrite=True`` re-snapshots all years.
    """

    years = configure_years(profile, target_year=target_year, chemistry_years=chemistry_years)

    downloads: dict[str, list[Path] | dict[str, Any]] = {}

    snapshot_overwrite = overwrite or refresh_snapshot_sources

    downloads["ecology_bulk"] = download_all_core_ecology(
        overwrite=snapshot_overwrite,
        extract=extract_ecology,
        convert_parquet=convert_ecology_to_parquet,
    )

    catalog = load_catalog()
    acquired_catalog_sources: list[Path] = []
    for source_id in DEFAULT_RELEASE_CATALOG_SOURCES:
        acquired_catalog_sources.extend(acquire_source(
            catalog[source_id], overwrite=snapshot_overwrite,
        ))
    downloads["catalog_sources"] = acquired_catalog_sources

    downloads["consented_discharges"] = download_consented_discharges(
        overwrite=snapshot_overwrite,
    )

    downloads["nrfa_station_catalogue"] = [
        download_nrfa_station_catalogue(overwrite=snapshot_overwrite)
    ]

    downloads["core_chemistry"] = _download_partitioned(
        download_core_chemistry,
        years.chemistry_years_with_lookback,
        refresh_recent_years=refresh_recent_years,
        overwrite=overwrite,
        workers=2,
    )
    # Refresh the current codelist once, then reuse the workspace cache for
    # both stable and recently refreshed year groups. Refreshing it inside
    # both partition calls needlessly repeats thousands of paginated records.
    if snapshot_overwrite:
        fetch_determinand_codelist(refresh=True)
    downloads["priority_metals"] = _download_partitioned(
        download_priority_metals,
        years.chemistry_years_with_lookback,
        refresh_recent_years=refresh_recent_years,
        overwrite=overwrite,
        workers=2,
        refresh_codelist=False,
    )

    # Acquire each requested year for target determinands outside the core and
    # priority-metals download groups.
    downloads["additional_determinand_partitions"] = (
        _download_partitioned(
            download_additional_target_determinands,
            years.chemistry_years,
            refresh_recent_years=refresh_recent_years,
            overwrite=overwrite,
            workers=2,
        )
        if years.chemistry_years
        else []
    )

    edm_start_year = 2020
    if years.target_year >= edm_start_year:
        downloads["edm_years"] = _download_partitioned(
            download_edm_years,
            range(edm_start_year, years.target_year + 1),
            refresh_recent_years=refresh_recent_years,
            overwrite=overwrite,
        )
    else:
        downloads["edm_years"] = []

    return DownloadArtifacts(
        profile=profile,
        target_year=years.target_year,
        online_start_year=years.online_start_year,
        public_extension_start_year=years.public_extension_start_year,
        chemistry_years=years.chemistry_years,
        chemistry_years_with_lookback=years.chemistry_years_with_lookback,
        downloaded=downloads,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire the complete reusable source layer without building a release."
    )
    parser.add_argument("--profile", default="connected_river")
    parser.add_argument("--target-year", type=int, required=True)
    parser.add_argument(
        "--latest", action="store_true",
        help="Refresh snapshot sources and the newest two annual partitions",
    )
    parser.add_argument(
        "--refresh-recent-years", type=int, default=0,
        help="Number of newest partition years to redownload",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Redownload every historical partition and snapshot",
    )
    args = parser.parse_args()
    recent_years = max(args.refresh_recent_years, 2 if args.latest else 0)
    result = run_download_stage(
        args.profile,
        target_year=args.target_year,
        overwrite=args.overwrite,
        refresh_snapshot_sources=args.latest,
        refresh_recent_years=recent_years,
    )
    print(
        f"Acquired {sum(len(paths) for paths in result.downloaded.values()):,} "
        f"source files for target year {result.target_year}."
    )


if __name__ == "__main__":
    main()
