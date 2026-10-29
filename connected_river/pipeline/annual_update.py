"""Refresh changing inputs and build a versioned connected-river release."""

from __future__ import annotations

import argparse
from pathlib import Path

from connected_river.pipeline.downloads import run_download_stage
from connected_river.pipeline.release import build_release
from connected_river.utils.config import load_profile


# An annual update uses the same full pipeline with a later end year. Keeping a
# single route avoids a separate refresh process drifting from the main method.
def annual_update(
    target_year: int, *, online_start_year: int | None = None,
    public_extension_start_year: int | None = None,
    overwrite: bool = False, refresh_recent_years: int = 2,
    run_root: str | Path = "runs",
) -> Path:
    """Renew changing sources, add ``target_year`` and build a new release.

    Existing historical partitions are reused. The newest two chemistry years
    are downloaded again by default so revised observations and the preceding
    year needed by January-June windows are not hidden by a valid-but-stale
    cache. ``overwrite=True`` instead re-snapshots the full online history.
    Snapshot sources and the annual quality-controlled NRFA archive are always
    refreshed, and gauges are selected independently for every site-year.
    """
    target_year = int(target_year)
    run_config = load_profile("connected_river").get("run", {})
    online_start_year = int(
        online_start_year
        if online_start_year is not None
        else run_config.get("online_matching_start_year", 2018)
    )
    public_extension_start_year = int(
        public_extension_start_year
        if public_extension_start_year is not None
        else run_config.get("public_extension_start_year", 2018)
    )
    if target_year < online_start_year:
        raise ValueError("target_year must be at least online_start_year")

    run_download_stage(
        "connected_river",
        target_year=target_year,
        chemistry_years=list(range(online_start_year, target_year + 1)),
        overwrite=overwrite,
        refresh_snapshot_sources=True,
        refresh_recent_years=refresh_recent_years,
    )
    return build_release(
        years=list(range(online_start_year, target_year + 1)),
        public_extension_start_year=public_extension_start_year,
        run_root=run_root, refresh_flow=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the connected-river dataset and append a complete year.")
    parser.add_argument("--target-year", required=True, type=int, help="New complete monitoring year to append")
    parser.add_argument(
        "--online-start-year", type=int,
        help="Earliest common online biology/chemistry year (profile default: 2018)",
    )
    parser.add_argument(
        "--public-extension-start-year", type=int,
        help="First biological survey year included in the automated product (profile default: 2018)",
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument(
        "--refresh-recent-years", type=int, default=2,
        help="Redownload this many newest partition years (default: 2)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Redownload the full historical archive instead of only recent partitions",
    )
    args = parser.parse_args()
    print(annual_update(
        args.target_year,
        online_start_year=args.online_start_year,
        public_extension_start_year=args.public_extension_start_year,
        overwrite=args.overwrite,
        refresh_recent_years=args.refresh_recent_years,
        run_root=args.run_root,
    ))


if __name__ == "__main__":
    main()
