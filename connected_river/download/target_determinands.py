"""Acquire target determinands missing from the core chemistry downloads."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import get_ident

import pandas as pd

from connected_river.chemistry_registry import additional_target_determinands
from connected_river.download.core_chemistry import RIVER_OBSERVATION_FILTERS
from connected_river.download.water_quality import (
    OBSERVATIONS_ENDPOINT,
    download_observations,
    is_valid_observation_cache,
)
from connected_river.utils.config import resolve_path
from connected_river.utils.manifest import write_manifest_entry

_QUARTERS = (
    ("01-01", "03-31"),
    ("04-01", "06-30"),
    ("07-01", "09-30"),
    ("10-01", "12-31"),
)


def _download_annual_partition(
    *, year: int, notation: str, output_path: Path,
) -> Path:
    """Download one annual partition in shallow, non-overlapping slices."""
    params = {"determinand": notation, **RIVER_OBSERVATION_FILTERS}
    with TemporaryDirectory(
        prefix=f".{output_path.stem}.quarters.", dir=output_path.parent,
    ) as temporary_directory:
        quarter_paths: list[Path] = []
        for number, (start, end) in enumerate(_QUARTERS, start=1):
            quarter_path = Path(temporary_directory) / f"quarter_{number}.parquet"
            download_observations(
                date_from=f"{year}-{start}",
                date_to=f"{year}-{end}",
                output_path=quarter_path,
                extra_params=params,
                limit=2500,
                page_workers=1,
                sleep_s=0.05,
                record_manifest=False,
            )
            quarter_paths.append(quarter_path)
        frames = [pd.read_parquet(path) for path in quarter_paths]
        populated = [frame for frame in frames if len(frame.columns)]
        observations = (
            pd.concat(populated, ignore_index=True).drop_duplicates()
            if populated else pd.DataFrame()
        )

    temporary_output = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.{get_ident()}.part{output_path.suffix}"
    )
    try:
        observations.to_parquet(temporary_output, index=False)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)
    write_manifest_entry(
        "outputs/logs/download_manifest.json",
        dataset="ea_water_quality_observations",
        source_url=OBSERVATIONS_ENDPOINT,
        output_path=output_path,
        metadata={
            "date_from": f"{year}-01-01",
            "date_to": f"{year}-12-31",
            "rows": len(observations),
            "columns": list(observations.columns),
            "extra_params": params,
            "limit": 2500,
            "page_workers": 2,
            "date_slices": [
                {"date_from": f"{year}-{start}", "date_to": f"{year}-{end}"}
                for start, end in _QUARTERS
            ],
        },
    )
    return output_path


# Download each determinand and year separately. Small partitions are easier to
# retry and make missing coverage more obvious than one large file.
def download_additional_target_determinands(
    years: list[int] | tuple[int, ...] | range,
    output_dir: str | Path = "data/raw/water_quality/target_determinands",
    *,
    overwrite: bool = False,
    workers: int = 2,
) -> list[Path]:
    """Download the 18 target determinands not covered by core/metals.

    The exact Water Quality Archive notations follow the study-inspired target
    registry. Year/determinand partitions retain the resumable cache convention
    and are discovered automatically by ingestion.
    """
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = additional_target_determinands()

    def one(year: int, key: str, notation: str) -> Path:
        path = output_dir / f"{year}_{key}_{notation}.parquet"
        if not overwrite and is_valid_observation_cache(path):
            return path
        return _download_annual_partition(
            year=year, notation=notation, output_path=path,
        )

    tasks = [
        (int(year), target.key, target.notation)
        for year in years
        for target in targets
    ]
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, *task) for task in tasks]
        for future in as_completed(futures):
            paths.append(future.result())
    return sorted(paths)
