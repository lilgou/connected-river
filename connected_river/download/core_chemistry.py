"""Targeted downloads for river-health chemistry required by interpretation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from connected_river.download.water_quality import (
    download_observations,
    is_valid_observation_cache,
)
from connected_river.utils.config import resolve_path

CORE_DETERMINANDS = {
    "ammoniacal_n": "0111", "nitrate_n": "0117", "orthophosphate_p": "0180",
    "dissolved_oxygen": "9924", "ph": "0061", "water_temperature": "0076",
    "bod_5day_atu": "0085", "suspended_solids": "0135", "alkalinity": "0162",
}

# Official Water Quality Archive codelist codes. Applying these at source avoids
# downloading large volumes of effluent, groundwater, lake and sediment records
# that the river pipeline would discard during harmonisation.
RIVER_OBSERVATION_FILTERS = {
    "samplingPointType": "F6",       # FRESHWATER - RIVERS
    "sampleMaterialType": "2AZZ",   # RIVER / RUNNING SURFACE WATER
}


def download_core_chemistry(
    years: list[int] | tuple[int, ...], output_dir: str | Path = "data/raw/water_quality/core_chemistry",
    *, overwrite: bool = False, workers: int = 2,
) -> list[Path]:
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def one(year: int, name: str, notation: str) -> Path:
        path = output_dir / f"{year}_{name}_{notation}.parquet"
        if not overwrite and is_valid_observation_cache(path):
            return path
        return download_observations(
            date_from=f"{year}-01-01", date_to=f"{year}-12-31", output_path=path,
            extra_params={"determinand": notation, **RIVER_OBSERVATION_FILTERS},
            limit=2500, page_workers=1, sleep_s=0.05,
        )

    tasks = [(year, name, notation) for year in years for name, notation in CORE_DETERMINANDS.items()]
    paths = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, *task) for task in tasks]
        for future in as_completed(futures):
            paths.append(future.result())
    return sorted(paths)
