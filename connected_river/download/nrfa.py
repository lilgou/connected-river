"""Resumable National River Flow Archive daily-flow acquisition."""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from threading import get_ident
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from connected_river.download.http import session_with_retries
from connected_river.utils.config import resolve_path
from connected_river.utils.manifest import write_manifest_entry

NRFA_STATION_INFO_URL = "https://nrfaapps.ceh.ac.uk/nrfa/ws/station-info"
NRFA_TIME_SERIES_URL = "https://nrfaapps.ceh.ac.uk/nrfa/ws/time-series"
NRFA_STATION_FIELDS = (
    "id", "name", "catchment-area", "easting", "northing", "latitude",
    "longitude", "river", "location", "gdf-statistics", "category",
)
NRFA_STATION_REQUIRED_COLUMNS = {
    "id", "name", "catchment-area", "easting", "northing", "river",
    "gdf-mean-flow", "gdf-percent-complete",
}


def is_valid_station_catalogue(path: str | Path) -> bool:
    path = resolve_path(path)
    if not path.is_file():
        return False
    try:
        metadata = pq.ParquetFile(path)
        return (
            metadata.metadata.num_rows > 0
            and NRFA_STATION_REQUIRED_COLUMNS <= set(metadata.schema.names)
        )
    except Exception:
        return False


# The station list is downloaded separately from the flow records because it
# is also used to decide which gauges are relevant to the river network.
def fetch_nrfa_station_catalogue(*, timeout: int = 180) -> pd.DataFrame:
    """Fetch the current public NRFA station catalogue in one API request."""
    with session_with_retries() as session:
        response = session.get(
            NRFA_STATION_INFO_URL,
            params={
                "station": "*",
                "format": "csv",
                "fields": ",".join(NRFA_STATION_FIELDS),
            },
            timeout=timeout,
        )
        response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    missing = sorted(NRFA_STATION_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(
            "The NRFA station catalogue is missing required columns: "
            + ", ".join(missing)
        )
    frame["id"] = pd.to_numeric(frame["id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["id"]).copy()
    frame["id"] = frame["id"].astype("int64")
    if frame.empty:
        raise RuntimeError("The NRFA station catalogue returned no valid stations")
    if frame["id"].duplicated().any():
        duplicate_ids = frame.loc[frame["id"].duplicated(), "id"].astype(str).tolist()
        raise RuntimeError(
            "The NRFA station catalogue contains duplicate station IDs: "
            + ", ".join(duplicate_ids[:10])
        )
    return frame.sort_values("id").reset_index(drop=True)


def download_nrfa_station_catalogue(
    output_path: str | Path = "data/raw/nrfa/nrfa_station_catalogue.parquet",
    *,
    overwrite: bool = False,
) -> Path:
    """Download and safely cache the metadata used to locate NRFA gauges."""
    output_path = resolve_path(output_path)
    if not overwrite and is_valid_station_catalogue(output_path):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_nrfa_station_catalogue()
    temporary = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.part{output_path.suffix}"
    )
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    write_manifest_entry(
        "outputs/logs/download_manifest.json",
        dataset="nrfa_station_catalogue",
        source_url=NRFA_STATION_INFO_URL,
        output_path=output_path,
        metadata={
            "records": len(frame),
            "fields": list(NRFA_STATION_FIELDS),
            "publisher_attribution": "Data from the UK National River Flow Archive",
            "licence": "NRFA Data Terms and Conditions for API access",
        },
    )
    return output_path


def is_valid_flow_cache(path: str | Path) -> bool:
    path = resolve_path(path)
    if not path.is_file():
        return False
    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:
        return False


def parse_nrfa_csv(text: str) -> pd.DataFrame:
    """Parse NRFA CSV-with-metadata while retaining quality flags."""
    lines = text.splitlines()
    start = next((index + 1 for index, line in enumerate(lines) if line.startswith("data,last,")), None)
    if start is None:
        return pd.DataFrame(columns=["date", "flow_m3s", "quality_flag"])
    rows = []
    for fields in csv.reader(lines[start:]):
        if not fields or not fields[0][:4].isdigit():
            continue
        if len(fields) >= 2:
            rows.append((fields[0], fields[1], fields[2] if len(fields) > 2 else None))
    frame = pd.DataFrame(rows, columns=["date", "flow_m3s", "quality_flag"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["flow_m3s"] = pd.to_numeric(frame["flow_m3s"], errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")


def fetch_nrfa_daily_flow(station_id: int, *, timeout: int = 120) -> pd.DataFrame:
    with session_with_retries() as session:
        response = session.get(
            NRFA_TIME_SERIES_URL,
            params={"station": int(station_id), "data-type": "gdf", "format": "nrfa-csv"},
            timeout=timeout,
        )
        response.raise_for_status()
    frame = parse_nrfa_csv(response.text)
    frame["nrfa_station_id"] = int(station_id)
    return frame


def download_nrfa_flows(
    station_ids: list[int] | tuple[int, ...], output_dir: str | Path = "data/raw/nrfa/flows",
    *, overwrite: bool = False, workers: int = 6,
) -> list[Path]:
    """Download the requested gauges, saving one retryable file per station."""
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def one(station_id: int) -> Path:
        path = output_dir / f"nrfa_gdf_{int(station_id)}.parquet"
        if not overwrite and is_valid_flow_cache(path):
            return path
        frame = fetch_nrfa_daily_flow(int(station_id))
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.{get_ident()}.part{path.suffix}"
        )
        try:
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        write_manifest_entry(
            "outputs/logs/download_manifest.json", dataset="nrfa_gauged_daily_flow",
            source_url=NRFA_TIME_SERIES_URL, output_path=path,
            metadata={"station_id": int(station_id), "rows": len(frame)},
        )
        return path

    paths = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, int(station_id)): int(station_id) for station_id in sorted(set(station_ids))}
        for future in as_completed(futures):
            paths.append(future.result())
    return sorted(paths)


# One file per gauge makes failed downloads easy to retry. This final step
# brings the successful files back together for matching and analysis.
def combine_nrfa_flow_partitions(
    input_dir: str | Path = "data/raw/nrfa/flows",
    output_path: str | Path = "data/raw/nrfa/nrfa_gdf_matched_stations.parquet",
) -> Path:
    """Rebuild the standard NRFA flow table after an annual partition refresh."""
    input_dir = resolve_path(input_dir)
    paths = [
        path for path in sorted(input_dir.glob("nrfa_gdf_*.parquet"))
        if is_valid_flow_cache(path)
    ]
    if not paths:
        raise FileNotFoundError(f"No NRFA daily-flow partitions found in {input_dir}")
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.part{output_path.suffix}")
    sql_paths = ", ".join(
        "'" + str(path.resolve()).replace("'", "''") + "'" for path in paths
    )
    connection = duckdb.connect()
    try:
        connection.execute("PRAGMA threads=4")
        connection.execute("PRAGMA preserve_insertion_order=false")
        destination = str(temporary.resolve()).replace("'", "''")
        connection.execute(f"""
            COPY (
                SELECT * EXCLUDE (_duplicate_rank)
                FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY nrfa_station_id, date ORDER BY date
                    ) AS _duplicate_rank
                    FROM read_parquet([{sql_paths}], union_by_name=true)
                )
                WHERE _duplicate_rank = 1
                ORDER BY nrfa_station_id, date
            ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        temporary.replace(output_path)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)
    return output_path
