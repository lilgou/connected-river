from __future__ import annotations

import calendar
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import get_ident
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from connected_river.utils.config import resolve_path
from connected_river.utils.io import ensure_dir
from connected_river.utils.manifest import write_manifest_entry

OBSERVATIONS_ENDPOINT = "https://environment.data.gov.uk/water-quality/data/observation"
PUBLIC_DATA_USER_AGENT = "BuildingTheConnectedRiver/1.0 (+https://environment.data.gov.uk/)"


# Check old downloads before trusting them. An API error page can otherwise
# look like a perfectly ordinary CSV file.
def is_valid_observation_cache(path: str | Path) -> bool:
    """Return whether an existing observation partition is structurally readable."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        if path.suffix.casefold() == ".parquet":
            pq.ParquetFile(path).metadata
            return True
        if path.suffix.casefold() == ".csv":
            try:
                pd.read_csv(path, nrows=1)
            except pd.errors.EmptyDataError:
                # A valid zero-row response may be stored as an empty CSV.
                pass
            return True
    except Exception:
        # Cache validation must treat every parser/footer failure as a miss so
        # the normal downloader can replace the partition atomically.
        return False
    return False


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        # The public endpoint can return a transient 403 when a burst crosses
        # its traffic controls. Retrying slowly is safe because these requests
        # are read-only and final partitions are written atomically.
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": PUBLIC_DATA_USER_AGENT})
    return s


def _request_observations_csv(
    *,
    date_from: str,
    date_to: str,
    limit: int = 2500,
    skip: int = 0,
    extra_params: dict[str, Any] | None = None,
    timeout: int = 180,
) -> pd.DataFrame:
    params = {"dateFrom": date_from, "dateTo": date_to, "limit": limit, "skip": skip}
    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v is not None})

    with _session() as s:
        r = s.post(
            OBSERVATIONS_ENDPOINT,
            params=params,
            headers={"Accept": "text/csv"},
            timeout=timeout,
        )
    r.raise_for_status()

    text = r.text.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), dtype=str)



def _estimate_total_observations(
    *,
    date_from: str,
    date_to: str,
    extra_params: dict[str, Any] | None = None,
    timeout: int = 60,
) -> int | None:
    params = {"dateFrom": date_from, "dateTo": date_to, "limit": 1, "skip": 0}
    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v is not None})

    try:
        with _session() as sess:
            r = sess.get(
                OBSERVATIONS_ENDPOINT,
                params=params,
                headers={"Accept": "application/json"},
                timeout=timeout,
            )
        r.raise_for_status()
        header_total = r.headers.get("x-total-items")
        if header_total is not None:
            return int(header_total)
        js = r.json()
        for key in ["totalItems", "total", "count", "totalResults"]:
            if key in js and js[key] is not None:
                return int(js[key])
        if "meta" in js:
            for key in ["totalItems", "total", "count", "totalResults"]:
                if key in js["meta"] and js["meta"][key] is not None:
                    return int(js["meta"][key])
    except Exception:
        return None

    return None

# The archive can be large, so this works through it in manageable pages and
# writes each completed query to disk before moving on.
def download_observations(
    *,
    date_from: str,
    date_to: str,
    output_path: str | Path,
    extra_params: dict[str, Any] | None = None,
    limit: int = 2500,
    sleep_s: float = 0.0,
    max_pages: int | None = None,
    page_workers: int = 1,
    progress_position: int = 1,
    record_manifest: bool = True,
) -> Path:
    output_path = resolve_path(output_path)
    ensure_dir(output_path.parent)

    chunks: list[pd.DataFrame] = []
    next_skip = 0
    stop = False
    failures: list[tuple[int, Exception]] = []
    pages_done = 0
    total_rows = 0
    estimated_rows = _estimate_total_observations(
        date_from=date_from,
        date_to=date_to,
        extra_params=extra_params,
    )
    estimated_pages = None
    if estimated_rows:
        estimated_pages = (estimated_rows + limit - 1) // limit

    with tqdm(
        total=estimated_pages,
        desc=f"WQ {date_from} to {date_to}",
        unit="page",
        position=progress_position,
        leave=False,
        dynamic_ncols=True,
    ) as pbar:
        with ThreadPoolExecutor(max_workers=page_workers) as ex:
            while not stop:
                batch = []
                for _ in range(page_workers):
                    if max_pages is not None and pages_done + len(batch) >= max_pages:
                        break
                    batch.append(next_skip)
                    next_skip += limit

                if not batch:
                    break

                futs = {
                    ex.submit(
                        _request_observations_csv,
                        date_from=date_from,
                        date_to=date_to,
                        limit=limit,
                        skip=skip,
                        extra_params=extra_params,
                    ): skip
                    for skip in batch
                }

                for fut in as_completed(futs):
                    skip = futs[fut]
                    try:
                        df = fut.result()
                    except Exception as e:
                        print(f"[WARN] failed page skip={skip}: {e}")
                        failures.append((skip, e))
                        stop = True
                        continue

                    pages_done += 1
                    pbar.update(1)

                    if df.empty:
                        stop = True
                        continue

                    chunks.append(df)
                    total_rows += len(df)
                    if estimated_rows:
                        pbar.set_postfix(rows=f"{total_rows:,}/{estimated_rows:,}")
                    else:
                        pbar.set_postfix(rows=f"{total_rows:,}")

                    if len(df) < limit:
                        stop = True

                    if sleep_s:
                        time.sleep(sleep_s)

                    if max_pages is not None and pages_done >= max_pages:
                        stop = True

    if failures:
        failed_skips = ", ".join(str(skip) for skip, _ in failures)
        raise RuntimeError(
            f"Water-quality download failed for page offsets {failed_skips}; "
            "no partial partition was written"
        ) from failures[0][1]

    out = pd.concat(chunks, ignore_index=True).drop_duplicates() if chunks else pd.DataFrame()

    # Write the partition atomically. If a process is interrupted, the final
    # cache path remains absent (or retains its previous valid file) instead of
    # looking complete while containing a truncated Parquet footer.
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.{get_ident()}.part{output_path.suffix}"
    )
    try:
        if output_path.suffix.lower() == ".parquet":
            out.to_parquet(temporary_output, index=False)
        elif output_path.suffix.lower() == ".csv":
            out.to_csv(temporary_output, index=False)
        else:
            raise ValueError("Output path must end in .parquet or .csv")
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    if record_manifest:
        write_manifest_entry(
            "outputs/logs/download_manifest.json",
            dataset="ea_water_quality_observations",
            source_url=OBSERVATIONS_ENDPOINT,
            output_path=output_path,
            metadata={
                "date_from": date_from,
                "date_to": date_to,
                "rows": len(out),
                "columns": list(out.columns),
                "extra_params": extra_params or {},
                "limit": limit,
                "page_workers": page_workers,
            },
        )
    return output_path


def month_bounds(year: int, month: int) -> tuple[str, str]:
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def download_month(
    year: int,
    month: int,
    *,
    output_dir: str | Path = "data/raw/water_quality/monthly",
    extra_params: dict[str, Any] | None = None,
    max_pages: int | None = None,
    overwrite: bool = False,
    limit: int = 2500,
    page_workers: int = 1,
    progress_position: int = 1,
) -> Path:
    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"ea_wq_observations_{year}_{month:02d}.parquet"
    if not overwrite and is_valid_observation_cache(output_path):
        return output_path

    date_from, date_to = month_bounds(year, month)
    return download_observations(
        date_from=date_from,
        date_to=date_to,
        output_path=output_path,
        extra_params=extra_params,
        max_pages=max_pages,
        limit=limit,
        page_workers=page_workers,
        progress_position=progress_position,
    )


def download_year_monthly(
    year: int,
    *,
    output_dir: str | Path = "data/raw/water_quality/monthly",
    extra_params: dict[str, Any] | None = None,
    max_pages_per_month: int | None = None,
    overwrite: bool = False,
    limit: int = 2500,
    page_workers: int = 1,
    month_workers: int = 3,
) -> list[Path]:
    months = list(range(1, 13))
    paths: list[Path] = []

    with ThreadPoolExecutor(max_workers=month_workers) as ex:
        futs = {
            ex.submit(
                download_month,
                year,
                month,
                output_dir=output_dir,
                extra_params=extra_params,
                max_pages=max_pages_per_month,
                overwrite=overwrite,
                limit=limit,
                page_workers=page_workers,
                progress_position=(i % month_workers) + 1,
            ): month
            for i, month in enumerate(months)
        }

        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc=f"WQ months {year}",
            unit="month",
            position=0,
            dynamic_ncols=True,
        ):
            paths.append(fut.result())

    return sorted(paths)


# Monthly files are handy when a download needs to be resumed. The combined
# file is much easier to use during the data-processing stage.
def combine_year(
    year: int,
    *,
    monthly_dir: str | Path = "data/raw/water_quality/monthly",
    output_dir: str | Path = "data/raw/water_quality",
) -> Path:
    monthly_dir = resolve_path(monthly_dir)
    output_dir = ensure_dir(output_dir)
    paths = sorted(monthly_dir.glob(f"ea_wq_observations_{year}_*.parquet"))

    if not paths:
        raise FileNotFoundError(f"No monthly files found for {year} in {monthly_dir}")

    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates()

    output_path = output_dir / f"ea_wq_observations_{year}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path
